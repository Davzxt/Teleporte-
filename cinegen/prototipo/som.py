"""H6 — Foley por física + ambiente + trilha + mixagem espacial."""
import wave
import numpy as np

SR = 22050


def ruido_banda(n, f_lo, f_hi, rng):
    X = np.fft.rfft(rng.standard_normal(n))
    f = np.fft.rfftfreq(n, 1 / SR)
    X *= ((f > f_lo) & (f < f_hi)) / np.sqrt(np.maximum(f, 20) / 100)
    y = np.fft.irfft(X, n)
    return y / (np.abs(y).max() + 1e-9)


def vento(dur, rng):
    n = int(dur * SR)
    t = np.arange(n) / SR
    rajada = 0.55 + 0.3 * np.sin(2 * np.pi * 0.13 * t + 1) + 0.15 * np.sin(2 * np.pi * 0.31 * t)
    return ruido_banda(n, 120, 900, rng) * rajada * 0.12


def batida_madeira(rng, forca=1.0):
    """Síntese modal: modos de ressonância de uma porta de madeira."""
    n = int(0.45 * SR)
    t = np.arange(n) / SR
    modos = [(96, 1.0, 18), (212, 0.7, 25), (377, 0.5, 35), (618, 0.35, 50), (1010, 0.2, 70), (1630, 0.1, 90)]
    y = sum(a * np.exp(-d * t) * np.sin(2 * np.pi * f * rng.uniform(.98, 1.02) * t) for f, a, d in modos)
    clique = rng.standard_normal(n) * np.exp(-t * 400) * 0.4
    return (y + clique) * forca * 0.5


def trilha(dur):
    """Drone grave em lá menor, entra e sai devagar."""
    n = int(dur * SR)
    t = np.arange(n) / SR
    y = np.zeros(n)
    for f, a in ((55, .5), (110, .35), (130.8, .2), (164.8, .18), (220, .08)):
        y += a * np.sin(2 * np.pi * f * t + 0.3 * np.sin(2 * np.pi * 0.07 * t))
        y += a * 0.5 * np.sin(2 * np.pi * f * 1.003 * t)
    env = np.clip(t / 3, 0, 1) * np.clip((dur - t) / 2, 0, 1)
    return y * env * 0.05


def reverb(x, dur_ir=0.7, mix=0.18, seed=3):
    rng = np.random.default_rng(seed)
    n = int(dur_ir * SR)
    ir = rng.standard_normal(n) * np.exp(-np.arange(n) / SR * 7)
    m = len(x) + n
    y = np.fft.irfft(np.fft.rfft(x, m) * np.fft.rfft(ir, m), m)[:len(x)]
    y /= np.abs(y).max() + 1e-9
    return (1 - mix) * x + mix * y * np.abs(x).max()


class Mixer:
    def __init__(self, dur):
        self.L = np.zeros(int(dur * SR) + SR)
        self.R = np.zeros_like(self.L)

    def add(self, sinal, t0, pan=0.0, ganho=1.0):
        """pan vem da posição do objeto em relação à câmera (-1 esquerda, +1 direita)."""
        i = int(t0 * SR)
        s = sinal[: max(len(self.L) - i, 0)] * ganho
        th = (np.clip(pan, -1, 1) + 1) * np.pi / 4
        self.L[i:i + len(s)] += s * np.cos(th)
        self.R[i:i + len(s)] += s * np.sin(th)

    def salvar(self, caminho, dur):
        n = int(dur * SR)
        st = np.stack([reverb(self.L[:n]), reverb(self.R[:n], seed=4)], 1)
        st = st / (np.abs(st).max() + 1e-9) * 0.9
        with wave.open(caminho, "wb") as w:
            w.setnchannels(2); w.setsampwidth(2); w.setframerate(SR)
            w.writeframes((st * 32767).astype("<i2").tobytes())
