"""Desenho de som da cena (H6): cada som nasce de um evento do programa de cena."""
import numpy as np

SR = 24000


def ruido(n, lo, hi, rng, inclin=0.5):
    X = np.fft.rfft(rng.standard_normal(n))
    f = np.fft.rfftfreq(n, 1 / SR)
    X *= ((f > lo) & (f < hi)) / np.maximum(f, 20) ** inclin
    y = np.fft.irfft(X, n)
    return y / (np.abs(y).max() + 1e-9)


def passa_banda(x, lo, hi):
    X = np.fft.rfft(x); f = np.fft.rfftfreq(len(x), 1 / SR)
    X *= 1 / (1 + (lo / np.maximum(f, 1)) ** 4) / (1 + (f / hi) ** 4)
    return np.fft.irfft(X, len(x))


def radio(voz, rng):
    """Voz pelo rádio do traje: banda telefônica, leve saturação, chiado, bips de abertura/fechamento."""
    v = passa_banda(voz, 320, 3400)
    v = np.tanh(v / (np.abs(v).max() + 1e-9) * 2.2) / np.tanh(2.2)
    chiado = ruido(len(v), 1500, 7000, rng, 0) * 0.025
    bip = lambda f, d: np.sin(2 * np.pi * f * np.arange(int(d * SR)) / SR) * 0.25 * np.hanning(int(d * SR))
    abre = np.concatenate([ruido(int(0.06 * SR), 800, 6000, rng, 0) * 0.15, bip(1400, 0.07), np.zeros(int(0.08 * SR))])
    fecha = np.concatenate([np.zeros(int(0.06 * SR)), bip(1000, 0.06), ruido(int(0.12 * SR), 800, 6000, rng, 0) * 0.12 * np.linspace(1, 0, int(0.12 * SR))])
    return np.concatenate([abre, v * 0.8 + chiado, fecha])


def respiracao(dur, rng, ciclo=3.6):
    n = int(dur * SR); t = np.arange(n) / SR
    fase = (t % ciclo) / ciclo
    env = np.where(fase < 0.4, np.sin(np.pi * fase / 0.4) * 0.6, np.where(fase < 0.85, np.sin(np.pi * (fase - 0.4) / 0.45), 0))
    corpo = ruido(n, 200, 2500, rng, 0.8)
    return corpo * env ** 1.5 * 0.18


def vento(dur, rng):
    n = int(dur * SR); t = np.arange(n) / SR
    raj = 0.5 + 0.35 * np.sin(2 * np.pi * 0.09 * t + 1) + 0.15 * np.sin(2 * np.pi * 0.23 * t)
    assobio = ruido(n, 600, 1400, rng, 0) * (0.3 + 0.7 * np.sin(2 * np.pi * 0.05 * t) ** 2)
    return (ruido(n, 40, 600, rng, 1.0) * 0.8 + assobio * 0.12) * raj * 0.22


def passo(rng):
    """Bota no cascalho: grãos curtos de ruído filtrado (síntese granular)."""
    n = int(0.35 * SR); y = np.zeros(n)
    for _ in range(40):
        p = int(abs(rng.normal(0.03, 0.04)) * SR)
        g = int(rng.uniform(0.002, 0.008) * SR)
        if p + g < n:
            y[p:p + g] += rng.standard_normal(g) * np.hanning(g) * rng.uniform(0.2, 1)
    baque = np.sin(2 * np.pi * 70 * np.arange(n) / SR) * np.exp(-np.arange(n) / SR * 30) * 0.6
    return passa_banda(y, 300, 6000) * 0.5 + baque


def zumbido(dur, inicio, rng):
    n = int(dur * SR); t = np.arange(n) / SR
    env = np.clip((t - inicio) / 6, 0, 1) ** 2
    y = sum(a * np.sin(2 * np.pi * f * t + 0.3 * np.sin(2 * np.pi * 0.3 * t)) for f, a in ((55, 1), (82.5, .5), (110.7, .35), (220.3, .12)))
    pulso = 0.7 + 0.3 * np.sin(2.3 * t)          # mesmo pulso do brilho do monólito
    return y * env * pulso * 0.12


def trilha(dur, crescendo_em):
    """Cordas/metais graves (serrotes filtrados) em ré menor, com crescendo no final."""
    n = int(dur * SR); t = np.arange(n) / SR
    acordes = [(0, (73.4, 110, 146.8, 174.6)), (12, (58.3, 87.3, 116.5, 174.6)), (24, (65.4, 98, 130.8, 164.8)),
               (crescendo_em, (73.4, 110, 146.8, 220, 293.7))]
    y = np.zeros(n)
    for i, (t0, notas) in enumerate(acordes):
        t1 = acordes[i + 1][0] if i + 1 < len(acordes) else dur
        m = (t >= t0 - 1.5) & (t < t1 + 1.5)
        env = np.clip((t - t0 + 1.5) / 3, 0, 1) * np.clip((t1 + 1.5 - t) / 3, 0, 1)
        for f in notas:
            for det in (0.997, 1.0, 1.004):
                fase = (f * det * t) % 1
                y += np.where(m, (2 * fase - 1) * env, 0) * 0.04
    y = passa_banda(y, 30, 900)
    vol = 0.5 + 0.5 * np.clip((t - crescendo_em) / 8, 0, 1) * 1.8
    return y * vol * np.clip((dur - t) / 3, 0, 1)
