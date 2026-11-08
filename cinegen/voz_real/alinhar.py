"""Alinhamento fonético automático por HMM com auto-treino (Viterbi, início plano).

Ninguém anotou onde cada fonema começa: o modelo descobre sozinho, iterando
  segmentar → estimar modelo de cada fonema → re-segmentar.
"""
import os
import sys
import json
import numpy as np

AQUI = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.join(AQUI, "..", "prototipo"))
import voz  # noqa: E402  (g2p em português)
import corpus  # noqa: E402

HOP = 240          # 10 ms a 24 kHz
JAN = 600          # 25 ms
SUB = 3            # sub-estados por fonema (início, meio, fim)


def _mel(n_fft, sr, n_mel=26):
    f = np.linspace(0, sr / 2, n_fft // 2 + 1)
    mel = lambda h: 2595 * np.log10(1 + h / 700)
    pts = 700 * (10 ** (np.linspace(mel(60), mel(sr / 2 - 200), n_mel + 2) / 2595) - 1)
    B = np.zeros((n_mel, len(f)))
    for i in range(n_mel):
        a, c, b = pts[i], pts[i + 1], pts[i + 2]
        B[i] = np.clip(np.minimum((f - a) / (c - a), (b - f) / (b - c)), 0, None)
    return B


_MEL = _mel(1024, corpus.SR)
_DCT = np.cos(np.pi / 26 * (np.arange(26) + 0.5)[None] * np.arange(13)[:, None])


def mfcc(x):
    n = 1 + (len(x) - JAN) // HOP
    idx = np.arange(JAN)[None] + HOP * np.arange(n)[:, None]
    fr = x[idx] * np.hanning(JAN)
    P = np.abs(np.fft.rfft(fr, 1024)) ** 2
    lm = np.log(P @ _MEL.T + 1e-8)
    c = lm @ _DCT.T
    energia = lm.mean(1, keepdims=True)
    c = np.concatenate([c[:, 1:], energia], 1)
    c = c - c.mean(0)
    d = np.gradient(c, axis=0)
    return np.concatenate([c, d], 1)


def fonemas(texto):
    """Sequência com pausa opcional ('sp') entre palavras e silêncio ('_') nas pontas."""
    seq = ["_"]
    for w in texto.lower().replace("-", " ").replace(".", "").split():
        seq += voz.g2p_palavra(w) + ["sp"]
    seq[-1] = "_"
    return seq


def _grafo(seq):
    """Estados expandidos e, para cada estado, de quem ele pode vir (permite pular 'sp')."""
    est, de_onde, dono = [], [], []
    for i, p in enumerate(seq):
        nsub = 1 if p == "sp" else SUB
        for s in range(nsub):
            est.append((p, s)); dono.append(i)
    for j, (p, s) in enumerate(est):
        prev = [j] + ([j - 1] if j > 0 else [])
        if s == 0 and j >= 2 and est[j - 1][0] == "sp":   # pode pular a pausa opcional
            prev.append(j - 2)
        de_onde.append(prev)
    return est, de_onde, dono


def viterbi(F, est, de_onde, modelos):
    T, S = len(F), len(est)
    ll = np.stack([modelos[e].logp(F) for e in est], 1)               # (T, S)
    D = np.full((T, S), -np.inf); B = np.zeros((T, S), int)
    D[0, 0] = ll[0, 0]
    for t in range(1, T):
        for j in range(S):
            cand = [D[t - 1, i] for i in de_onde[j]]
            k = int(np.argmax(cand))
            D[t, j] = cand[k] + ll[t, j]; B[t, j] = de_onde[j][k]
    j = S - 1; cam = [j]
    for t in range(T - 1, 0, -1):
        j = B[t, j]; cam.append(j)
    return np.array(cam[::-1])


class Gauss:
    def __init__(self, X, piso):
        self.m = X.mean(0); self.v = np.maximum(X.var(0), piso)

    def logp(self, F):
        return -0.5 * (((F - self.m) ** 2) / self.v + np.log(2 * np.pi * self.v)).sum(1)


def treinar(iters=8):
    audios = corpus.carregar()
    Fs = [mfcc(a) for a in audios]
    seqs = [fonemas(t) for t in corpus.FRASES]
    grafos = [_grafo(s) for s in seqs]
    piso = 0.05 * np.concatenate(Fs).var(0)
    # início plano: divide cada frase igualmente entre os estados (ignorando pausas opcionais)
    caminhos = []
    for F, (est, _, _) in zip(Fs, grafos):
        uteis = [j for j, e in enumerate(est) if e[0] != "sp"]
        caminhos.append(np.array(uteis)[np.minimum((np.arange(len(F)) * len(uteis)) // len(F), len(uteis) - 1)])
    for it in range(iters):
        dados = {}
        for F, cam, (est, _, _) in zip(Fs, caminhos, grafos):
            for t, j in enumerate(cam):
                dados.setdefault(est[j], []).append(F[t])
        modelos = {k: Gauss(np.array(v), piso) for k, v in dados.items()}
        for (p, s) in {e for g in grafos for e in g[0]}:          # estado sem dados: herda vizinho
            if (p, s) not in modelos:
                modelos[(p, s)] = modelos.get((p, 0)) or modelos[("_", 0)]
        novos, total = [], 0.0
        for F, (est, de_onde, _) in zip(Fs, grafos):
            cam = viterbi(F, est, de_onde, modelos)
            novos.append(cam)
            total += sum(modelos[est[j]].logp(F[t:t + 1])[0] for t, j in enumerate(cam))
        mud = np.mean([np.mean(a != b) for a, b in zip(caminhos, novos)])
        caminhos = novos
        print(f"  iteração {it}: log-verossimilhança média {total / sum(len(F) for F in Fs):.2f}, quadros que mudaram {mud:.1%}")
    # converte em segmentos de fonemas (amostra inicial/final)
    unidades = []
    for k, (cam, (est, _, dono), seq) in enumerate(zip(caminhos, grafos, seqs)):
        donos = np.array(dono)[cam]
        for i in np.unique(donos):
            ts = np.where(donos == i)[0]
            unidades.append(dict(frase=k, idx=int(i), fon=seq[i], ini=int(ts[0] * HOP), fim=int((ts[-1] + 1) * HOP),
                                 esq=seq[i - 1] if i > 0 else "_", dir=seq[i + 1] if i + 1 < len(seq) else "_"))
    return unidades, audios, Fs


if __name__ == "__main__":
    u, _, _ = treinar()
    with open(os.path.join(AQUI, "unidades.json"), "w") as fh:
        json.dump(u, fh)
    from collections import Counter
    c = Counter(x["fon"] for x in u if x["fon"] not in ("sp",))
    print(len(u), "unidades;", dict(sorted(c.items(), key=lambda z: -z[1])))
