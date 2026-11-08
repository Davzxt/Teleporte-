"""Validação deixa-um-de-fora: sintetiza cada frase SEM a gravação dela e compara com a real (MCD, dB).

MCD = distorção mel-cepstral após alinhamento DTW (menor = mais próximo). Variante baseada em MFCC:
serve para comparar métodos aqui, não é comparável a valores publicados.
Referência: a mesma frase pelo sintetizador articulatório antigo (prototipo/voz.py).
"""
import os
import sys
import numpy as np

AQUI = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, AQUI); sys.path.insert(0, os.path.join(AQUI, "..", "prototipo"))
import corpus, alinhar, selecao, voz  # noqa: E401,E402


def mcc(x):
    n = 1 + (len(x) - alinhar.JAN) // alinhar.HOP
    idx = np.arange(alinhar.JAN)[None] + alinhar.HOP * np.arange(n)[:, None]
    P = np.abs(np.fft.rfft(x[idx] * np.hanning(alinhar.JAN), 1024)) ** 2
    lm = np.log(P @ alinhar._MEL.T + 1e-8)
    c = 0.5 * np.sqrt(2 / 26) * (lm @ alinhar._DCT.T)   # DCT ortonormal do log da AMPLITUDE
    fala = lm.mean(1) > np.percentile(lm.mean(1), 30)     # ignora silêncio
    return c[fala, 1:13]


def dtw(a, b):
    D = np.sqrt(((a[:, None] - b[None]) ** 2).sum(-1))
    T = np.full((len(a) + 1, len(b) + 1), np.inf); T[0, 0] = 0
    for i in range(1, len(a) + 1):
        for j in range(1, len(b) + 1):
            T[i, j] = D[i - 1, j - 1] + min(T[i - 1, j], T[i, j - 1], T[i - 1, j - 1])
    i, j, s, n = len(a), len(b), 0.0, 0
    while i > 0 and j > 0:
        s += D[i - 1, j - 1]; n += 1
        k = np.argmin([T[i - 1, j - 1], T[i - 1, j], T[i, j - 1]])
        i, j = (i - 1, j - 1) if k == 0 else ((i - 1, j) if k == 1 else (i, j - 1))
    return s / n


def mcd(x, ref):
    return 10 / np.log(10) * np.sqrt(2) * dtw(mcc(x), mcc(ref))


if __name__ == "__main__":
    v = selecao.Voz()
    r_sel, r_ant, trechos = [], [], []
    for k, frase in enumerate(corpus.FRASES):
        a, _, nt = v.sintetizar(frase, excluir=k)
        b, _, _ = voz.falar(frase, "THEO", 24, seed=k)
        b = corpus.reamostrar(b, voz.SR, corpus.SR)
        r_sel.append(mcd(a, v.A[k])); r_ant.append(mcd(b, v.A[k])); trechos.append(nt)
    print(f"MCD voz por recombinação (sem a própria frase): {np.mean(r_sel):.2f} ± {np.std(r_sel):.2f} dB  (trechos/frase: {np.mean(trechos):.1f})")
    print(f"MCD voz articulatória antiga:                    {np.mean(r_ant):.2f} ± {np.std(r_ant):.2f} dB")
    print(f"melhora em {sum(a < b for a, b in zip(r_sel, r_ant))}/20 frases")
