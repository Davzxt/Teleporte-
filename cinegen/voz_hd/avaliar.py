"""Deixa-um-de-fora em 20 frases do próprio banco: sintetiza a sequência de fonemas SEM a gravação
original e compara com ela (MCD, mesma métrica da voz anterior) + emendas por segundo."""
import numpy as np
import vozhd as voz

def mcd(a, b):
    A, B = voz.mfcc(a)[:, 1:13], voz.mfcc(b)[:, 1:13]
    ea, eb = voz.mfcc(a)[:, 0], voz.mfcc(b)[:, 0]
    A, B = A[ea > np.percentile(ea, 30)], B[eb > np.percentile(eb, 30)]
    D = np.sqrt(((A[:, None] - B[None]) ** 2).sum(-1))
    T = np.full((len(A) + 1, len(B) + 1), np.inf); T[0, 0] = 0
    for i in range(1, len(A) + 1):
        T[i, 1:] = D[i - 1] + np.minimum(T[i - 1, 1:], T[i - 1, :-1])
        for j in range(1, len(B) + 1):
            T[i, j] = min(T[i, j], D[i - 1, j - 1] + T[i, j - 1])
    # comprimento médio do caminho ~ (len A + len B)/2
    return 10 / np.log(10) * np.sqrt(2) * T[-1, -1] / ((len(A) + len(B)) / 2)

if __name__ == "__main__":
    v = voz.VozHD()
    frases = sorted({u["f"] for u in v.U})[::50][:20]
    seqs = {}
    for u in v.U:
        seqs.setdefault(u["f"], []).append(u["fon"])
    dup = lambda f: {g for g, s in seqs.items() if s == seqs[f]}      # gravações repetidas também saem
    r, emendas = [], []
    for f in frases:
        fones = [u["fon"] for u in v.U if u["f"] == f]
        a, _, nt = v.sintetizar(fones, excluir=dup(f))
        r.append(mcd(a, v.A[f])); emendas.append(nt / (len(a) / voz.SR))
    print(f"VOZ HD: MCD {np.mean(r):.2f} ± {np.std(r):.2f} dB | emendas por segundo {np.mean(emendas):.1f}")
