"""Voz por recombinação (seleção de unidades) a partir de fala humana real.

Para uma frase nova:
  1. texto → fonemas (mesmo g2p do alinhamento);
  2. Viterbi escolhe, entre as 727 unidades reais, a sequência que minimiza
     custo de alvo (contexto fonético errado) + custo de emenda (salto espectral);
     trechos contínuos da mesma gravação custam zero → longos pedaços naturais;
  3. emenda com crossfade curto; devolve áudio + tempo de cada fonema (para a boca).
"""
import json
import os
import sys
import wave
import numpy as np

AQUI = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, AQUI)
sys.path.insert(0, os.path.join(AQUI, "..", "prototipo"))
import corpus  # noqa: E402
import alinhar  # noqa: E402
import voz  # noqa: E402

SR = corpus.SR
SUBST = {"O": "o", "E": "e", "S": "s", "Z": "z", "A": "a", "x": "r", "b": "p", "g": "k"}


class Voz:
    def __init__(self):
        caminho = os.path.join(AQUI, "unidades.json")
        if not os.path.exists(caminho):
            u, _, _ = alinhar.treinar()
            json.dump(u, open(caminho, "w"))
        self.U = [u for u in json.load(open(caminho)) if u["fon"] != "sp"]
        self.A = corpus.carregar()
        self.F = [alinhar.mfcc(a) for a in self.A]
        self.sd = np.concatenate(self.F).std(0)
        self.por_fon = {}
        for i, u in enumerate(self.U):
            self.por_fon.setdefault(u["fon"], []).append(i)
        for i, u in enumerate(self.U):   # bordas espectrais de cada unidade
            fi, ff = u["ini"] // alinhar.HOP, max(u["fim"] // alinhar.HOP - 1, u["ini"] // alinhar.HOP)
            F = self.F[u["frase"]]
            u["f_ini"] = F[min(fi, len(F) - 1)] / self.sd
            u["f_fim"] = F[min(ff, len(F) - 1)] / self.sd
            u["energia"] = float(np.sqrt(np.mean(self.A[u["frase"]][u["ini"]:u["fim"]] ** 2) + 1e-10))

    def _alvo(self, texto):
        seq = ["_"]
        for tok in texto.lower().replace("-", " ").replace("?", " ? ").replace(",", " , ").replace(".", " ").split():
            if tok in "?,":
                seq.append("_")
            else:
                seq += voz.g2p_palavra(tok)
        seq.append("_")
        return [p if p in self.por_fon else SUBST.get(p, "_") for p in seq]

    def _seguido(self, a, b):
        ua, ub = self.U[a], self.U[b]
        return ua["frase"] == ub["frase"] and ub["idx"] in (ua["idx"] + 1, ua["idx"] + 2)

    def sintetizar(self, texto, w_alvo=1.0, w_emenda=0.15, excluir=None):
        alvo = self._alvo(texto)
        cands = [[i for i in self.por_fon[p] if self.U[i]["frase"] != excluir] or self.por_fon[p] for p in alvo]
        custo_alvo = []
        for t, cs in enumerate(cands):
            esq = alvo[t - 1] if t else "_"
            dir_ = alvo[t + 1] if t + 1 < len(alvo) else "_"
            c = np.array([(self.U[i]["esq"] != esq) + (self.U[i]["dir"] != dir_) +
                          0.5 * ((self.U[i]["dir"] == "_") != (dir_ == "_")) for i in cs], float)
            custo_alvo.append(w_alvo * c)
        D = [custo_alvo[0]]; B = []
        for t in range(1, len(alvo)):
            prev, cur = cands[t - 1], cands[t]
            Fa = np.stack([self.U[i]["f_fim"] for i in prev]); Fb = np.stack([self.U[i]["f_ini"] for i in cur])
            J = np.sqrt(((Fa[:, None] - Fb[None]) ** 2).mean(-1)) * w_emenda
            seg = np.array([[self._seguido(a, b) for b in cur] for a in prev])
            J[seg] = 0.0
            tot = D[-1][:, None] + J
            B.append(tot.argmin(0)); D.append(tot.min(0) + custo_alvo[t])
        k = int(np.argmin(D[-1])); escolha = [k]
        for t in range(len(alvo) - 2, -1, -1):
            k = int(B[t][k]); escolha.append(k)
        escolha = [cands[t][k] for t, k in enumerate(escolha[::-1])]
        return self._montar(alvo, escolha)

    def _montar(self, alvo, escolha, xf=int(0.008 * SR)):
        # agrupa unidades contíguas da mesma gravação em "trechos" naturais
        trechos = []
        for t, i in enumerate(escolha):
            if trechos and self._seguido(trechos[-1]["ult"], i):
                trechos[-1]["ult"] = i; trechos[-1]["fones"].append((alvo[t], i))
            else:
                trechos.append(dict(pri=i, ult=i, fones=[(alvo[t], i)]))
        saida = np.zeros(0, np.float32); tempos = []
        rms_ref = np.median([u["energia"] for u in self.U if u["fon"] in "aeiou"])
        for tr in trechos:
            u0, u1 = self.U[tr["pri"]], self.U[tr["ult"]]
            a = self.A[u0["frase"]]
            ini, fim = u0["ini"], u1["fim"]
            if u0["fon"] == "_":
                ini = max(ini, u0["fim"] - int(0.12 * SR))     # pausas curtas (apara só o silêncio inicial)
            seg = a[max(ini - xf, 0):fim + xf].copy()
            vog = [self.U[i]["energia"] for _, i in tr["fones"] if self.U[i]["fon"] in "aeiou"]
            if vog:
                seg *= np.clip(rms_ref / np.median(vog), 0.6, 1.6)  # nivela volume entre trechos
            base = max(len(saida) - xf, 0)
            if len(saida) >= xf:
                rampa = 0.5 - 0.5 * np.cos(np.linspace(0, np.pi, xf))
                saida[-xf:] = saida[-xf:] * (1 - rampa) + seg[:xf] * rampa
                saida = np.concatenate([saida, seg[xf:]])
            else:
                saida = np.concatenate([saida, seg])
            lead = ini - max(ini - xf, 0)            # posição da amostra `ini` dentro de seg
            for p, i in tr["fones"]:
                u = self.U[i]
                s = max(u["ini"], ini) - ini + base + lead
                e = u["fim"] - ini + base + lead
                tempos.append((p, s / SR, e / SR))
        n_trechos = len(trechos)
        return saida / (np.abs(saida).max() + 1e-9) * 0.9, tempos, n_trechos


def boca(tempos, fps, dur):
    """Trajetória articulatória a partir dos tempos reais dos fonemas (mesma tabela da voz sintética)."""
    n = int(dur * voz.CTRL) + 1
    alvo = {k: np.zeros(n) for k in ("jaw", "rnd", "close", "lips")}
    for p, a, b in tempos:
        i0, i1 = int(a * voz.CTRL), max(int(b * voz.CTRL), int(a * voz.CTRL) + 1)
        f = voz.FONEMAS.get(p, voz.FONEMAS["_"])
        for k in alvo:
            alvo[k][i0:i1] = f[k]
    for k, tau in (("jaw", .022), ("rnd", .03), ("close", .008), ("lips", .012)):
        alvo[k] = voz._suaviza(alvo[k], tau)
    tq = np.arange(int(np.ceil(dur * fps))) / fps
    tc = np.arange(n) / voz.CTRL
    return dict(abre=np.clip(np.interp(tq, tc, alvo["jaw"]) * (1 - 0.6 * np.interp(tq, tc, alvo["close"])), 0, 1),
                redonda=np.interp(tq, tc, alvo["rnd"]), labios=np.interp(tq, tc, alvo["lips"]))


def salvar_wav(caminho, x, sr=SR):
    with wave.open(caminho, "wb") as w:
        w.setnchannels(1); w.setsampwidth(2); w.setframerate(sr)
        w.writeframes((np.clip(x, -1, 1) * 32767).astype("<i2").tobytes())


if __name__ == "__main__":
    v = Voz()
    saida = os.path.join(AQUI, "..", "saida")
    for i, t in enumerate(["Você ouviu isso?", "Foi só o vento.", "O vento não bate na porta.",
                           "Hoje à noite eu vou encontrar a razão de tudo isso."]):
        a, tempos, nt = v.sintetizar(t)
        salvar_wav(os.path.join(saida, f"voz_real_{i + 1}.wav"), a)
        print(f"{t!r}: {len(a) / SR:.2f} s, {len(tempos)} fonemas em {nt} trechos de voz real")
