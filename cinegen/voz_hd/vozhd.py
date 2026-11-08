"""VOZ HD: seleção de unidades em 97,5 min de fala real alinhada (FalaBrasil hts-br, BSD-2).

Custos: alvo = contexto fonético (esq/dir) + posição na frase; emenda = espectro (MFCC) + entonação (log F0)
+ energia; trechos contíguos custam 0. Emendas por sobreposição no ponto de maior correlação da onda.
"""
import glob
import os
import pickle
import wave
import numpy as np

AQUI = os.path.dirname(os.path.abspath(__file__))
D = os.environ.get("HTS_DIR", "/tmp/claude-0/-home-user-Teleporte-/0ffc3a91-1c16-542a-a221-0372f80381d3/scratchpad/dados/falabrasil_hts-br/HTSTrainingScripts/data")
SR, HOP, JAN = 16000, 160, 400
SUBST = {"h": "R", "c": "k", "L": "l", "J": "n"}


def _mel(n_fft=512, n_mel=24):
    f = np.linspace(0, SR / 2, n_fft // 2 + 1)
    mel = lambda h: 2595 * np.log10(1 + h / 700)
    pts = 700 * (10 ** (np.linspace(mel(60), mel(7600), n_mel + 2) / 2595) - 1)
    B = np.zeros((n_mel, len(f)))
    for i in range(n_mel):
        a, c, b = pts[i:i + 3]
        B[i] = np.clip(np.minimum((f - a) / (c - a), (b - f) / (b - c)), 0, None)
    return B


MEL = _mel()
DCT = np.cos(np.pi / 24 * (np.arange(24) + 0.5)[None] * np.arange(13)[:, None]) * np.sqrt(2 / 24)


def quadros(x):
    n = max(1 + (len(x) - JAN) // HOP, 1)
    idx = np.arange(JAN)[None] + HOP * np.arange(n)[:, None]
    return np.pad(x, (0, JAN))[idx]


def mfcc(x):
    fr = quadros(x) * np.hanning(JAN)
    P = np.abs(np.fft.rfft(fr, 512)) ** 2
    lm = 0.5 * np.log(P @ MEL.T + 1e-9)
    return lm @ DCT.T


def f0(x):
    """F0 por autocorrelação normalizada (60-300 Hz); 0 = não vozeado."""
    fr = quadros(x)
    fr = fr - fr.mean(1, keepdims=True)
    F = np.fft.rfft(fr, 1024)
    ac = np.fft.irfft(np.abs(F) ** 2, 1024)[:, :400]
    ac = ac / (ac[:, :1] + 1e-9)
    lo, hi = SR // 300, SR // 60
    k = lo + ac[:, lo:hi].argmax(1)
    forte = ac[np.arange(len(k)), k] > 0.45
    return np.where(forte, SR / k, 0.0)


class VozHD:
    def __init__(self):
        cache = os.path.join(D, "_unidades.pkl")
        if os.path.exists(cache):
            self.A, self.U = pickle.load(open(cache, "rb"))
        else:
            self.A, self.U = [], []
            for k, lab in enumerate(sorted(glob.glob(f"{D}/labels/mono/*.lab"))):
                base = os.path.basename(lab)[:-4]
                raw = f"{D}/raw/{base}.raw"
                if not os.path.exists(raw):
                    continue
                x = np.fromfile(raw, "<i2").astype(np.float32) / 32768
                M, F0 = mfcc(x), f0(x)
                linhas = [l.split() for l in open(lab) if l.strip()]
                fon = [l[2] for l in linhas]
                ia = len(self.A); self.A.append(x)
                for i, (s, e, p) in enumerate(linhas):
                    a, b = int(int(s) * 1e-7 * SR), min(int(int(e) * 1e-7 * SR), len(x))
                    fa, fb = min(a // HOP, len(M) - 1), min(max(b // HOP - 1, a // HOP), len(M) - 1)
                    v = F0[fa:fb + 1]; v = v[v > 0]
                    self.U.append(dict(f=ia, i=i, fon=p, ini=a, fim=b, esq=fon[i - 1] if i else "#",
                                       dir=fon[i + 1] if i + 1 < len(fon) else "#", final=(i + 2 >= len(fon)),
                                       m_ini=M[fa], m_fim=M[fb], f0_ini=float(F0[fa]), f0_fim=float(F0[fb]),
                                       f0=float(np.median(v)) if len(v) else 0.0,
                                       en=float(np.sqrt(np.mean(x[a:b] ** 2) + 1e-10))))
            pickle.dump((self.A, self.U), open(cache, "wb"))
        self.por = {}
        for j, u in enumerate(self.U):
            if u["fim"] - u["ini"] >= 160:         # descarta marcações degeneradas (< 10 ms)
                self.por.setdefault(u["fon"], []).append(j)
        self.Mi = np.stack([u["m_ini"] for u in self.U]); self.Mf = np.stack([u["m_fim"] for u in self.U])
        self.Fi = np.array([u["f0_ini"] for u in self.U]); self.Ff = np.array([u["f0_fim"] for u in self.U])
        self.sd = self.Mi.std(0) + 1e-6

    def _seguido(self, a, b):
        return self.U[a]["f"] == self.U[b]["f"] and self.U[b]["i"] == self.U[a]["i"] + 1

    def sintetizar(self, fones, excluir=None, max_cand=400, w_f0=0.6, seed=0):
        rng = np.random.default_rng(seed)
        alvo = [p if p in self.por else SUBST.get(p, "#") for p in fones]
        cands = []
        for t, p in enumerate(alvo):
            excl = excluir if isinstance(excluir, (set, list, tuple)) else {excluir}
            c = [j for j in self.por[p] if self.U[j]["f"] not in excl]
            if len(c) > max_cand:              # poda: prioriza contexto certo (preserva continuações naturais)
                esq = alvo[t - 1] if t else "#"; dr = alvo[t + 1] if t + 1 < len(alvo) else "#"
                nota = [(self.U[j]["esq"] == esq) + (self.U[j]["dir"] == dr) + rng.random() * 0.5 for j in c]
                c = [c[i] for i in np.argsort(nota)[::-1][:max_cand]]
            cands.append(np.array(c))
        n = len(alvo)
        def c_alvo(t):
            esq = alvo[t - 1] if t else "#"; dr = alvo[t + 1] if t + 1 < n else "#"
            fin = t + 2 >= n
            return np.array([(self.U[j]["esq"] != esq) * 1.0 + (self.U[j]["dir"] != dr) * 1.0 +
                             0.7 * (self.U[j]["final"] != fin) for j in cands[t]])
        Dp = c_alvo(0); B = []
        for t in range(1, n):
            a, b = cands[t - 1], cands[t]
            Ja = np.sqrt((((self.Mf[a][:, None] - self.Mi[b][None]) / self.sd) ** 2).mean(-1))
            fa, fb = self.Ff[a][:, None], self.Fi[b][None]
            Jf = np.where((fa > 0) & (fb > 0), np.abs(np.log((fa + 1e-6) / (fb + 1e-6))) * 6, 0.0)
            J = 0.3 * Ja + w_f0 * Jf
            seg = (np.array([self.U[x]["f"] for x in a])[:, None] == np.array([self.U[x]["f"] for x in b])[None]) & \
                  (np.array([self.U[x]["i"] for x in a])[:, None] + 1 == np.array([self.U[x]["i"] for x in b])[None])
            J[seg] = 0
            tot = Dp[:, None] + J
            B.append(tot.argmin(0)); Dp = tot.min(0) + c_alvo(t)
        k = int(Dp.argmin()); esc = [k]
        for t in range(n - 2, -1, -1):
            k = int(B[t][k]); esc.append(k)
        esc = [int(cands[t][k]) for t, k in enumerate(esc[::-1])]
        return self._montar(alvo, esc)

    def _montar(self, alvo, esc, ov=int(0.012 * SR), busca=int(0.005 * SR)):
        trechos = []
        for t, j in enumerate(esc):
            if trechos and self._seguido(trechos[-1][-1][1], j):
                trechos[-1].append((alvo[t], j))
            else:
                trechos.append([(alvo[t], j)])
        out = np.zeros(0, np.float32); tempos = []
        ref = np.median([u["en"] for u in self.U if u["fon"] in ("a", "e", "o")])
        for tr in trechos:
            u0, u1 = self.U[tr[0][1]], self.U[tr[-1][1]]
            x = self.A[u0["f"]]
            ini, fim = u0["ini"], u1["fim"]
            if u0["fon"] == "#":
                ini = max(ini, u0["fim"] - int(0.15 * SR))      # apara só o silêncio inicial
            seg = x[max(ini - ov, 0):fim + ov].astype(np.float32).copy()
            vog = [self.U[j]["en"] for _, j in tr if self.U[j]["fon"] in ("a", "e", "o", "i", "u")]
            if vog:
                seg *= np.clip(ref / np.median(vog), 0.7, 1.4)
            lead = ini - max(ini - ov, 0)
            if len(out) > ov + 2 * busca and len(seg) > 2 * ov:
                # emenda no deslocamento de maior correlação (evita cancelamento de fase / cliques)
                cauda = out[-ov - 2 * busca:]
                melhor, d_m = -1e9, 0
                for d in range(0, 2 * busca, 2):
                    c = float(np.dot(cauda[d:d + ov], seg[:ov]))
                    if c > melhor:
                        melhor, d_m = c, d
                corte = len(out) - ov - 2 * busca + d_m
                rampa = 0.5 - 0.5 * np.cos(np.linspace(0, np.pi, ov))
                mist = out[corte:corte + ov] * (1 - rampa) + seg[:ov] * rampa
                base = corte
                out = np.concatenate([out[:corte], mist, seg[ov:]])
            else:
                base = len(out)
                out = np.concatenate([out, seg])
            for p, j in tr:
                u = self.U[j]
                tempos.append((p, (base + max(u["ini"], ini) - ini + lead) / SR, (base + u["fim"] - ini + lead) / SR))
        return out / (np.abs(out).max() + 1e-9) * 0.9, tempos, len(trechos)


def salvar(caminho, x, sr=SR):
    with wave.open(caminho, "wb") as w:
        w.setnchannels(1); w.setsampwidth(2); w.setframerate(sr)
        w.writeframes((np.clip(x, -1, 1) * 32767).astype("<i2").tobytes())
