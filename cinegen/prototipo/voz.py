"""H5 — Articulação única: texto → fonemas → trajetória articulatória → (voz E boca).

A mesma trajetória (mandíbula, lábios, língua, vozeamento...) alimenta:
  * o sintetizador fonte-filtro (aqui: formantes; em produção: vocoder neural condicionado);
  * a boca do ator no renderizador.
Por isso a sincronia é estrutural: não há alinhamento posterior.
"""
import re
import numpy as np

SR = 22050
CTRL = 200  # Hz, taxa da trajetória articulatória

# jaw: mandíbula, rnd: arredondamento, fr: língua à frente, close: oclusão do trato,
# lips: lábios fechados (visual), v: vozeado, nz: ruído (fricção), nf: freq. do ruído, nas: nasal
def F(jaw=.3, rnd=0, fr=.5, close=0, lips=0, v=1, nz=0, nf=4000, nas=0, d=.07):
    return dict(jaw=jaw, rnd=rnd, fr=fr, close=close, lips=lips, v=v, nz=nz, nf=nf, nas=nas, d=d)

FONEMAS = {
    "a": F(.95, 0, .45, d=.11), "A": F(.7, 0, .45, nas=.7, d=.13),
    "e": F(.5, 0, .85, d=.10), "E": F(.75, 0, .8, d=.11), "i": F(.22, 0, 1.0, d=.09),
    "o": F(.5, .8, .12, d=.10), "O": F(.75, .6, .15, d=.11), "u": F(.2, 1, 0, d=.09),
    "p": F(.12, 0, .5, 1, 1, 0, d=.075), "b": F(.12, 0, .5, .92, 1, 1, d=.065),
    "m": F(.12, 0, .5, .8, 1, 1, nas=1, d=.07),
    "t": F(.28, 0, .85, 1, 0, 0, .25, 4500, d=.065), "d": F(.28, 0, .85, .92, 0, 1, d=.055),
    "n": F(.28, 0, .85, .8, 0, 1, nas=1, d=.06),
    "k": F(.3, 0, .2, 1, 0, 0, .25, 2000, d=.07), "g": F(.3, 0, .2, .92, 0, 1, d=.06),
    "f": F(.18, 0, .5, .85, .55, 0, .7, 6500, d=.08), "v": F(.18, 0, .5, .8, .55, 1, .4, 6500, d=.06),
    "s": F(.18, 0, .9, .88, 0, 0, .9, 5800, d=.09), "z": F(.18, 0, .9, .8, 0, 1, .5, 5800, d=.07),
    "S": F(.22, .6, .7, .88, 0, 0, .9, 3200, d=.09), "Z": F(.22, .6, .7, .8, 0, 1, .5, 3200, d=.07),
    "l": F(.32, 0, .75, .35, d=.05), "r": F(.35, 0, .75, .45, d=.035),
    "x": F(.4, 0, .2, .6, 0, 0, .45, 1300, d=.08),
    "_": F(.06, 0, .5, 1, .6, 0, d=.1),
}

ACENTO = {"á": "a", "à": "a", "â": "a", "ã": "A", "é": "E", "ê": "e", "í": "i",
          "ó": "O", "ô": "o", "õ": "o", "ú": "u"}
VOG = set("aeiouáàâãéêíóôõú")
FRONT = set("eiéêí")


def g2p_palavra(w):
    out, i = [], 0
    while i < len(w):
        c, nx = w[i], (w[i + 1] if i + 1 < len(w) else "")
        pv = w[i - 1] if i else ""
        two = w[i:i + 2]
        if two == "ch": out.append("S"); i += 2; continue
        if two == "lh": out.append("l"); i += 2; continue
        if two == "nh": out.append("n"); i += 2; continue
        if two in ("rr", "ss"): out.append("x" if two == "rr" else "s"); i += 2; continue
        if two == "qu": out.append("k"); i += 2; continue
        if two == "gu" and i + 2 < len(w) and w[i + 2] in FRONT: out.append("g"); i += 2; continue
        if two == "ão": out += ["A", "u"]; i += 2; continue
        if c in VOG:
            out.append(ACENTO.get(c, c))
        elif c == "c": out.append("s" if nx and nx in FRONT else "k")
        elif c == "ç": out.append("s")
        elif c == "s": out.append("z" if (pv in VOG and nx in VOG and nx) else "s")
        elif c == "r":
            out.append("x" if (i == 0 or pv in "nls") else "r")
        elif c == "g": out.append("Z" if nx and nx in FRONT else "g")
        elif c == "j": out.append("Z")
        elif c == "x": out.append("S")
        elif c == "l" and (not nx or nx not in VOG): out.append("u")
        elif c == "h": pass
        elif c in FONEMAS: out.append(c)
        i += 1
    # vogais finais átonas: -o → u, -e → i
    if out and w[-1] == "o": out[-1] = "u"
    if out and w[-1] == "e": out[-1] = "i"
    if len(w) > 1 and w.endswith("os") and out[-2] == "o": out[-2] = "u"
    return out


def g2p(texto):
    seq = [("_", 0.12)]
    for tok in re.findall(r"[a-záàâãéêíóôõúç]+|[?.!,]", texto.lower()):
        if tok in "?.!,":
            seq.append(("_", 0.25))
        else:
            seq += [(p, FONEMAS[p]["d"]) for p in g2p_palavra(tok)]
    seq.append(("_", 0.15))
    # alongamento da última vogal da frase
    for j in range(len(seq) - 1, -1, -1):
        if seq[j][0] in "aAeEioOu":
            seq[j] = (seq[j][0], seq[j][1] * 1.8)
            break
    return seq


def _suaviza(x, tau):
    """Coarticulação: filtro de 1 polo ida-e-volta (sem atraso de fase)."""
    a = np.exp(-1.0 / (tau * CTRL))
    y = x.copy()
    for i in range(1, len(y)):
        y[i] = a * y[i - 1] + (1 - a) * y[i]
    for i in range(len(y) - 2, -1, -1):
        y[i] = a * y[i + 1] + (1 - a) * y[i]
    return y


def trajetoria(texto, rng):
    seq = g2p(texto)
    alvo = {k: [] for k in ("jaw", "rnd", "fr", "close", "lips", "v", "nz", "nf", "nas")}
    for p, d in seq:
        n = max(int(round(d * rng.uniform(0.9, 1.1) * CTRL)), 2)
        for k in alvo:
            alvo[k] += [FONEMAS[p][k]] * n
    tr = {k: np.array(v, np.float64) for k, v in alvo.items()}
    for k, tau in (("jaw", .022), ("rnd", .03), ("fr", .028), ("close", .008), ("lips", .012),
                   ("v", .01), ("nz", .008), ("nf", .01), ("nas", .02)):
        tr[k] = _suaviza(tr[k], tau)
    return tr, len(tr["jaw"]) / CTRL


def _rosenberg(fase):
    g = np.where(fase < 0.4, 0.5 * (1 - np.cos(np.pi * fase / 0.4)),
                 np.where(fase < 0.56, np.cos(np.pi * (fase - 0.4) / 0.32), 0.0))
    return np.diff(g, prepend=g[0])  # derivada = fonte + radiação labial


def sintetizar(tr, dur, voz, pergunta, rng):
    N = int(dur * SR)
    t_ctrl = np.arange(len(tr["jaw"])) / CTRL
    t = np.arange(N) / SR
    P = {k: np.interp(t, t_ctrl, v) for k, v in tr.items()}
    fs = voz["formantes"]
    # entonação: declinação + subida final se for pergunta + micro-variações
    prog = t / dur
    f0 = voz["f0"] * (1.08 - 0.18 * prog)
    if pergunta:
        f0 *= 1 + 0.45 * np.clip((prog - 0.72) / 0.2, 0, 1)
    else:
        f0 *= 1 - 0.1 * np.clip((prog - 0.75) / 0.2, 0, 1)
    f0 *= 1 + 0.012 * np.sin(2 * np.pi * 5.2 * t) + 0.004 * rng.standard_normal(N).cumsum() / np.sqrt(N)
    fonte = _rosenberg(np.cumsum(f0 / SR) % 1.0) * 6.0
    fonte += voz["sopro"] * rng.standard_normal(N) * 0.02
    av = P["v"] * (1 - 0.92 * P["close"]) + P["nas"] * P["close"] * 0.35
    jaw, rnd, fr, cl = P["jaw"], P["rnd"], P["fr"], P["close"]
    Fm = [(230 + 620 * jaw) * (1 - 0.3 * cl) * fs,
          np.maximum(750 + 1550 * fr - 420 * rnd, 560) * fs,
          (2450 + 450 * fr - 250 * rnd) * fs,
          np.full(N, 3500.0 * fs)]
    Bw = [70 + 80 * P["nas"], np.full(N, 90.0), np.full(N, 150.0), np.full(N, 250.0)]
    co = []
    for Fk, Bk in zip(Fm, Bw):  # ressonador de Klatt (ganho DC = 1)
        r = np.exp(-np.pi * Bk / SR)
        b = 2 * r * np.cos(2 * np.pi * Fk / SR)
        c = -r * r
        co.append(((1 - b - c).tolist(), b.tolist(), c.tolist()))
    # ruído de fricção (passa-banda RBJ)
    w0 = 2 * np.pi * np.minimum(P["nf"], SR * 0.45) / SR
    al = np.sin(w0) / (2 * 2.0)
    nb0 = (al / (1 + al)).tolist(); na1 = (-2 * np.cos(w0) / (1 + al)).tolist()
    na2 = ((1 - al) / (1 + al)).tolist()
    ruido = (rng.standard_normal(N) * P["nz"] * 0.35).tolist()
    x_in = (fonte * av).tolist()
    nasr = np.exp(-np.pi * 100 / SR); nb = 2 * nasr * np.cos(2 * np.pi * 270 / SR); nc = -nasr * nasr
    nasal = (fonte * P["nas"] * P["v"]).tolist()
    out = [0.0] * N
    s1 = [0.0] * 4; s2 = [0.0] * 4
    n1 = n2 = 0.0; x1 = x2 = 0.0; y1 = y2 = 0.0
    for n in range(N):
        x = x_in[n]
        for k in range(4):
            A, B, C = co[k]
            y = A[n] * x + B[n] * s1[k] + C[n] * s2[k]
            s2[k] = s1[k]; s1[k] = y; x = y
        m = (1 - nb - nc) * nasal[n] + nb * n1 + nc * n2; n2 = n1; n1 = m
        xn = ruido[n]
        yn = nb0[n] * (xn - x2) - na1[n] * y1 - na2[n] * y2
        x2 = x1; x1 = xn; y2 = y1; y1 = yn
        out[n] = x + 0.6 * m + yn
    a = np.array(out, np.float32)
    a -= np.convolve(a, np.ones(64) / 64, mode="same")  # tira DC
    return a / (np.abs(a).max() + 1e-9) * 0.8


VOZES = {
    "LIA": dict(f0=215, formantes=1.16, sopro=1.2),
    "THEO": dict(f0=118, formantes=1.0, sopro=0.7),
}


def falar(texto, personagem, fps, seed=0):
    """Devolve (áudio, duração, boca por quadro). Boca e voz vêm da MESMA trajetória."""
    rng = np.random.default_rng(seed)
    tr, dur = trajetoria(texto, rng)
    audio = sintetizar(tr, dur, VOZES[personagem], texto.strip().endswith("?"), rng)
    nq = int(np.ceil(dur * fps))
    tq = np.arange(nq) / fps
    tc = np.arange(len(tr["jaw"])) / CTRL
    boca = dict(
        abre=np.clip(np.interp(tq, tc, tr["jaw"]) * (1 - 0.6 * np.interp(tq, tc, tr["close"])), 0, 1),
        redonda=np.interp(tq, tc, tr["rnd"]),
        labios=np.interp(tq, tc, tr["lips"]),
    )
    return audio, dur, boca
