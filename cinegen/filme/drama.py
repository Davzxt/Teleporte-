"""CENA: "O Interrogatório" — dois rostos gerados (Fotomontagem HD + harmonizador) dialogando,
duas vozes humanas recombinadas (FalaBrasil hts-br e TTS-Portuguese-Corpus), campo/contracampo.

Uso: python3 drama.py [etapa]   etapas: candidatos | cena
"""
import os
import subprocess
import sys
import time
import wave
import numpy as np
import torch
import torch.nn.functional as Fn
from PIL import Image, ImageFilter

AQUI = os.path.dirname(os.path.abspath(__file__))
for p in ("prototipo", "video_real", "ciencia", "voz_real", "voz_hd", "filme"):
    sys.path.insert(0, os.path.join(AQUI, "..", p))
np.seterr(over="ignore")

FPS, W, H, SR = 24, 960, 540, 24000
SAIDA = os.path.join(AQUI, "..", "saida")
RES_C = os.path.join(AQUI, "..", "ciencia", "resultados")

DIALOGO = [
    ("A", "Onde você estava ontem à noite?"),
    ("B", "Em casa. Sozinho."),
    ("A", "Ninguém viu você sair?"),
    ("B", "Você já sabe a resposta."),
    ("A", "Eu sei que você mentiu."),
    ("B", "Então prove."),
]


# ------------------------------------------------------------------ rostos
def gerar_candidatos(n=48, seed=3):
    import dados_hd, metodos, harmonizador
    tr8, _ = dados_hd.carregar(256)
    tr = tr8.astype(np.float32) / 255
    m = metodos.Fotomontagem(base="gmm", grupos=150, regioes=metodos.REGIOES_FFHQ, regioes_res=256,
                             doadores=8, juntar=[[1, 2], [3, 4]]).treinar(tr)
    del tr, tr8
    G = np.clip(m.gerar(n, np.random.default_rng(seed)), 0, 1)
    caminho = os.path.join(RES_C, "harmonizador.pt")
    if os.path.exists(caminho) and os.environ.get("HARMONIZAR"):      # desligado: v1/v2 refutados (ver CADERNO)
        net = harmonizador.UNet(); net.load_state_dict(torch.load(caminho)); net.eval()
        T = torch.tensor(G).permute(0, 3, 1, 2)
        baixo = Fn.interpolate(T, size=(128, 128), mode="area")
        alto = T - Fn.interpolate(baixo, size=(256, 256), mode="bicubic", align_corners=False)   # banda > 128 px
        with torch.no_grad():
            Hm = net(baixo)
        G = (Fn.interpolate(Hm, size=(256, 256), mode="bicubic", align_corners=False) + alto).clamp(0, 1).permute(0, 2, 3, 1).numpy()
    np.save(os.path.join(SAIDA, "_candidatos_drama.npy"), G)
    grade = np.concatenate([np.concatenate(list(G[i * 8:(i + 1) * 8]), 1) for i in range(n // 8)], 0)
    Image.fromarray((grade * 255).astype(np.uint8)).resize((grade.shape[1] // 2, grade.shape[0] // 2)).save(
        os.path.join(SAIDA, "_candidatos_drama.png"))
    return G


def _g(yy, xx, cy, cx, sy, sx):
    return np.exp(-0.5 * (((yy - cy) / sy) ** 2 + ((xx - cx) / sx) ** 2))


class RostoHD:
    """Animação por deformação (como video_real.falante.Rosto), na geometria FFHQ 256 px."""

    def __init__(self, I):
        self.I = I.astype(np.float32); N = I.shape[0]
        L = I.mean(2)
        self.my = 172 + int(np.argmin(L[172:205, 116:140].mean(1)))
        self.mx = 128
        yy, xx = np.mgrid[:N, :N].astype(np.float32); self.yy, self.xx = yy, xx
        abaixo = 1 / (1 + np.exp(-np.clip((yy - self.my) * 0.6, -50, 50)))
        self.w_queixo = abaixo * np.exp(-0.5 * ((xx - self.mx) / 38) ** 2) * np.clip((self.my + 75 - yy) / 40, 0, 1)
        acima = 1 / (1 + np.exp(np.clip((yy - self.my) * 0.6, -50, 50)))
        self.w_sup = acima * _g(yy, xx, self.my, self.mx, 14, 28)
        self.w_lados = _g(yy, xx, self.my, self.mx, 18, 36)
        pele = I[self.my - 30:self.my - 18, 100:156].reshape(-1, 3).mean(0)
        self.interior = np.array([0.18, 0.06, 0.06], np.float32) * (0.5 + pele.mean())
        self.dentes = np.clip(np.array([0.80, 0.76, 0.68]) * (0.5 + 0.5 * pele.mean()), 0, 1).astype(np.float32)
        self.palp = np.asarray(Image.fromarray((I * 255).astype(np.uint8)).filter(ImageFilter.GaussianBlur(8)), np.float32) / 255
        self.w_olhos = np.clip(_g(yy, xx, 120, 95, 7, 17) + _g(yy, xx, 120, 161, 7, 17), 0, 1)

    def quadro(self, abre, redonda=0.0, piscar=0.0):
        N = self.I.shape[0]
        h = 18.0 * abre
        dy = h * 0.8 * self.w_queixo - h * 0.25 * self.w_sup
        dx = -redonda * 0.18 * (self.xx - self.mx) * self.w_lados
        grade = torch.tensor(np.stack([(self.xx - dx) / (N - 1) * 2 - 1, (self.yy - dy) / (N - 1) * 2 - 1], -1))[None].float()
        out = Fn.grid_sample(torch.tensor(self.I).permute(2, 0, 1)[None], grade, align_corners=True,
                             padding_mode="border")[0].permute(1, 2, 0).numpy()
        if h > 1.2:
            topo, base = self.my - 0.25 * h, self.my + 0.8 * h
            cy, ry = (topo + base) / 2, max((base - topo) / 2, 1)
            rx = 26.0 * (1 - 0.35 * redonda)
            px = np.clip(1 - ((self.xx - self.mx) / rx) ** 2, 0, 1) ** 0.6
            dv = np.abs(self.yy - cy) / (ry * px + 1e-3)
            mv = np.clip((1.25 - dv) / 0.55, 0, 1) * (px > 0) * np.clip(px * 4, 0, 1)
            prof = np.clip(1 - dv, 0, 1)
            cor = (self.interior * 2.2)[None, None] * (1 - prof[..., None]) + (self.interior * 0.9)[None, None] * prof[..., None]
            if h > 4:
                alt = 0.32 * h
                vy = np.clip(1 - np.abs(self.yy - (topo + alt * 0.7)) / (alt * 0.75), 0, 1)
                vx = np.clip(1 - np.abs(self.xx - self.mx) / (rx * 0.62), 0, 1) ** 0.5
                md = np.clip(vy * 2, 0, 1) * vx * min((h - 4) / 5, 1)
                cor = cor * (1 - md[..., None]) + self.dentes[None, None] * (0.75 + 0.25 * vx)[..., None] * md[..., None]
            out = out * (1 - mv[..., None]) + cor * mv[..., None]
            borr = np.asarray(Image.fromarray((np.clip(out, 0, 1) * 255).astype(np.uint8)).filter(ImageFilter.GaussianBlur(1.6)), np.float32) / 255
            mb = np.clip(_g(self.yy, self.xx, cy, self.mx, ry + 6, rx + 6) * 1.5, 0, 1)[..., None]
            out = out * (1 - mb) + borr * mb
        if piscar > 0:
            w = (self.w_olhos * piscar)[..., None]
            out = out * (1 - w) + self.palp * w * 0.9
        return out


# ------------------------------------------------------------------ vozes
MAPA_HTS = {"a~": "A", "e~": "e", "i~": "i", "o~": "o", "u~": "u", "w": "u", "j": "i", "w~": "u", "j~": "i",
            "tS": "t", "dZ": "d", "X": "x", "R": "x", "h": "x", "L": "l", "J": "n", "#": "_"}


def falar_A(v, texto, tomadas=12):
    """Voz A (FalaBrasil): várias tomadas, o crítico de sincronia escolhe."""
    import g2p, selecao, critico, vozhd as vz
    fones = g2p.frase(texto)
    melhor, nota_m = None, -9
    for k in range(tomadas):
        a, tempos, _ = v.sintetizar(fones, seed=k, max_cand=300 + 40 * k)
        tempos = [(MAPA_HTS.get(p, p), s, e) for p, s, e in tempos]
        b = selecao.boca(tempos, FPS, len(a) / vz.SR)
        r = critico.sincronia(a, vz.SR, b["abre"], FPS)
        nota = r["correlacao"] - (0 if abs(r["atraso_ms"]) <= 42 else 1)
        if nota > nota_m:
            melhor, nota_m = (a, tempos, vz.SR), nota
    return melhor


def falar_B(v, texto, tomadas=12):
    import selecao, critico
    melhor, nota_m = None, -9
    for k in [None] + list(range(tomadas - 1)):
        a, tempos, _ = v.sintetizar(texto, excluir=k)
        b = selecao.boca(tempos, FPS, len(a) / selecao.SR)
        r = critico.sincronia(a, selecao.SR, b["abre"], FPS)
        nota = r["correlacao"] - (0 if abs(r["atraso_ms"]) <= 42 else 1)
        if nota > nota_m:
            melhor, nota_m = (a, tempos, selecao.SR), nota
    return melhor


def reamostra(a, de, para):
    n = int(len(a) * para / de)
    return np.interp(np.arange(n) / para, np.arange(len(a)) / de, a).astype(np.float32)


# ------------------------------------------------------------------ fotografia
class Fotografia:
    """Close cinematográfico: rosto à esquerda/direita do quadro, fundo estendido e desfocado,
    luz dura lateral (claro-escuro), vinheta, gradação fria nas sombras."""

    def __init__(self, lado):
        self.lado = lado                                   # -1: personagem à esquerda; +1: à direita
        yy, xx = np.mgrid[:H, :W].astype(np.float32)
        self.yy, self.xx = yy, xx
        self.cx = W / 2 + lado * 165

    def render(self, face, t, empurra=0.0, dx=0.0, dy=0.0):
        tam = int(560 * (1 + 0.06 * empurra))
        f = torch.tensor(face, dtype=torch.float32).permute(2, 0, 1)[None] ** 2.2
        grande = Fn.interpolate(f, size=(tam, tam), mode="bicubic", align_corners=False)[0].permute(1, 2, 0).numpy()
        fundo = Fn.interpolate(Fn.avg_pool2d(f, 9, 1, 4), size=(H * 2, H * 2), mode="bilinear", align_corners=False)
        fundo = fundo[0].permute(1, 2, 0).numpy()
        ox = int(H - (self.cx + dx))
        img = np.take(fundo[H // 4:H // 4 + H], np.clip(np.arange(W) + ox, 0, 2 * H - 1), axis=1) * 0.55
        x0, y0 = int(self.cx - tam / 2 + dx), int(215 - 0.47 * tam + dy)        # olhos no terço superior
        gy, gx = np.mgrid[:tam, :tam].astype(np.float32) / (tam - 1) * 2 - 1
        alfa = np.clip((1.0 - np.maximum(np.abs(gx), np.abs(gy) * 0.85)) / 0.45, 0, 1)[..., None] ** 1.5
        xa, ya, xb, yb = max(x0, 0), max(y0, 0), min(x0 + tam, W), min(y0 + tam, H)
        reg = img[ya:yb, xa:xb]; fg = grande[ya - y0:yb - y0, xa - x0:xb - x0]; al = alfa[ya - y0:yb - y0, xa - x0:xb - x0]
        img[ya:yb, xa:xb] = reg * (1 - al) + fg * al
        # luz dura vinda de cima/do lado de fora (claro-escuro)
        chave = 0.42 + 0.75 * (1 / (1 + np.exp(np.clip(-self.lado * (self.xx - self.cx) / 90, -50, 50))))
        topo = np.clip(1.15 - (self.yy / H) * 0.6, 0.4, 1.2)
        img = img * (chave * topo)[..., None] * 0.95
        img += np.exp(-((self.xx - (W / 2 - self.lado * 380)) ** 2 + (self.yy - 60) ** 2) / (2 * 120 ** 2))[..., None] * np.array([0.25, 0.2, 0.12])
        return img.astype(np.float32)


# ------------------------------------------------------------------ som
def ruido(n, lo, hi, rng, incl=0.5):
    X = np.fft.rfft(rng.standard_normal(n)); f = np.fft.rfftfreq(n, 1 / SR)
    X *= ((f > lo) & (f < hi)) / np.maximum(f, 20) ** incl
    y = np.fft.irfft(X, n); return y / (np.abs(y).max() + 1e-9)


def chuva(dur, rng):
    n = int(dur * SR); y = ruido(n, 400, 9000, rng, 0.3) * 0.25
    for _ in range(int(dur * 25)):                              # gotas no vidro
        i = rng.integers(0, n - 800); g = rng.integers(60, 300)
        y[i:i + g] += rng.standard_normal(g) * np.hanning(g) * rng.uniform(0.05, 0.3)
    return y * 0.35


def tique(rng):
    n = int(0.03 * SR); t = np.arange(n) / SR
    return (np.sin(2 * np.pi * 3100 * t) + 0.5 * np.sin(2 * np.pi * 5200 * t)) * np.exp(-t * 300) * 0.25


def trovao(rng):
    n = int(4.0 * SR); t = np.arange(n) / SR
    return ruido(n, 20, 300, rng, 1.2) * np.exp(-t * 1.2) * np.clip(t / 0.15, 0, 1) * 0.9


def trilha(dur):
    n = int(dur * SR); t = np.arange(n) / SR
    y = sum(a * np.sin(2 * np.pi * f * t + 0.4 * np.sin(2 * np.pi * 0.11 * t)) for f, a in ((55, 1), (82.4, .4), (110, .3), (116.5, .12)))
    tensao = np.clip((t - dur * 0.55) / (dur * 0.4), 0, 1)
    y = y + tensao * 0.5 * np.sin(2 * np.pi * 233.1 * t) * (0.5 + 0.5 * np.sin(2 * np.pi * 6 * t))
    return y * 0.05 * (0.6 + 0.8 * tensao) * np.clip(t / 3, 0, 1)


# ------------------------------------------------------------------ cena
def cena():
    import vozhd as vz_hd, selecao, pos, critico, falante
    t_ini = time.time()
    rng = np.random.default_rng(9)
    C = np.load(os.path.join(SAIDA, "_candidatos_drama.npy"))
    escolha = [int(x) for x in os.environ.get("ROSTOS", "0,1").split(",")]
    rostos = {"A": RostoHD(C[escolha[0]]), "B": RostoHD(C[escolha[1]][:, ::-1].copy())}   # B espelhado: olha p/ A
    foto = {"A": Fotografia(-1), "B": Fotografia(+1)}
    vA, vB = vz_hd.VozHD(), selecao.Voz()
    planos, t = [], 0.8
    for quem, texto in DIALOGO:
        a, tempos, sr = falar_A(vA, texto) if quem == "A" else falar_B(vB, texto)
        a = reamostra(a, sr, SR)
        pre, pos_ = 0.45, (1.6 if texto.startswith("Então") else 0.55)
        planos.append(dict(quem=quem, texto=texto, t0=t - 0.45 if planos else 0.0, t_fala=t, audio=a,
                           tempos=[(p, t + s, t + e) for p, s, e in tempos]))
        t += len(a) / SR + pre + pos_
    for i, p in enumerate(planos):
        p["fim"] = planos[i + 1]["t0"] if i + 1 < len(planos) else t + 0.8
    DUR = planos[-1]["fim"]
    N = int(DUR * FPS)
    bocas = {q: selecao.boca([x for p in planos if p["quem"] == q for x in p["tempos"]], FPS, DUR) for q in "AB"}
    pisc = {q: set() for q in "AB"}
    for q in "AB":
        tb = rng.uniform(0.5, 2)
        while tb < DUR:
            pisc[q].update(range(int(tb * FPS), int(tb * FPS) + 3)); tb += rng.uniform(2.5, 5)
    ang, tx, ty, _ = falante.movimento(N, np.zeros(N), rng)
    pel = pos.Pelicula(W, H)
    video = os.path.join(SAIDA, "_drama.mp4")
    ff = subprocess.Popen(["ffmpeg", "-y", "-loglevel", "error", "-f", "rawvideo", "-pix_fmt", "rgb24", "-s", f"{W}x{H}",
                           "-r", str(FPS), "-i", "-", "-c:v", "libx264", "-crf", "18", "-pix_fmt", "yuv420p", video], stdin=subprocess.PIPE)
    medidas, amostras = {i: [] for i in range(len(planos))}, []
    t_trovao = planos[-1]["t_fala"] + len(planos[-1]["audio"]) / SR + 0.5
    for n in range(N):
        tt = n / FPS
        k = next(i for i, p in enumerate(planos) if p["t0"] <= tt < p["fim"] or i == len(planos) - 1)
        p = planos[k]; q = p["quem"]
        b = bocas[q]; i = min(n, len(b["abre"]) - 1)
        ab = float(b["abre"][i]) if p["t_fala"] <= tt else 0.0
        fr = rostos[q].quadro(ab, float(b["redonda"][i]), 1.0 if n in pisc[q] else 0.0)
        r0 = rostos[q]
        medidas[k].append(float(np.abs(fr[r0.my - 8:r0.my + 20, 100:156] - r0.I[r0.my - 8:r0.my + 20, 100:156]).mean()))
        fr = falante.transformar(fr, ang[n] * 0.5, tx[n] * 0.8, ty[n] * 0.8, 1.0)
        u = (tt - p["t0"]) / (p["fim"] - p["t0"])
        img = foto[q].render(fr, tt, empurra=u, dx=tx[n] * 3, dy=ty[n] * 3)
        if tt > t_trovao and tt < t_trovao + 0.25:                       # relâmpago na janela
            img = img * (1 + 1.5 * np.exp(-(tt - t_trovao) * 18))
        expo = min(1.0, tt / 1.0) * min(1.0, (DUR - tt) / 1.0)
        quadro = pel(img, max(expo, 0))
        ff.stdin.write(quadro.tobytes())
        if abs(u - 0.55) < 0.6 / ((p["fim"] - p["t0"]) * FPS) and len(amostras) < 4 and (not amostras or amostras[-1][0] != k):
            amostras.append((k, quadro))
    ff.stdin.close(); ff.wait()
    # som
    n_a = int(DUR * SR); L = np.zeros(n_a); Rr = np.zeros(n_a)
    def add(s, t0, g=1.0, pan=0.0):
        i = int(t0 * SR); s = s[:max(n_a - i, 0)] * g; th = (pan + 1) * np.pi / 4
        L[i:i + len(s)] += s * np.cos(th); Rr[i:i + len(s)] += s * np.sin(th)
    add(chuva(DUR, rng), 0, 0.8, -0.5); add(chuva(DUR, rng), 0, 0.8, 0.5)
    tt_ = np.arange(n_a) / SR
    add(np.sin(2 * np.pi * 60 * tt_) * 0.01 + ruido(n_a, 50, 3000, rng, 1) * 0.02, 0)          # lâmpada e ar da sala
    for s in np.arange(0.3, DUR, 1.0):
        add(tique(rng), s, 0.5, 0.6)
    add(trilha(DUR), 0, 1.0)
    add(trovao(rng), t_trovao, 1.0, -0.3)
    ir = np.exp(-np.arange(int(0.35 * SR)) / SR * 14) * rng.standard_normal(int(0.35 * SR)) * 0.06; ir[0] = 1
    for p in planos:
        a = np.convolve(p["audio"], ir)[:len(p["audio"]) + len(ir) - 1]
        add(a / (np.abs(a).max() + 1e-9) * 0.85, p["t_fala"], 1.0, -0.15 if p["quem"] == "A" else 0.15)
    st = np.stack([L, Rr], 1); st *= (np.clip(tt_ / 1, 0, 1) * np.clip((DUR - tt_) / 1.5, 0, 1))[:, None]
    st = st / (np.abs(st).max() + 1e-9) * 0.9
    wav = os.path.join(SAIDA, "_drama.wav")
    with wave.open(wav, "wb") as w:
        w.setnchannels(2); w.setsampwidth(2); w.setframerate(SR); w.writeframes((st * 32767).astype("<i2").tobytes())
    final = os.path.join(SAIDA, "o_interrogatorio.mp4")
    subprocess.run(["ffmpeg", "-y", "-loglevel", "error", "-i", video, "-i", wav, "-c:v", "copy", "-c:a", "aac", "-b:a", "160k",
                    "-shortest", final], check=True)
    os.remove(video); os.remove(wav)
    Image.fromarray(np.concatenate([np.concatenate([a[1] for a in amostras[:2]], 1), np.concatenate([a[1] for a in amostras[2:4]], 1)], 0)
                    ).resize((W, H)).save(os.path.join(SAIDA, "o_interrogatorio_quadros.png"))
    for k, p in enumerate(planos):
        ns = medidas[k]
        a = np.zeros(int(len(ns) / FPS * SR) + SR)
        j = int((p["t_fala"] - p["t0"]) * SR); a[j:j + len(p["audio"])] = p["audio"][:len(a) - j]
        r = critico.sincronia(a, SR, np.array(ns), FPS)
        print(f"  {p['quem']}: \"{p['texto']}\" → sincronia nos pixels: correlação {r['correlacao']:.2f}, atraso {r['atraso_ms']:.0f} ms")
    print(f"pronto: {final} ({DUR:.1f} s) em {time.time() - t_ini:.0f} s")


if __name__ == "__main__":
    {"candidatos": gerar_candidatos, "cena": cena}[sys.argv[1] if len(sys.argv) > 1 else "cena"]()
