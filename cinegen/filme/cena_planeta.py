"""CENA COMPLETA: "O Monólito" — um explorador em outro planeta.

Programa de cena (H1) → decupagem → render procedural do mundo + rosto gerado falando no capacete
→ voz humana recombinada (rádio) → foley por eventos (H6) → trilha → película (H9) → crítico (H10).
Uso: python3 cena_planeta.py
"""
import os
import subprocess
import sys
import time
import wave
import numpy as np
import torch
import torch.nn.functional as Fn

AQUI = os.path.dirname(os.path.abspath(__file__))
for p in ("prototipo", "video_real", "ciencia", "voz_real", "filme"):
    sys.path.insert(0, os.path.join(AQUI, "..", p))
import universo as U        # noqa: E402
import capacete             # noqa: E402
import som_espaco as S      # noqa: E402
import pos, critico         # noqa: E402,E401
import selecao              # noqa: E402
import falante              # noqa: E402

np.seterr(over="ignore")
FPS, W, H, SR = 24, U.W, U.H, S.SR
SAIDA = os.path.join(AQUI, "..", "saida")

ROTEIRO = [
    dict(plano="orbita", dur=8.0),
    dict(plano="geral", pre=1.6, fala=("radio", "Base, aqui é o explorador. Cheguei ao planeta."), pos=1.4),
    dict(plano="capacete", pre=0.7, fala=("capacete", "O céu tem dois sóis, e o vento não para."), pos=1.0),
    dict(plano="capacete", pre=1.8, fala=("capacete", "Espere. Você ouviu isso?"), pos=0.8, evento="zumbido"),
    dict(plano="revelacao", pre=1.4, fala=("radio", "Tem alguma coisa ali, em cima do morro."), pos=1.8),
    dict(plano="capacete", pre=0.5, fala=("capacete", "Eu vou até lá."), pos=0.9),
    dict(plano="final", dur=10.0),
]


def ease(u):
    u = np.clip(u, 0, 1)
    return u * u * (3 - 2 * u)


def zoom(img, cx, cy, k):
    """Aproximação de câmera (dolly digital) em torno de (cx, cy)."""
    if k <= 1.0001:
        return img
    T = torch.tensor(img).permute(2, 0, 1)[None]
    ax, ay = (cx / (W - 1)) * 2 - 1, (cy / (H - 1)) * 2 - 1
    M = torch.tensor([[1 / k, 0, ax * (1 - 1 / k)], [0, 1 / k, ay * (1 - 1 / k)]], dtype=torch.float32)[None]
    g = Fn.affine_grid(M, T.shape, align_corners=False)
    return Fn.grid_sample(T, g, align_corners=False, padding_mode="border")[0].permute(1, 2, 0).numpy()


def melhor_tomada(v, texto, n_max=21):
    """H10: várias tomadas (cada uma proíbe uma gravação do banco → outros pedaços de voz);
    o crítico de sincronia escolhe a aprovada com maior correlação boca × áudio."""
    melhor, nota_m = None, -9
    for k in [None] + list(range(n_max - 1)):
        a, tempos, _ = v.sintetizar(texto, excluir=k)
        b = selecao.boca(tempos, FPS, len(a) / selecao.SR)
        r = critico.sincronia(a, selecao.SR, b["abre"], FPS)
        nota = r["correlacao"] - (0 if abs(r["atraso_ms"]) <= 42 else 1)
        if nota > nota_m:
            melhor, nota_m, info = (a, tempos), nota, r
    print(f"  tomada escolhida para \"{texto}\": correlação {info['correlacao']:.2f}, atraso {info['atraso_ms']:.0f} ms")
    return melhor


def montar(v, rng):
    """Resolve tempos: cada fala sintetizada define a duração do seu plano."""
    t, planos = 0.0, []
    for p in ROTEIRO:
        p = dict(p)
        if "fala" in p:
            a, tempos = melhor_tomada(v, p["fala"][1])
            a = np.interp(np.arange(int(len(a) * SR / selecao.SR)) / SR, np.arange(len(a)) / selecao.SR, a)
            if p["fala"][0] == "radio":
                a = S.radio(a, rng)
                off = 0.21                                 # o bip vem antes da voz
            else:
                off = 0.0
            p["audio"], p["t_fala"] = a, t + p["pre"]
            p["tempos"] = [(f, t + p["pre"] + off + s, t + p["pre"] + off + e) for f, s, e in tempos]
            p["dur"] = p["pre"] + len(a) / SR + p["pos"]
        p["t0"] = t
        t += p["dur"]
        planos.append(p)
    return planos, t


def main():
    t_ini = time.time()
    rng = np.random.default_rng(21)
    os.makedirs(SAIDA, exist_ok=True)
    ceu, planeta, sup = U.Ceu(), U.Planeta(), U.Superficie()
    cap = capacete.Capacete()
    z = np.load(os.path.join(SAIDA, "_identidade.npz"))
    rosto = falante.Rosto(z["I"])
    v = selecao.Voz()
    planos, DUR = montar(v, rng)
    N = int(DUR * FPS)
    print(f"cena: {DUR:.1f} s, {N} quadros, {len(planos)} planos")

    # boca: trajetória de TODAS as falas no capacete, na linha do tempo global
    tempos_cap = [x for p in planos if p.get("fala", ("",))[0] == "capacete" for x in p["tempos"]]
    boca = selecao.boca(tempos_cap, FPS, DUR)
    t_zumbido = next(p["t0"] for p in planos if p.get("evento") == "zumbido")
    fin = planos[-1]
    passos = [fin["t0"] + 1.2 + 0.55 * k for k in range(int((fin["dur"] - 2.5) / 0.55))]
    piscadas = set()
    tb = 1.0
    while tb < DUR:
        piscadas.update(range(int(tb * FPS), int(tb * FPS) + 3)); tb += rng.uniform(2.5, 4.5)
    ang, tx, ty, esc = falante.movimento(N, np.zeros(N), rng)

    pel = pos.Pelicula(W, H)
    video = os.path.join(SAIDA, "_cena.mp4")
    ff = subprocess.Popen(["ffmpeg", "-y", "-loglevel", "error", "-f", "rawvideo", "-pix_fmt", "rgb24", "-s", f"{W}x{H}",
                           "-r", str(FPS), "-i", "-", "-c:v", "libx264", "-crf", "20", "-pix_fmt", "yuv420p", video],
                          stdin=subprocess.PIPE)
    medida_boca, amostras = [], {}
    for n in range(N):
        t = n / FPS
        p = next((q for q in planos if q["t0"] <= t < q["t0"] + q["dur"]), planos[-1])
        u = (t - p["t0"]) / p["dur"]
        tipo = p["plano"]
        if tipo == "orbita":
            img = planeta.render(t, ceu, push=ease(u))
        elif tipo == "geral":
            ox = -60 + 80 * u
            ax = 330
            ay = sup.altura(4, ox)[ax] + 4
            img = sup.render(ox, t, astronauta=dict(camada=4, x=ax, y=ay, escala=0.95))
        elif tipo == "capacete":
            i = min(n, len(boca["abre"]) - 1)
            ab = float(boca["abre"][i]) if t >= p.get("t_fala", 1e9) else 0.0
            q = rosto.quadro(ab, float(boca["redonda"][i]), 1.0 if n in piscadas else 0.0)
            medida_boca.append((n, float(np.abs(q[rosto.my - 4:rosto.my + 10, 50:78] - rosto.quadro(0.0)[rosto.my - 4:rosto.my + 10, 50:78]).mean())))
            q = falante.transformar(q, ang[n] * 0.6, tx[n] * 0.6, ty[n] * 0.6, 1.0)
            fase = (t % 3.6) / 3.6
            emb = 0.35 * np.sin(np.pi * (fase - 0.4) / 0.45) ** 2 if (0.4 < fase < 0.85 and ab == 0) else 0.0
            fundo = sup.render(-220 + 25 * u, t, monolito=1.0 if t > t_zumbido + 1 else 0.0)
            img = cap.render(fundo, q, t, dx=tx[n] * 1.5, dy=ty[n] * 1.5, embacado=emb)
            img = zoom(img, W / 2, H * 0.52, 1.0 + 0.04 * u)
        elif tipo == "revelacao":
            ox = 150
            img = sup.render(ox, t, monolito=float(np.clip(u * 1.6, 0, 1)))
            mx = W * 0.5 + 260 - ox * sup.CAMADAS[sup.MORRO][3]
            img = zoom(img, mx, H * 0.55, 1.0 + 0.45 * ease(u))
        else:  # final: caminha até o monólito
            ox = 150
            ax = int(260 + 140 * u)
            ay = sup.altura(4, ox)[ax] + 4
            img = sup.render(ox, t, monolito=1.0, astronauta=dict(camada=4, x=ax, y=ay, escala=1.15,
                                                                  fase=(t - fin["t0"]) * np.pi / 0.55, andando=True))
            img = zoom(img, W * 0.55, H * 0.55, 1.0 + 0.12 * u)
        expo = min(1.0, t / 2.0) * min(1.0, (DUR - t) / 3.0)
        quadro = pel(np.nan_to_num(img), max(expo, 0.0))
        ff.stdin.write(quadro.tobytes())
        if abs(u - 0.6) < 0.5 / (p["dur"] * FPS) and tipo not in amostras:
            amostras[tipo] = quadro
        if n % 120 == 0:
            print(f"  quadro {n}/{N} ({time.time() - t_ini:.0f} s)")
    ff.stdin.close(); ff.wait()

    # ---- som
    n_a = int(DUR * SR)
    L = np.zeros(n_a); R = np.zeros(n_a)
    def add(sig, t0, ganho=1.0, pan=0.0):
        i = int(t0 * SR); s = sig[:max(n_a - i, 0)] * ganho
        th = (pan + 1) * np.pi / 4
        L[i:i + len(s)] += s * np.cos(th); R[i:i + len(s)] += s * np.sin(th)
    tt = np.arange(n_a) / SR
    g_vento = np.zeros(n_a); g_resp = np.zeros(n_a)
    for p in planos:
        m = (tt >= p["t0"]) & (tt < p["t0"] + p["dur"])
        g_vento[m] = {"orbita": 0, "capacete": 0.35}.get(p["plano"], 1.0)
        g_resp[m] = {"orbita": 0, "capacete": 1.0}.get(p["plano"], 0.25)
    k = int(0.3 * SR); kern = np.ones(k) / k
    g_vento = np.convolve(g_vento, kern, "same"); g_resp = np.convolve(g_resp, kern, "same")
    vento = S.vento(DUR, rng); resp = S.respiracao(DUR, rng)
    add(vento * g_vento, 0, 1.0, -0.2); add(vento[::-1] * g_vento, 0, 1.0, 0.2)
    add(resp * g_resp, 0, 1.0)
    add(S.trilha(DUR, fin["t0"]), 0, 1.0)
    add(S.zumbido(DUR, t_zumbido, rng), 0, 1.0, 0.3)
    rumor = S.ruido(int(8 * SR), 20, 120, rng, 1.0) * np.linspace(0, 1, int(8 * SR)) ** 0.5 * 0.25
    add(rumor, 0, 1.0)
    for p in planos:
        if "audio" in p:
            a = p["audio"]
            if p["fala"][0] == "capacete":       # dentro do capacete: seco, com ressonância curta
                ir = np.exp(-np.arange(int(0.04 * SR)) / SR * 120) * rng.standard_normal(int(0.04 * SR)) * 0.25
                ir[0] = 1
                a = np.convolve(a, ir)[:len(a)]
                a = a / (np.abs(a).max() + 1e-9) * 0.8
            add(a, p["t_fala"], 0.95)
    for k_, tp in enumerate(passos):
        add(S.passo(rng), tp, 0.5, -0.1 + 0.2 * (k_ % 2))
    st = np.stack([L, R], 1)
    fade = np.clip(tt / 1.5, 0, 1) * np.clip((DUR - tt) / 3.0, 0, 1)
    st *= fade[:, None]
    st = st / (np.abs(st).max() + 1e-9) * 0.9
    wav = os.path.join(SAIDA, "_cena.wav")
    with wave.open(wav, "wb") as w:
        w.setnchannels(2); w.setsampwidth(2); w.setframerate(SR)
        w.writeframes((st * 32767).astype("<i2").tobytes())
    final = os.path.join(SAIDA, "o_monolito.mp4")
    subprocess.run(["ffmpeg", "-y", "-loglevel", "error", "-i", video, "-i", wav, "-c:v", "copy", "-c:a", "aac",
                    "-b:a", "160k", "-shortest", final], check=True)
    os.remove(video); os.remove(wav)

    # ---- folha de quadros e crítico de sincronia nos planos de capacete
    from PIL import Image
    ordem = ["orbita", "geral", "capacete", "revelacao", "final"]
    fol = [amostras[k] for k in ordem if k in amostras]
    Image.fromarray(np.concatenate([np.concatenate(fol[:3], 1)[:, :, :], np.concatenate(fol[3:] + [np.zeros_like(fol[0])], 1)], 0)
                    ).resize((W * 3 // 2, H)).save(os.path.join(SAIDA, "o_monolito_quadros.png"))
    for p in planos:
        if p.get("fala", ("",))[0] == "capacete":
            ns = [(i, m) for i, m in medida_boca if p["t0"] <= i / FPS < p["t0"] + p["dur"]]
            i0 = ns[0][0]
            a = np.zeros(int(len(ns) / FPS * SR) + SR)
            j = int((p["t_fala"] - i0 / FPS) * SR)
            a[j:j + len(p["audio"])] = p["audio"][:len(a) - j]
            r = critico.sincronia(a, SR, np.array([m for _, m in ns]), FPS)
            print(f"  sincronia \"{p['fala'][1]}\": correlação {r['correlacao']:.2f}, atraso {r['atraso_ms']:.0f} ms")
    print(f"pronto: {final} ({DUR:.1f} s) em {time.time() - t_ini:.0f} s")


if __name__ == "__main__":
    main()
