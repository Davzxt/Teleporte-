"""Orquestrador: programa de cena → decupagem automática → render → voz/foley → pós → críticos → .mp4

Uso:  python3 gerar.py            (gera ../saida/a_porta.mp4)
Requer: numpy, Pillow, ffmpeg.  Roda em CPU.
"""
import os
import subprocess
import time
import numpy as np
from PIL import Image

import cena, voz, som, pos, critico, codec_generativo
from roteiro import ROTEIRO

FPS, W, H = 24, 640, 360
AQUI = os.path.dirname(os.path.abspath(__file__))
SAIDA = os.path.join(AQUI, "..", "saida")
PORTA_X, PORTA_Z = 215, 1.15


def ease(u):
    u = np.clip(u, 0, 1)
    return u * u * (3 - 2 * u)


# ------------------------------------------------------------ decupagem automática
def montar_linha_do_tempo(rot):
    """Regras de 'diretor': fala → plano médio no falante; batida → insert da porta; etc."""
    planos, falas, batidas, t = [], [], [], 0.0
    for i, ev in enumerate(rot["eventos"]):
        tp = ev["tipo"]
        if tp == "fala":
            audio, dur, boca = voz.falar(ev["texto"], ev["quem"], FPS, seed=i)
            falas.append(dict(t0=t, quem=ev["quem"], texto=ev["texto"], audio=audio, boca=boca, dur=dur))
            planos.append(dict(tipo="medio", quem=ev["quem"], t0=t, dur=dur))
        elif tp == "batida":
            dur = 0.5 + 0.38 * ev["n"]
            for k in range(ev["n"]):
                batidas.append((t + 0.35 + 0.38 * k, ev["forca"]))
            planos.append(dict(tipo="porta", t0=t, dur=dur))
        else:
            planos.append(dict(tipo=tp, t0=t, dur=ev["dur"]))
        t += planos[-1]["dur"]
    return planos, falas, batidas, t


def camera(plano, t):
    u = (t - plano["t0"]) / plano["dur"]
    tp = plano["tipo"]
    if tp in ("abertura", "pausa"):
        return cena.Camera(0, -10, 1.0 + 0.06 * u)
    if tp == "porta":
        return cena.Camera(PORTA_X * PORTA_Z, -20 * PORTA_Z, 2.0 + 0.1 * u)
    if tp == "medio":
        x = cena.ELENCO[plano["quem"]]["x"]
        return cena.Camera(x + (45 if x < 0 else -45), -50, 2.1 + 0.08 * u)
    if tp == "final":
        e = ease(u)
        return cena.Camera(150 * e, -10 - 15 * e, 1.0 + 0.55 * e)


def estado_ator(nome, t, plano, falas, batidas, rng_piscar):
    e = dict(boca_abre=0.04, boca_redonda=0.0, labios=0.6, piscar=0.0,
             olhar=0.0, respira=np.sin(2 * np.pi * 0.25 * t + (0 if nome == "LIA" else 1.7)))
    for f in falas:                        # boca: a MESMA trajetória que gerou a voz
        if f["t0"] <= t < f["t0"] + f["dur"]:
            if f["quem"] == nome:
                i = min(int((t - f["t0"]) * FPS), len(f["boca"]["abre"]) - 1)
                e.update(boca_abre=f["boca"]["abre"][i], boca_redonda=f["boca"]["redonda"][i],
                         labios=f["boca"]["labios"][i])
                e["olhar"] = 0.6 if nome == "LIA" else -0.6
            else:
                e["olhar"] = 0.7 if nome == "LIA" else -0.7
    ult = [b for b, _ in batidas if b <= t]
    if plano["tipo"] in ("porta", "pausa", "final") or (ult and t - ult[-1] < 1.2):
        e["olhar"] = 1.0
    if (t % rng_piscar[0]) < 0.12:          # piscar periódico, fase própria de cada ator
        e["piscar"] = 1.0
    return e


def tremor_porta(t, batidas):
    return sum(f * np.exp(-(t - b) * 18) * np.sin((t - b) * 90) for b, f in batidas if t >= b)


# ------------------------------------------------------------ principal
def main():
    os.makedirs(SAIDA, exist_ok=True)
    tempos = {}
    t0 = time.time()
    planos, falas, batidas, DUR = montar_linha_do_tempo(ROTEIRO)
    tempos["voz (síntese articulatória)"] = time.time() - t0

    locacao = cena.criar_cozinha(ROTEIRO["locacao"]["semente"]) + cena.mesa_e_abajur()
    pelicula = pos.Pelicula(W, H)
    piscar = {"LIA": (3.7,), "THEO": (4.6,)}
    N = int(DUR * FPS)

    video = os.path.join(SAIDA, "_video.mp4")
    ff = subprocess.Popen(["ffmpeg", "-y", "-loglevel", "error", "-f", "rawvideo", "-pix_fmt", "rgb24",
                           "-s", f"{W}x{H}", "-r", str(FPS), "-i", "-", "-c:v", "libx264", "-crf", "24",
                           "-pix_fmt", "yuv420p", video], stdin=subprocess.PIPE)
    t_render = t_pos = 0.0
    folha, crus = [], {}
    for n in range(N):
        t = n / FPS
        p = next(pl for pl in planos if pl["t0"] <= t < pl["t0"] + pl["dur"] + 1e-9) if t < DUR else planos[-1]
        cam = camera(p, t)
        a = time.time()
        splats = list(locacao) + cena.porta(tremor_porta(t, batidas))
        for nome in ROTEIRO["elenco"]:
            splats += cena.ator(nome, **estado_ator(nome, t, p, falas, batidas, piscar[nome]))
        lin = cena.renderizar(splats, cam, W, H)
        b = time.time(); t_render += b - a
        expo = min(1.0, t / 0.8) * min(1.0, (DUR - t) / 1.2)
        q = pelicula(lin, max(expo, 0.0))
        t_pos += time.time() - b
        ff.stdin.write(q.tobytes())
        if p["tipo"] == "final":           # quadros crus de um plano com câmera em movimento (para H4)
            crus.setdefault("final", []).append((np.clip(lin, 0, 1)[::2, ::2] * 255).astype(np.uint8))
        meio = p["t0"] + p["dur"] / 2
        if abs(t - meio) < 0.5 / FPS:
            folha.append(q)
    ff.stdin.close(); ff.wait()
    tempos["render (gaussianas, CPU)"] = t_render
    tempos["pós (película)"] = t_pos

    # ---- som: tudo derivado dos eventos e da posição em relação à câmera
    a = time.time()
    rng = np.random.default_rng(5)
    mix = som.Mixer(DUR)
    mix.add(som.vento(DUR, rng), 0, 0.0, 1.0)
    mix.add(som.trilha(DUR), 0, 0.0, 1.0)
    for f in falas:
        pl = next(pl for pl in planos if abs(pl["t0"] - f["t0"]) < 1e-6)
        cam = camera(pl, f["t0"])
        xs = (cena.ELENCO[f["quem"]]["x"] - cam.x) * cam.zoom / (W / 2)
        mix.add(f["audio"], f["t0"], pan=xs * 0.8, ganho=0.9)
    for tb, forca in batidas:
        mix.add(som.batida_madeira(rng, forca), tb, pan=0.6, ganho=0.8)
    audio = os.path.join(SAIDA, "_audio.wav")
    mix.salvar(audio, DUR)
    tempos["som (foley + trilha + mix)"] = time.time() - a

    final = os.path.join(SAIDA, "a_porta.mp4")
    subprocess.run(["ffmpeg", "-y", "-loglevel", "error", "-i", video, "-i", audio, "-c:v", "copy",
                    "-c:a", "aac", "-b:a", "160k", "-shortest", final], check=True)
    os.remove(video); os.remove(audio)

    # ---- folha de contato
    th = [Image.fromarray(f).resize((320, 180)) for f in folha]
    cols = 3
    sheet = Image.new("RGB", (320 * cols, 180 * ((len(th) + cols - 1) // cols)))
    for i, im in enumerate(th):
        sheet.paste(im, ((i % cols) * 320, (i // cols) * 180))
    sheet.save(os.path.join(SAIDA, "quadros.png"))

    # ---- críticos (H10) e codec generativo (H4)
    linhas = ["# Relatório do CINEGEN (protótipo)", "",
              f"Duração: {DUR:.1f} s · {N} quadros · {W}x{H} @ {FPS} qps", "", "## Tempos (CPU, sem GPU)"]
    for k, v in tempos.items():
        linhas.append(f"- {k}: {v:.1f} s")
    linhas.append(f"- velocidade de render: {N / t_render:.1f} qps (renderizador numpy em CPU)")
    linhas += ["", "## Crítico de sincronia labial (H5 + H10)"]
    for f in falas:
        r = critico.sincronia(f["audio"], voz.SR, f["boca"]["abre"], FPS)
        ruim = critico.sincronia(f["audio"], voz.SR, np.roll(f["boca"]["abre"], 5), FPS)
        linhas.append(f"- {f['quem']}: \"{f['texto']}\" → correlação {r['correlacao']:.2f}, "
                      f"atraso {r['atraso_ms']:.0f} ms, {'APROVADO' if r['aprovado'] else 'REPROVADO'}")
        linhas.append(f"    controle (boca atrasada 208 ms de propósito) → atraso detectado {ruim['atraso_ms']:.0f} ms, "
                      f"{'APROVADO' if ruim['aprovado'] else 'REPROVADO'}")
    c = codec_generativo.avaliar(crus["final"])
    linhas += ["", "## Codec generativo (H4) — plano final com câmera em movimento (320x180)",
               f"- números por quadro em pixels: {c['numeros_por_quadro_pixels']}",
               f"- números por quadro em vetores de movimento: {c['numeros_por_quadro_movimento']}",
               f"- redução: {c['reducao']:.0f}x menos números para o modelo prever",
               f"- qualidade só com movimento (sem resíduo): PSNR {c['psnr_medio_so_movimento']:.1f} dB"]
    rel = "\n".join(linhas) + "\n"
    with open(os.path.join(SAIDA, "relatorio.md"), "w") as fh:
        fh.write(rel)
    print(rel)
    print("Vídeo:", os.path.normpath(final))


if __name__ == "__main__":
    main()
