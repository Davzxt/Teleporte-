"""H2 + H5 em 3D — o ator treinado fala, com a câmera girando em volta dele.

Rig simples aplicado às gaussianas do ator:
  * mandíbula: gaussianas abaixo da linha da boca giram em torno de uma dobradiça;
  * lábios: arredondamento estreita e projeta a boca;
  * cavidade da boca e dentes: gaussianas extras do rig (escondidas atrás dos lábios).
A abertura da mandíbula e o arredondamento vêm da MESMA trajetória articulatória que gera a voz.
O crítico de sincronia mede a boca nos PIXELS do vídeo final, não nos parâmetros.

Uso: python3 falar3d.py   (precisa de ../saida/ator3d.pt, gerado por treinar_ator.py)
"""
import os
import subprocess
import sys
import time
import numpy as np
import torch

AQUI = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.join(AQUI, "..", "prototipo"))
import voz, som, pos, critico  # noqa: E402
import referencia, splat3d      # noqa: E402

SAIDA = os.path.join(AQUI, "..", "saida")
FPS, W, H = 24, 480, 270
TEXTO = "Eu fui criado a partir de dezoito fotos. Agora posso falar de qualquer ângulo."
BOCA = np.array([0.0, -0.44, 0.80])


def carregar_ator():
    sd = torch.load(os.path.join(SAIDA, "ator3d.pt"))
    g = splat3d.Gaussianas(sd["pos"], torch.full_like(sd["pos"], 0.5))
    g.load_state_dict(sd)
    return {k: v.detach() for k, v in g.valores().items()}


def pecas_do_rig(n_cav=70, seed=0):
    """Cavidade escura e dentes: fazem parte do 'arquivo de ator' como um blendshape."""
    rng = np.random.default_rng(seed)
    cav = rng.normal(size=(n_cav, 3)) * [0.06, 0.04, 0.015] + [0, -0.47, 0.715]
    xs = np.linspace(-0.11, 0.11, 9)
    sup = np.stack([xs, np.full(9, -0.415), 0.69 - 2 * xs ** 2], 1)
    inf = np.stack([xs, np.full(9, -0.475), 0.67 - 2 * xs ** 2], 1)
    p = np.concatenate([cav, sup, inf])
    cor = np.concatenate([np.tile([0.14, 0.03, 0.04], (n_cav, 1)), np.tile([0.88, 0.86, 0.8], (18, 1))])
    esc = np.concatenate([np.full((n_cav, 3), 0.03), np.tile([0.014, 0.008, 0.008], (18, 1))])
    t = lambda a: torch.tensor(a, dtype=torch.float32)
    n = len(p)
    q = torch.zeros(n, 4); q[:, 0] = 1
    extra = dict(pos=t(p), esc=t(esc), quat=q, cor=t(cor), op=torch.full((n,), 0.97))
    dentes_inf = torch.zeros(n, dtype=torch.bool); dentes_inf[n_cav + 9:] = True
    return extra, dentes_inf


def rot_x(theta):
    c, s = torch.cos(theta), torch.sin(theta)
    o, z = torch.ones_like(c), torch.zeros_like(c)
    return torch.stack([o, z, z, z, c, -s, z, s, c], -1).reshape(-1, 3, 3)


def aplicar_rig(base, extra, dentes_inf, abre, redonda, aceno):
    g = {k: torch.cat([base[k], extra[k]]) for k in base}
    p = g["pos"].clone()
    x, y, z = p.unbind(-1)
    n_base = base["pos"].shape[0]
    # peso da mandíbula: abaixo da boca, na frente, acima do pescoço
    # transição dura só nos lábios; nas bochechas a pele estica (transição larga) em vez de rasgar
    largura = 0.04 + 0.22 * ((x.abs() - 0.14) / 0.12).clamp(0, 1)
    w = ((-0.43 - y) / largura).clamp(0, 1) * ((z + 0.1) / 0.3).clamp(0, 1) * ((y + 1.05) / 0.2).clamp(0, 1)
    w = w * (1 - 0.6 * ((x.abs() - 0.3) / 0.25).clamp(0, 1))
    w[n_base:] = 0
    w[n_base:][dentes_inf] = 1
    eixo = torch.tensor([0.0, -0.12, -0.1])
    theta = 0.38 * abre * w
    Rj = rot_x(theta)
    p = eixo + ((Rj @ (p - eixo)[..., None])[..., 0])
    # lábio superior sobe um pouco
    sup = ((y > -0.43) & (y < -0.33) & (z > 0.55)).float()
    sup[n_base:] = 0
    p[:, 1] += 0.03 * abre * sup
    # arredondamento: estreita e projeta
    lab = ((x.abs() < 0.3) & (y > -0.62) & (y < -0.3) & (z > 0.45)).float()
    p[:, 0] *= 1 - 0.28 * redonda * lab
    p[:, 2] += 0.035 * redonda * lab
    # aceno de cabeça (corpo inteiro acima do pescoço)
    pesc = torch.tensor([0.0, -0.95, 0.0])
    Rn = rot_x(torch.tensor([aceno]))[0]
    acima = ((y + 1.0) / 0.3).clamp(0, 1)[:, None]
    p = p + acima * ((p - pesc) @ Rn.T + pesc - p)
    g["pos"] = p
    return g


def escuro_na_boca(img, R, t):
    """Mede a abertura da boca direto nos pixels: fração de pixels escuros em volta da boca."""
    pc = R @ BOCA + t
    f = 2.6 * H / 2
    u, v = int(f * pc[0] / pc[2] + W / 2), int(f * pc[1] / pc[2] + H / 2)
    jan = img[max(v - 6, 0):v + 7, max(u - 9, 0):u + 10].astype(np.float32) / 255
    return float((jan.mean(2) < 0.22).mean())


def main():
    t0 = time.time()
    base = carregar_ator()
    extra, dentes_inf = pecas_do_rig()
    audio, dur, boca = voz.falar(TEXTO, "THEO", FPS, seed=3)
    pre, pos_ = 0.6, 0.9
    N = int((pre + dur + pos_) * FPS)
    pel = pos.Pelicula(W, H)
    video = os.path.join(SAIDA, "_v3d.mp4")
    ff = subprocess.Popen(["ffmpeg", "-y", "-loglevel", "error", "-f", "rawvideo", "-pix_fmt", "rgb24",
                           "-s", f"{W}x{H}", "-r", str(FPS), "-i", "-", "-c:v", "libx264", "-crf", "22",
                           "-pix_fmt", "yuv420p", video], stdin=subprocess.PIPE)
    medida, t_render = [], 0.0
    with torch.no_grad():
        for n in range(N):
            t = n / FPS
            k = int((t - pre) * FPS)
            fala = 0 <= k < len(boca["abre"])
            ab = float(boca["abre"][k]) if fala else 0.0
            rd = float(boca["redonda"][k]) if fala else 0.0
            u = n / max(N - 1, 1)
            az = -45 + 90 * (u * u * (3 - 2 * u))          # câmera dá meia volta em torno do ator
            R, tt = referencia.camera(az, 6 + 4 * np.sin(2 * np.pi * u), dist=3.4, alvo=(0, -0.3, 0))
            g = aplicar_rig(base, extra, dentes_inf, ab, rd, 0.03 * np.sin(2 * np.pi * 0.45 * t))
            a = time.time()
            img = splat3d.renderizar(g, R, tt, (W, H), 2.6).numpy()
            t_render += time.time() - a
            q = pel(img, min(1.0, t / 0.4) * min(1.0, (N / FPS - t) / 0.6))
            ff.stdin.write(q.tobytes())
            if fala:
                medida.append(escuro_na_boca((np.clip(img, 0, 1) * 255).astype(np.uint8), R, tt))
    ff.stdin.close(); ff.wait()

    mix = som.Mixer(N / FPS)
    mix.add(audio, pre, 0.0, 1.0)
    wav = os.path.join(SAIDA, "_a3d.wav")
    mix.salvar(wav, N / FPS)
    final = os.path.join(SAIDA, "ator3d_fala.mp4")
    subprocess.run(["ffmpeg", "-y", "-loglevel", "error", "-i", video, "-i", wav, "-c:v", "copy",
                    "-c:a", "aac", "-b:a", "128k", "-shortest", final], check=True)
    os.remove(video); os.remove(wav)

    r = critico.sincronia(audio, voz.SR, np.array(medida[:len(boca["abre"])]), FPS)
    rel = (f"\n## Ator 3D falando (H2 + H5)\n\n"
           f"- fala: \"{TEXTO}\"\n- {N} quadros {W}x{H}, câmera girando 90° em volta do ator\n"
           f"- render: {N / t_render:.1f} qps em CPU ({base['pos'].shape[0] + extra['pos'].shape[0]} gaussianas)\n"
           f"- crítico de sincronia medindo a boca **nos pixels do vídeo**: correlação {r['correlacao']:.2f}, "
           f"atraso {r['atraso_ms']:.0f} ms → {'APROVADO' if r['aprovado'] else 'REPROVADO'}\n"
           f"- tempo total: {time.time() - t0:.0f} s\n")
    with open(os.path.join(SAIDA, "ator3d_relatorio.md"), "a") as fh:
        fh.write(rel)
    print(rel)


if __name__ == "__main__":
    main()
