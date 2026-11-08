"""Vídeo de um rosto NOVO (gerado) falando com voz humana recombinada.

1. Identidade: Fotomontagem (ciencia/) em homens 20-55 anos, 128x128; um crítico escolhe a melhor de 64.
2. Fala: voz por seleção de unidades (voz_real/); os tempos dos fonemas viram trajetória de boca.
3. Animação por deformação: lábio inferior/queixo descem, lábio superior sobe, vão preenchido
   (interior escuro + dentes); piscadas; micro-movimento de cabeça e acenos nas sílabas fortes.
4. Pós: ampliação, ótica/película, ambiente de sala, crítico de sincronia medindo os pixels.
"""
import os
import subprocess
import sys
import time
import numpy as np
import torch

np.seterr(over="ignore")
import torch.nn.functional as Fn
from PIL import Image, ImageFilter

AQUI = os.path.dirname(os.path.abspath(__file__))
for p in ("ciencia", "voz_real", "prototipo"):
    sys.path.insert(0, os.path.join(AQUI, "..", p))
import dados, metodos, metricas      # noqa: E402,E401
import selecao                       # noqa: E402
import som, pos, critico             # noqa: E402,E401

SAIDA = os.path.join(AQUI, "..", "saida")
FPS, RES, OUT = 24, 128, 512
FALAS = ["Você ouviu isso?", "O vento não bate na porta.",
         "Hoje à noite eu vou encontrar a razão de tudo isso."]


# ------------------------------------------------------------------ 1. identidade
def gerar_identidade(rng, n_cand=64):
    tr, _ = dados.carregar(RES)
    r, _ = dados.rotulos()
    X = tr[(r[:, 1] == 0) & (r[:, 0] >= 20) & (r[:, 0] <= 55)]
    del tr
    m = metodos.Fotomontagem(base="gmm", grupos=300).treinar(X)
    cand = np.clip(m.gerar(n_cand, rng), 0, 1)
    # crítico (H10): detector real×gerado treinado na hora; escolhe o candidato "mais real"
    falsos = np.clip(np.concatenate([m.gerar(500, rng) for _ in range(4)]), 0, 1)
    reais = X[rng.permutation(len(X))[:2000]]
    pequeno = lambda A: metodos.reduzir(A.astype(np.float32), 64)
    torch.manual_seed(0)
    net = metricas._cnn(); opt = torch.optim.Adam(net.parameters(), 1e-3)
    Xt = torch.tensor(np.concatenate([pequeno(reais), pequeno(falsos)])).permute(0, 3, 1, 2)
    yt = torch.cat([torch.ones(len(reais)), torch.zeros(len(falsos))])
    for _ in range(5):
        perm = torch.randperm(len(Xt))
        for i in range(0, len(Xt), 128):
            j = perm[i:i + 128]
            l = Fn.binary_cross_entropy_with_logits(net(Xt[j]).squeeze(1), yt[j])
            opt.zero_grad(); l.backward(); opt.step()
    net.eval()
    with torch.no_grad():
        nota = net(torch.tensor(pequeno(cand)).permute(0, 3, 1, 2)).squeeze(1).numpy()
    ordem = np.argsort(-nota)
    # entre os 12 mais "reais", prefere boca neutra (sem dentes à mostra): fala sobre sorriso parado fica falsa
    L = cand.mean(3)
    pele = L[:, 70:80, 40:88].mean((1, 2))
    dentes = (L[:, 86:104, 52:76] > pele[:, None, None] * 1.12).mean((1, 2))
    topo = ordem[:12]
    ordem = np.concatenate([topo[np.argsort(dentes[topo])], ordem[12:]])
    # novidade: distância ao rosto real mais próximo
    d = torch.cdist(torch.tensor(cand[ordem[:1], ::4, ::4].reshape(1, -1)),
                    torch.tensor(X[:, ::4, ::4].reshape(len(X), -1))).min().item()
    d_ref = np.median(torch.cdist(torch.tensor(X[:200, ::4, ::4].reshape(200, -1)),
                                  torch.tensor(X[200:, ::4, ::4].reshape(len(X) - 200, -1))).min(1).values.numpy())
    return cand[ordem[0]], cand[ordem[:8]], nota[ordem[:8]], d / d_ref


# ------------------------------------------------------------------ 2. geometria do rosto
def achar_boca(I):
    L = I.mean(2)
    perfil = L[80:108, 56:72].mean(1)
    return 80 + int(np.argmin(perfil))          # linha mais escura = encontro dos lábios


def _gauss(yy, xx, cy, cx, sy, sx):
    return np.exp(-0.5 * (((yy - cy) / sy) ** 2 + ((xx - cx) / sx) ** 2))


class Rosto:
    def __init__(self, I):
        self.I = I.astype(np.float32)
        self.my = achar_boca(I); self.mx = 64
        yy, xx = np.mgrid[:RES, :RES].astype(np.float32)
        self.yy, self.xx = yy, xx
        # campo do queixo: tudo abaixo da boca desce, decaindo para os lados e para o pescoço
        abaixo = 1 / (1 + np.exp(-(yy - self.my) * 1.2))
        self.w_queixo = abaixo * np.exp(-0.5 * ((xx - self.mx) / 20) ** 2) * np.clip((RES + 6 - yy) / 30, 0, 1)
        acima = 1 / (1 + np.exp((yy - self.my) * 1.2))
        self.w_labio_sup = acima * _gauss(yy, xx, self.my, self.mx, 7, 14)
        self.w_lados = _gauss(yy, xx, self.my, self.mx, 9, 18)
        pele = I[self.my - 16:self.my - 10, 44:84].reshape(-1, 3).mean(0)
        self.interior = np.array([0.18, 0.06, 0.06], np.float32) * (0.5 + pele.mean())
        self.dentes = np.clip(np.array([0.80, 0.76, 0.68]) * (0.5 + 0.5 * pele.mean()), 0, 1).astype(np.float32)
        # pálpebra: pele borrada da região dos olhos
        self.palpebra = np.asarray(Image.fromarray((I * 255).astype(np.uint8)).filter(ImageFilter.GaussianBlur(5)),
                                   np.float32) / 255
        self.w_olhos = (_gauss(yy, xx, 30, 32, 4.5, 10) + _gauss(yy, xx, 30, 96, 4.5, 10)).clip(0, 1)

    def quadro(self, abre, redonda=0.0, piscar=0.0):
        h = 10.0 * abre                                              # altura do vão em pixels
        dy = h * 0.8 * self.w_queixo - h * 0.25 * self.w_labio_sup  # deslocamento vertical
        dx = -redonda * 0.18 * (self.xx - self.mx) * self.w_lados
        grade = torch.tensor(np.stack([(self.xx - dx) / (RES - 1) * 2 - 1,
                                       (self.yy - dy) / (RES - 1) * 2 - 1], -1))[None].float()
        img = torch.tensor(self.I).permute(2, 0, 1)[None]
        out = Fn.grid_sample(img, grade, align_corners=True, padding_mode="border")[0].permute(1, 2, 0).numpy()
        if h > 0.6:
            topo, base = self.my - 0.25 * h, self.my + 0.8 * h
            cy, ry = (topo + base) / 2, max((base - topo) / 2, 0.5)
            rx = 14.0 * (1 - 0.35 * redonda)
            # vão em forma de lente: mais alto no centro, fecha nos cantos
            perfil_x = np.clip(1 - ((self.xx - self.mx) / rx) ** 2, 0, 1) ** 0.6
            dv = np.abs(self.yy - cy) / (ry * perfil_x + 1e-3)
            m_vao = np.clip((1.25 - dv) / 0.55, 0, 1) * (perfil_x > 0)
            m_vao = m_vao * np.clip(perfil_x * 4, 0, 1)
            # interior: avermelhado junto aos lábios, escuro no fundo
            prof = np.clip(1 - dv, 0, 1)
            cor = (self.interior * 2.2)[None, None] * (1 - prof[..., None]) + (self.interior * 0.9)[None, None] * prof[..., None]
            if h > 2.0:                                   # dentes superiores, suaves e sombreados
                alt = 0.32 * h
                vy = np.clip(1 - np.abs(self.yy - (topo + alt * 0.7)) / (alt * 0.75), 0, 1)
                vx = np.clip(1 - np.abs(self.xx - self.mx) / (rx * 0.62), 0, 1) ** 0.5
                m_d = np.clip(vy * 2, 0, 1) * vx * min((h - 2.0) / 2.5, 1)
                sombra = 0.75 + 0.25 * vx
                dente = self.dentes[None, None] * sombra[..., None]
                cor = cor * (1 - m_d[..., None]) + dente * m_d[..., None]
            out = out * (1 - m_vao[..., None]) + cor * m_vao[..., None]
            # mesma maciez óptica do resto do rosto: desfoca só em volta da boca
            borr = np.asarray(Image.fromarray((np.clip(out, 0, 1) * 255).astype(np.uint8)).filter(ImageFilter.GaussianBlur(1.0)), np.float32) / 255
            mb = np.clip(_gauss(self.yy, self.xx, cy, self.mx, ry + 3, rx + 3) * 1.5, 0, 1)[..., None]
            out = out * (1 - mb) + borr * mb
        if piscar > 0:
            w = (self.w_olhos * piscar)[..., None]
            out = out * (1 - w) + self.palpebra * w * 0.92
        return out


# ------------------------------------------------------------------ 3. movimento de cabeça
def movimento(n, energia, rng):
    t = np.arange(n) / FPS
    ruido = lambda f, a: sum(a / (k + 1) * np.sin(2 * np.pi * f * (k + 1) * t + rng.uniform(0, 6.3)) for k in range(3))
    ang = ruido(0.21, 0.9) + 0.4 * energia
    tx = ruido(0.17, 1.2)
    ty = ruido(0.25, 0.8) + 1.6 * energia                 # aceno leve nas sílabas fortes
    esc = 1 + 0.015 * ruido(0.11, 1) + np.linspace(0, 0.06, n)   # leve aproximação de câmera
    return ang, tx, ty, esc


def transformar(img, ang, tx, ty, esc):
    a = np.radians(ang)
    M = torch.tensor([[np.cos(a) / esc, -np.sin(a) / esc, -tx * 2 / RES],
                      [np.sin(a) / esc, np.cos(a) / esc, -ty * 2 / RES]], dtype=torch.float32)[None]
    T = torch.tensor(img, dtype=torch.float32).permute(2, 0, 1)[None]
    g = Fn.affine_grid(M, T.shape, align_corners=False)
    return Fn.grid_sample(T, g, align_corners=False, padding_mode="border")[0].permute(1, 2, 0).numpy()


def main():
    t0 = time.time()
    rng = np.random.default_rng(11)
    os.makedirs(SAIDA, exist_ok=True)
    cache = os.path.join(SAIDA, "_identidade.npz")
    if os.path.exists(cache) and os.environ.get("REUSAR"):
        z = np.load(cache); I, top8, notas, novidade = z["I"], z["top8"], z["notas"], float(z["nov"])
    else:
        I, top8, notas, novidade = gerar_identidade(rng)
        np.savez(cache, I=I, top8=top8, notas=notas, nov=novidade)
    Image.fromarray((np.concatenate(list(top8), 1) * 255).astype(np.uint8)).save(os.path.join(SAIDA, "falante_candidatos.png"))
    print(f"identidade escolhida (nota do crítico {notas[0]:.2f}); novidade {novidade:.2f} ({time.time() - t0:.0f} s)")

    v = selecao.Voz()
    pausa = 0.45
    audio, tempos, t = [np.zeros(int(0.6 * selecao.SR), np.float32)], [], 0.6
    for frase in FALAS:
        a, tp, _ = v.sintetizar(frase)
        tempos += [(p, t + s, t + e) for p, s, e in tp]
        audio += [a, np.zeros(int(pausa * selecao.SR), np.float32)]
        t += len(a) / selecao.SR + pausa
    audio = np.concatenate(audio + [np.zeros(int(0.6 * selecao.SR), np.float32)])
    dur = len(audio) / selecao.SR
    b = selecao.boca(tempos, FPS, dur)
    n = len(b["abre"])
    hop = selecao.SR / FPS
    energia = np.array([np.sqrt(np.mean(audio[int(i * hop):int((i + 1) * hop)] ** 2)) for i in range(n)])
    energia = np.convolve(energia / (energia.max() + 1e-9), np.ones(5) / 5, "same")

    rosto = Rosto(I)
    ang, tx, ty, esc = movimento(n, energia, rng)
    piscadas = np.zeros(n)
    tb = 1.0
    while tb < dur:
        k = int(tb * FPS)
        piscadas[k:k + 4] = [0.6, 1.0, 0.8, 0.3][:len(piscadas[k:k + 4])]
        tb += rng.uniform(2.2, 4.5)
    pel = pos.Pelicula(OUT, OUT)
    pel.barra = 0                                           # retrato: sem tarjas
    video = os.path.join(SAIDA, "_falante.mp4")
    ff = subprocess.Popen(["ffmpeg", "-y", "-loglevel", "error", "-f", "rawvideo", "-pix_fmt", "rgb24",
                           "-s", f"{OUT}x{OUT}", "-r", str(FPS), "-i", "-", "-c:v", "libx264", "-crf", "20",
                           "-pix_fmt", "yuv420p", video], stdin=subprocess.PIPE)
    medida, quadros_chave = [], []
    fechada = rosto.quadro(0.0)[rosto.my - 4:rosto.my + 10, 50:78]
    for i in range(n):
        q = rosto.quadro(float(b["abre"][i]), float(b["redonda"][i]), piscadas[i])
        medida.append(float(np.abs(q[rosto.my - 4:rosto.my + 10, 50:78] - fechada).mean()))   # quanto a boca mudou
        q = transformar(q, ang[i], tx[i], ty[i], esc[i])
        grande = np.asarray(Image.fromarray((np.clip(q, 0, 1) * 255).astype(np.uint8)).resize((OUT, OUT), Image.LANCZOS)
                            .filter(ImageFilter.UnsharpMask(2, 60, 2)), np.float32) / 255
        expo = min(1.0, (i / FPS) / 0.5) * min(1.0, (dur - i / FPS) / 0.5)
        f = pel(grande ** 2.2 * 0.9, max(expo, 0))            # volta para luz linear antes da película
        ff.stdin.write(f.tobytes())
        if i in (int(n * 0.2), int(n * 0.45), int(n * 0.62), int(n * 0.8)):
            quadros_chave.append(f)
    ff.stdin.close(); ff.wait()
    Image.fromarray(np.concatenate(quadros_chave, 1)).resize((OUT * 2, OUT // 2)).save(os.path.join(SAIDA, "falante_quadros.png"))

    mix = som.Mixer(dur)
    mix.add(np.interp(np.arange(int(dur * som.SR)) / som.SR, np.arange(len(audio)) / selecao.SR, audio), 0, 0.0, 1.0)
    rng2 = np.random.default_rng(3)
    mix.add(som.ruido_banda(int(dur * som.SR), 60, 4000, rng2) * 0.012, 0, 0.0, 1.0)   # ar da sala
    wav = os.path.join(SAIDA, "_falante.wav")
    mix.salvar(wav, dur)
    final = os.path.join(SAIDA, "falante_real.mp4")
    subprocess.run(["ffmpeg", "-y", "-loglevel", "error", "-i", video, "-i", wav, "-c:v", "copy", "-c:a", "aac",
                    "-b:a", "128k", "-shortest", final], check=True)
    os.remove(video); os.remove(wav)
    r = critico.sincronia(audio, selecao.SR, np.array(medida), FPS)
    print(f"vídeo: {final}  ({dur:.1f} s, {n} quadros, {time.time() - t0:.0f} s no total)")
    print(f"sincronia medida nos pixels: correlação {r['correlacao']:.2f}, atraso {r['atraso_ms']:.0f} ms")


if __name__ == "__main__":
    main()
