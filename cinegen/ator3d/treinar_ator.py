"""H2 — Criar um "arquivo de ator" 3D a partir de fotos de vários ângulos.

1. Casco visual: a silhueta de cada foto esculpe um volume → pontos iniciais na superfície.
2. Otimiza as gaussianas 3D (posição, escala, rotação, cor, opacidade) para reproduzir as fotos.
3. Testa em ângulos que o treino NUNCA viu (o teste da H2) e salva o ator.

Uso: python3 treinar_ator.py   (~5 min em CPU; segundos em GPU)
"""
import os
import time
import numpy as np
import torch
from PIL import Image

import referencia
import splat3d

AQUI = os.path.dirname(os.path.abspath(__file__))
SAIDA = os.path.join(AQUI, "..", "saida")
RES, FOCO, ITERS, N_GAUSS = 128, 2.6, 1500, 4000


def projetar(p, R, t, res):
    pc = p @ R.T + t
    f = FOCO * res / 2
    return pc[:, 0] / pc[:, 2] * f + res / 2, pc[:, 1] / pc[:, 2] * f + res / 2


def casco_visual(vistas, rng, n=150_000):
    def dentro(p):
        ok = np.ones(len(p), bool)
        for (R, t), _, mask in vistas:
            u, v = projetar(p, R, t, RES)
            ui, vi = np.clip(u.astype(int), 0, RES - 1), np.clip(v.astype(int), 0, RES - 1)
            ok &= (u >= 0) & (u < RES) & (v >= 0) & (v < RES) & mask[vi, ui]
        return ok
    p = rng.uniform([-1.1, -2.0, -1.1], [1.1, 1.2, 1.1], (n, 3))
    p = p[dentro(p)]
    sup = np.zeros(len(p), bool)
    for eixo in range(3):                 # ponto de superfície: um vizinho próximo cai fora do casco
        for s in (-1, 1):
            q = p.copy(); q[:, eixo] += s * 0.06
            sup |= ~dentro(q)
    p = p[sup]
    return p[rng.choice(len(p), min(N_GAUSS, len(p)), replace=False)]


def cor_inicial(p, vistas):
    soma = np.zeros((len(p), 3)); cont = np.zeros(len(p))
    for (R, t), img, mask in vistas:
        u, v = projetar(p, R, t, RES)
        ui, vi = np.clip(u.astype(int), 0, RES - 1), np.clip(v.astype(int), 0, RES - 1)
        soma += img[vi, ui]; cont += 1     # média simples (ignora oclusão; o treino corrige)
    return soma / np.maximum(cont, 1)[:, None]


def psnr(a, b):
    mse = float(((a - b) ** 2).mean())
    return 10 * np.log10(1 / max(mse, 1e-10))


def main():
    torch.manual_seed(0)
    rng = np.random.default_rng(0)
    os.makedirs(SAIDA, exist_ok=True)
    t0 = time.time()
    treino, teste = referencia.sessao(RES)
    print(f"fotos: {len(treino)} de treino, {len(teste)} de teste ({time.time() - t0:.0f} s)")

    pts = casco_visual(treino, rng)
    print(f"casco visual: {len(pts)} gaussianas iniciais")
    ator = splat3d.Gaussianas(torch.tensor(pts, dtype=torch.float32),
                              torch.tensor(cor_inicial(pts, treino), dtype=torch.float32))
    opt = torch.optim.Adam([
        {"params": [ator.pos], "lr": 2e-3},
        {"params": [ator.log_esc], "lr": 1e-2},
        {"params": [ator.quat], "lr": 2e-3},
        {"params": [ator.cor_logit], "lr": 3e-2},
        {"params": [ator.op_logit], "lr": 3e-2},
    ])
    sched = torch.optim.lr_scheduler.LambdaLR(opt, lambda i: 0.1 ** (i / ITERS))
    alvos = [torch.tensor(img) for _, img, _ in treino]
    t1 = time.time()
    for it in range(ITERS):
        k = rng.integers(len(treino))
        (R, t), _, _ = treino[k]
        img = splat3d.renderizar(ator.valores(), R, t, RES, FOCO)
        loss = (img - alvos[k]).abs().mean() + 0.5 * ((img - alvos[k]) ** 2).mean()
        opt.zero_grad(); loss.backward(); opt.step(); sched.step()
        if it % 250 == 0 or it == ITERS - 1:
            print(f"  passo {it:5d}  erro {loss.item():.4f}  ({time.time() - t1:.0f} s)")
    t_treino = time.time() - t1

    # avaliação em ângulos novos
    linhas_cmp, ps_teste, ps_treino = [], [], []
    with torch.no_grad():
        v = ator.valores()
        for (R, t), ref, _ in teste:
            out = splat3d.renderizar(v, R, t, RES, FOCO).numpy()
            ps_teste.append(psnr(out, ref))
            linhas_cmp.append(np.concatenate([ref, out], 1))
        for (R, t), ref, _ in treino[::3]:
            ps_treino.append(psnr(splat3d.renderizar(v, R, t, RES, FOCO).numpy(), ref))
    cmp = np.concatenate(linhas_cmp, 0)
    Image.fromarray((np.clip(cmp, 0, 1) * 255).astype(np.uint8)).resize((cmp.shape[1] * 2, cmp.shape[0] * 2),
                                                                         Image.NEAREST).save(os.path.join(SAIDA, "ator3d_angulos_novos.png"))
    torch.save({k: p.detach() for k, p in ator.state_dict().items()}, os.path.join(SAIDA, "ator3d.pt"))
    rel = (f"# Ator 3D (H2) — criado a partir de {len(treino)} fotos\n\n"
           f"- gaussianas: {len(pts)}\n- treino: {ITERS} passos em {t_treino / 60:.1f} min (CPU, {RES}x{RES})\n"
           f"- PSNR nas fotos de treino: {np.mean(ps_treino):.1f} dB\n"
           f"- **PSNR em 3 ângulos nunca vistos: {np.mean(ps_teste):.1f} dB** "
           f"({', '.join(f'{p:.1f}' for p in ps_teste)})\n"
           f"- tamanho do arquivo do ator: {os.path.getsize(os.path.join(SAIDA, 'ator3d.pt')) / 1024:.0f} KB\n")
    with open(os.path.join(SAIDA, "ator3d_relatorio.md"), "w") as fh:
        fh.write(rel)
    print(rel)


if __name__ == "__main__":
    main()
