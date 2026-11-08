"""Renderizador 3D Gaussian Splatting diferenciável (PyTorch puro).

Cada gaussiana: posição 3D, escala 3D, rotação (quatérnio), cor e opacidade.
Projeção EWA (covariância 3D → 2D pela jacobiana da perspectiva), ordenação por profundidade
e composição alfa. Recorte por tiles de 16x16 pixels (cada tile só processa as gaussianas que o tocam).
Para filmes em 1080p use um rasterizador CUDA (ex.: gsplat), com a mesma matemática.
Roda em CPU aqui; no seu PC basta device="cuda".
"""
import torch


def quat_para_matriz(q):
    q = q / q.norm(dim=-1, keepdim=True)
    w, x, y, z = q.unbind(-1)
    return torch.stack([
        1 - 2 * (y * y + z * z), 2 * (x * y - w * z), 2 * (x * z + w * y),
        2 * (x * y + w * z), 1 - 2 * (x * x + z * z), 2 * (y * z - w * x),
        2 * (x * z - w * y), 2 * (y * z + w * x), 1 - 2 * (x * x + y * y),
    ], -1).reshape(*q.shape[:-1], 3, 3)


def quat_mult(a, b):
    aw, ax, ay, az = a.unbind(-1)
    bw, bx, by, bz = b.unbind(-1)
    return torch.stack([aw * bw - ax * bx - ay * by - az * bz,
                        aw * bx + ax * bw + ay * bz - az * by,
                        aw * by - ax * bz + ay * bw + az * bx,
                        aw * bz + ax * by - ay * bx + az * bw], -1)


class Gaussianas(torch.nn.Module):
    def __init__(self, pos, cor):
        super().__init__()
        n = pos.shape[0]
        self.pos = torch.nn.Parameter(pos.clone())
        self.log_esc = torch.nn.Parameter(torch.full((n, 3), -3.2))
        q = torch.zeros(n, 4); q[:, 0] = 1
        self.quat = torch.nn.Parameter(q)
        self.cor_logit = torch.nn.Parameter(torch.logit(cor.clamp(0.02, 0.98)))
        self.op_logit = torch.nn.Parameter(torch.full((n,), 1.0))

    def valores(self):
        return dict(pos=self.pos, esc=self.log_esc.exp(), quat=self.quat,
                    cor=torch.sigmoid(self.cor_logit), op=torch.sigmoid(self.op_logit))


def renderizar(g, R, t, res, foco=2.6, fundo=(0.08, 0.08, 0.09), tile=16):
    """g: dict(pos, esc, quat, cor, op). R (3x3), t (3) mundo→câmera.
    res: int (quadrado) ou (largura, altura). Retorna imagem (altura, largura, 3)."""
    W, H = (res, res) if isinstance(res, int) else res
    dev = g["pos"].device
    R = torch.as_tensor(R, dtype=torch.float32, device=dev)
    t = torch.as_tensor(t, dtype=torch.float32, device=dev)
    f = foco * H / 2
    pc = g["pos"] @ R.T + t
    z = pc[:, 2].clamp(min=0.2)
    ordem = torch.argsort(z)                          # da frente para trás
    pc, z = pc[ordem], z[ordem]
    Rg = quat_para_matriz(g["quat"][ordem])
    M = Rg * g["esc"][ordem][:, None, :]
    cov3 = R @ (M @ M.transpose(1, 2)) @ R.T
    J = torch.zeros(len(z), 2, 3, device=dev)
    J[:, 0, 0] = f / z; J[:, 0, 2] = -f * pc[:, 0] / z ** 2
    J[:, 1, 1] = f / z; J[:, 1, 2] = -f * pc[:, 1] / z ** 2
    cov2 = J @ cov3 @ J.transpose(1, 2) + 0.3 * torch.eye(2, device=dev)
    a, b, c = cov2[:, 0, 0], cov2[:, 0, 1], cov2[:, 1, 1]
    det = (a * c - b * b).clamp(min=1e-8)
    ia, ib, ic = c / det, -b / det, a / det
    mu = torch.stack([f * pc[:, 0] / z + W / 2, f * pc[:, 1] / z + H / 2], -1)
    cor, op = g["cor"][ordem], g["op"][ordem]

    # raio de influência (3 sigmas) para recortar por blocos de pixels (tiles)
    tr = (a + c) / 2
    raio = 3 * (tr + (tr * tr - det).clamp(min=0).sqrt()).sqrt()
    fundo = torch.tensor(fundo, device=dev)
    img = torch.empty(H, W, 3, device=dev)
    lado = tile
    for y0 in range(0, H, lado):
        for x0 in range(0, W, lado):
            y1, x1 = min(y0 + lado, H), min(x0 + lado, W)
            dentro = ((mu[:, 0] + raio > x0) & (mu[:, 0] - raio < x1) &
                      (mu[:, 1] + raio > y0) & (mu[:, 1] - raio < y1))
            idx = dentro.nonzero().squeeze(1)
            if idx.numel() == 0:
                img[y0:y1, x0:x1] = fundo
                continue
            ys, xs = torch.meshgrid(torch.arange(y0, y1, device=dev) + 0.5,
                                    torch.arange(x0, x1, device=dev) + 0.5, indexing="ij")
            d = torch.stack([xs.reshape(-1), ys.reshape(-1)], -1)[None] - mu[idx][:, None]
            pot = -0.5 * (ia[idx, None] * d[..., 0] ** 2 + 2 * ib[idx, None] * d[..., 0] * d[..., 1]
                          + ic[idx, None] * d[..., 1] ** 2)
            alfa = (op[idx, None] * pot.clamp(max=0).exp()).clamp(max=0.99)
            T = torch.cumprod(torch.cat([torch.ones_like(alfa[:1]), 1 - alfa[:-1]], 0), 0)
            col = (T * alfa).T @ cor[idx] + (T[-1] * (1 - alfa[-1]))[:, None] * fundo
            img[y0:y1, x0:x1] = col.reshape(y1 - y0, x1 - x0, 3)
    return img
