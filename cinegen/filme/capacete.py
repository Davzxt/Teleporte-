"""Close no capacete: rosto gerado (video_real.falante.Rosto) atrás de um visor com reflexo físico.

O visor esconde de propósito os limites do rosto (128 px, só de frente): tom dourado, reflexo curvo
da paisagem com os dois sóis, luz do painel (HUD) por baixo e embaçado da respiração.
"""
import numpy as np
import torch
import torch.nn.functional as Fn
import universo as U

W, H = U.W, U.H


class Capacete:
    def __init__(self, cx=480, cy=300, R=255, vrx=172, vry=150):
        self.cx, self.cy, self.R, self.vrx, self.vry = cx, cy, R, vrx, vry
        yy, xx = np.mgrid[:H, :W].astype(np.float32)
        self.yy, self.xx = yy, xx
        x, y = (xx - cx) / R, (yy - cy) / R
        r2 = x * x + y * y
        self.casco = U.suave(1.0, 0.985, np.sqrt(r2))
        z = np.sqrt(np.clip(1 - r2, 0, 1))
        L = np.array([-0.7, -0.35, 0.62]); L /= np.linalg.norm(L)
        dif = np.clip(x * L[0] + y * L[1] + z * L[2], 0, 1)
        ceu = np.clip(-y * 0.5 + 0.5, 0, 1)
        self.cor_casco = (np.array([0.82, 0.80, 0.76]) * (0.12 + 0.9 * dif[..., None])
                          + np.array([0.10, 0.12, 0.18]) * ceu[..., None]).astype(np.float32)
        vcy = cy - 12
        ev = np.sqrt(((xx - cx) / vrx) ** 2 + ((yy - vcy) / vry) ** 2)
        self.ev, self.vcy = ev, vcy
        self.visor = U.suave(1.0, 0.98, ev)
        self.moldura = U.suave(1.09, 1.07, ev) * (1 - self.visor)
        # coordenadas "olho de peixe" para o reflexo curvo
        k = 1 + 0.9 * np.clip(ev, 0, 1) ** 2
        rx = cx - (xx - cx) * 1.35 * k          # espelhado na horizontal
        ry = vcy + (yy - vcy) * 1.35 * k - 60
        self.grade_ref = torch.tensor(np.stack([rx / (W - 1) * 2 - 1, ry / (H - 1) * 2 - 1], -1))[None].float()
        self.fresnel = (0.04 + 0.5 * np.clip(ev, 0, 1) ** 4)[..., None]
        rng = np.random.default_rng(4)
        self.hud = [(rng.uniform(-0.82, -0.6), rng.uniform(-0.15, 0.35), rng.uniform(8, 26)) for _ in range(9)]

    def render(self, fundo, rosto128, t, dx=0.0, dy=0.0, embacado=0.0):
        """fundo: paisagem (linear, H x W x 3); rosto128: quadro do rosto (sRGB 0..1, 128 x 128 x 3)."""
        img = U.desfocar(fundo, 7) * 0.9                       # profundidade de campo
        # rosto: amplia, centraliza no visor
        tam = int(self.vry * 1.75)
        rost = torch.tensor(np.clip(rosto128, 0, 1) ** 2.2, dtype=torch.float32).permute(2, 0, 1)[None]
        rost = Fn.interpolate(rost, size=(tam, tam), mode="bicubic", align_corners=False)[0].permute(1, 2, 0).numpy()
        camada = np.zeros_like(img)
        x0, y0 = int(self.cx - tam / 2 + dx), int(self.vcy - tam / 2 + 8 + dy)
        xa, ya, xb, yb = max(x0, 0), max(y0, 0), min(x0 + tam, W), min(y0 + tam, H)
        gy, gx = np.mgrid[:tam, :tam].astype(np.float32) / (tam - 1) * 2 - 1
        alfa = U.suave(1.0, 0.72, np.sqrt(gx ** 2 * 1.1 + gy ** 2))[..., None]     # bordas somem no escuro
        rost = rost * alfa + np.array([0.02, 0.018, 0.016]) * (1 - alfa)
        camada[:] = np.array([0.02, 0.018, 0.016])
        camada[ya:yb, xa:xb] = rost[ya - y0:yb - y0, xa - x0:xb - x0]
        # luz no rosto: quente da esquerda (sóis), ciano do painel por baixo, interior escuro nas bordas
        luz = 0.55 + 0.6 * np.clip((self.cx - self.xx) / self.vrx * 0.5 + 0.5, 0, 1)
        painel = np.exp(-((self.yy - (self.vcy + self.vry)) / 70) ** 2)[..., None] * np.array([0.05, 0.16, 0.22])
        vinh = np.clip(1.25 - self.ev, 0, 1)[..., None]
        dentro = camada * luz[..., None] * vinh * np.array([1.0, 0.92, 0.8]) + painel
        dentro = dentro * np.array([0.85, 0.78, 0.62])          # filme dourado do visor
        # reflexo da paisagem no vidro curvo + dois sóis refletidos
        ref = Fn.grid_sample(torch.tensor(fundo).permute(2, 0, 1)[None], self.grade_ref, align_corners=True,
                             padding_mode="border")[0].permute(1, 2, 0).numpy()
        vidro = dentro * (1 - self.fresnel) + ref * self.fresnel * 0.9
        for (sx, sy, r, f) in ((-0.45, -0.55, 7, 3.0), (-0.32, -0.45, 5, 1.6)):
            vidro = U.brilho(vidro, self.cx + sx * self.vrx, self.vcy + sy * self.vry, r, (1.2, 1.05, 0.85), f)
        faixa = np.exp(-(((self.xx - self.cx + 0.3 * self.vrx) * 0.7 + (self.yy - self.vcy + 0.55 * self.vry)) / 14) ** 2)
        vidro += (faixa * U.suave(1.0, 0.6, self.ev))[..., None] * 0.12
        # HUD refletido (texto espelhado = barrinhas), piscando
        for i, (hx, hy, comp) in enumerate(self.hud):
            if (np.sin(t * 3 + i * 1.7) > -0.6):
                px, py = self.cx + hx * self.vrx, self.vcy + hy * self.vry
                m = (np.abs(self.yy - py) < 1.5) & (np.abs(self.xx - px) < comp / 2)
                vidro[m] += np.array([0.10, 0.45, 0.55])
        # embaçado da respiração
        if embacado > 0:
            nevoa = np.exp(-(((self.xx - self.cx - dx) / 75) ** 2 + ((self.yy - self.vcy - 95 - dy) / 38) ** 2)) * embacado
            vidro = vidro * (1 - nevoa[..., None] * 0.5) + nevoa[..., None] * np.array([0.5, 0.5, 0.5])
        v = self.visor[..., None]
        img = img * (1 - self.casco[..., None]) + self.cor_casco * self.casco[..., None]
        img = img * (1 - v) + vidro * v
        img = img * (1 - self.moldura[..., None] * 0.9) + self.moldura[..., None] * np.array([0.10, 0.10, 0.11])
        # luzinha lateral do capacete
        img = U.brilho(img, self.cx + self.R * 0.86, self.cy - self.R * 0.25, 3, (3, 0.5, 0.3), 0.6 + 0.4 * np.sin(t * 6))
        return img.astype(np.float32)
