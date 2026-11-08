"""Renderizadores procedurais do mundo (H1 + H9): nada aqui é aprendido nem difusão.

Cada função devolve uma imagem em luz LINEAR (float32, H x W x 3) pronta para a película (pos.py).
Tudo é determinístico a partir de sementes: o mesmo planeta em qualquer plano/episódio (H2).
"""
import numpy as np
import torch
import torch.nn.functional as Fn

W, H = 960, 540


# ------------------------------------------------------------------ ruídos
def fbm2d(h, w, oitavas=6, seed=0, base=4, ganho=0.5):
    g = torch.Generator().manual_seed(seed)
    out = torch.zeros(h, w); amp, tot = 1.0, 0.0
    for o in range(oitavas):
        n = base * 2 ** o
        r = torch.rand(1, 1, max(n * h // w, 2) + 1, n + 1, generator=g)
        out += amp * Fn.interpolate(r, size=(h, w), mode="bicubic", align_corners=True)[0, 0]
        tot += amp; amp *= ganho
    return (out / tot).numpy()


def fbm1d(n, oitavas=7, seed=0, base=6):
    rng = np.random.default_rng(seed)
    x = np.linspace(0, 1, n); out = np.zeros(n); amp = 1.0; tot = 0.0
    for o in range(oitavas):
        k = base * 2 ** o
        pts = rng.random(k + 1)
        out += amp * np.interp(x * k, np.arange(k + 1), pts); tot += amp; amp *= 0.5
    return out / tot


def suave(a, b, x):
    t = np.clip((x - a) / (b - a), 0, 1)
    return t * t * (3 - 2 * t)


def brilho(img, cx, cy, raio, cor, forca=1.0):
    yy, xx = np.ogrid[:img.shape[0], :img.shape[1]]
    d2 = (xx - cx) ** 2 + (yy - cy) ** 2
    img += (forca * np.exp(-d2 / (2 * raio ** 2)))[..., None] * np.asarray(cor, np.float32)
    return img


def desfocar(img, r):
    t = torch.tensor(img).permute(2, 0, 1)[None]
    k = 2 * r + 1
    for _ in range(2):
        t = Fn.avg_pool2d(Fn.pad(t, (r, r, r, r), mode="replicate"), k, 1)
    return t[0].permute(1, 2, 0).numpy()


# ------------------------------------------------------------------ espaço profundo
class Ceu:
    def __init__(self, seed=1, h=900, w=2400):
        rng = np.random.default_rng(seed)
        img = np.zeros((h, w, 3), np.float32)
        n = 9000
        ys, xs = rng.integers(0, h, n), rng.integers(0, w, n)
        b = np.clip(rng.pareto(2.2, n) * 0.05, 0, 6)
        temp = rng.random(n)
        cor = np.stack([0.8 + 0.2 * temp, 0.85 + 0.1 * temp, 1.0 - 0.25 * temp], 1)
        np.add.at(img, (ys, xs), cor * b[:, None])
        estrelas = desfocar(img, 1) * 3 + desfocar(img, 4) * 0.6
        neb = suave(0.45, 0.8, fbm2d(h, w, 7, seed + 1, base=3))
        tom = fbm2d(h, w, 4, seed + 2, base=2)
        nebcor = (np.array([0.10, 0.25, 0.35]) * (1 - tom[..., None]) + np.array([0.35, 0.08, 0.25]) * tom[..., None])
        self.img = (estrelas + neb[..., None] * nebcor * 0.35).astype(np.float32)

    def recorte(self, ox, oy):
        ox, oy = int(ox) % (self.img.shape[1] - W), int(oy) % (self.img.shape[0] - H)
        return self.img[oy:oy + H, ox:ox + W].copy()


# ------------------------------------------------------------------ planeta visto da órbita
class Planeta:
    def __init__(self, seed=3):
        th, tw = 512, 1024
        alt = fbm2d(th, tw, 7, seed, base=14)
        lat = np.abs(np.linspace(-1, 1, th))[:, None]
        terra = np.array([0.55, 0.28, 0.14]) * (1 - alt[..., None]) + np.array([0.85, 0.62, 0.40]) * alt[..., None]
        mar = suave(0.42, 0.38, alt)[..., None]
        terra = terra * (1 - mar) + np.array([0.12, 0.18, 0.22]) * mar
        gelo = suave(0.8, 0.9, lat + 0.1 * alt)[..., None]
        self.tex = torch.tensor(terra * (1 - gelo) + 0.9 * gelo, dtype=torch.float32).permute(2, 0, 1)[None]
        nuv = suave(0.55, 0.8, fbm2d(th, tw, 7, seed + 9, base=18))
        self.nuvens = torch.tensor(nuv, dtype=torch.float32)[None, None]

    def render(self, t, ceu, push=0.0):
        img = ceu.recorte(300 + 8 * t, 120)
        R = W * (0.95 + 0.08 * push)
        cx, cy = W * 0.5, H * 0.5 + R * 0.86 - 30 * push
        yy, xx = np.mgrid[:H, :W].astype(np.float32)
        x, y = (xx - cx) / R, (yy - cy) / R
        r2 = x * x + y * y
        dentro = r2 < 1
        z = np.sqrt(np.clip(1 - r2, 0, 1))
        # inclina a esfera para vermos a borda de cima de frente
        lon = np.arctan2(x, z + 1e-6) * 0.5 + 0.01 * t      # z >= 0 no lado visível: sem emenda
        lat = np.arcsin(np.clip(-y, -1, 1)) * 0.9
        g = torch.tensor(np.stack([lon / np.pi, -lat / (np.pi / 2)], -1))[None].float()
        cor = Fn.grid_sample(self.tex, g, align_corners=True, padding_mode="border")[0].permute(1, 2, 0).numpy()
        g2 = torch.tensor(np.stack([((lon + 0.012 * t + np.pi) % (2 * np.pi) - np.pi) / np.pi, -lat / (np.pi / 2)], -1))[None].float()
        nv = Fn.grid_sample(self.nuvens, g2, align_corners=True, padding_mode="border")[0, 0].numpy()
        L = np.array([-0.55, -0.62, -0.1]); L /= np.linalg.norm(L)
        luz = np.clip(x * L[0] + y * L[1] + z * L[2] + 0.12, 0, 1) ** 0.8
        sup = (cor * (1 - nv[..., None]) + 0.95 * nv[..., None]) * luz[..., None] * 0.9
        borda = np.exp(-z / 0.045)[..., None]
        atm = np.array([0.35, 0.6, 1.1], np.float32)
        sup = sup * (1 - 0.7 * borda) + atm * borda * (luz[..., None] * 1.3 + 0.03)
        img = np.where(dentro[..., None], sup, img)
        # atmosfera fora do disco (espalhamento)
        rr = np.sqrt(r2)
        halo = np.exp(-np.clip(rr - 1, 0, None) / 0.015) * (~dentro)
        lado = np.clip((x * L[0] + y * L[1]) / (rr + 1e-6) + 0.3, 0, 1)
        img += (halo * lado)[..., None] * atm * 1.4
        # sol nascendo na borda
        sx, sy = cx + R * (-0.42 + 0.004 * t), cy - R * 0.915
        img = brilho(img, sx, sy, 6, (6, 5.5, 4.5), 1)
        img = brilho(img, sx, sy, 60, (1.2, 0.9, 0.6), 0.8)
        img = brilho(img, sx, sy, 260, (0.4, 0.35, 0.3), 0.6)
        faixa = np.exp(-((yy - sy) ** 2) / 6) * np.exp(-np.abs(xx - sx) / 350)
        img += faixa[..., None] * np.array([0.6, 0.75, 1.2]) * 0.8          # reflexo anamórfico
        return img.astype(np.float32)


# ------------------------------------------------------------------ superfície do planeta
class Superficie:
    CAMADAS = [  # (altura base da crista, amplitude, px por amostra, paralaxe, névoa)
        (0.58, 0.18, 0.7, 0.15, 0.78),
        (0.63, 0.16, 1.0, 0.30, 0.58),
        (0.70, 0.14, 1.4, 0.50, 0.38),
        (0.80, 0.15, 1.8, 0.75, 0.16),
        (0.92, 0.13, 2.2, 1.00, 0.00),
    ]
    MORRO = 2   # camada onde está o monólito

    def __init__(self, seed=5):
        self.cristas = [1 - np.abs(2 * fbm1d(9000, 8, seed + i, base=24) - 1) ** 1.3 for i in range(len(self.CAMADAS))]
        self.rocha = fbm2d(H, W * 2, 7, seed + 20, base=8)
        self.rocha2 = fbm2d(H, W * 2, 5, seed + 21, base=40)
        rng = np.random.default_rng(seed)
        self.poeira = rng.random((260, 4))           # x, y, profundidade, fase
        self.gigante_tex = fbm2d(256, 512, 5, seed + 30, base=3)

    def ceu(self, ox):
        yy = np.arange(H, dtype=np.float32)[:, None, None] / H
        horiz = np.array([0.80, 0.48, 0.30]); zen = np.array([0.07, 0.12, 0.24])
        e = suave(0.62, 0.05, yy)
        img = np.broadcast_to(horiz * (1 - e) + zen * e, (H, W, 3)).astype(np.float32).copy()
        return img

    def gigante(self, img, ox, raio=165):
        cx, cy = W * 0.70 - ox * 0.04, H * 0.27
        yy, xx = np.mgrid[:H, :W].astype(np.float32)
        x, y = (xx - cx) / raio, (yy - cy) / raio
        r2 = x * x + y * y
        z = np.sqrt(np.clip(1 - r2, 0, 1))
        dentro = r2 < 1
        lat = np.arcsin(np.clip(y, -1, 1))
        ty = ((lat / np.pi + 0.5) * 255).astype(int).clip(0, 255)
        tx = (((np.arctan2(x, z + 1e-6) / np.pi + 1) * 120 + 40) % 512).astype(int)
        perturb = self.gigante_tex[ty, tx]
        faixas = 0.5 + 0.5 * np.sin(lat * 18 + perturb * 5)
        cor = np.array([0.95, 0.78, 0.55]) * faixas[..., None] + np.array([0.6, 0.38, 0.25]) * (1 - faixas[..., None])
        L = np.array([-0.75, -0.25, 0.6]); L /= np.linalg.norm(L)
        luz = np.clip(x * L[0] + y * L[1] + z * L[2], 0, 1) ** 0.9 + 0.03
        disco = cor * luz[..., None] * (0.55 + 0.45 * z[..., None])
        # anéis inclinados: atrás do planeta (metade de cima) e na frente (metade de baixo)
        ang = np.radians(-14)
        rx_, ry_ = x * np.cos(ang) + y * np.sin(ang), -x * np.sin(ang) + y * np.cos(ang)
        rr = np.sqrt(rx_ ** 2 + (ry_ / 0.22) ** 2)
        anel = suave(1.35, 1.42, rr) * suave(2.35, 2.2, rr) * (0.55 + 0.45 * np.sin(rr * 40) ** 2)
        anel *= 1 - 0.75 * suave(1.75, 1.8, rr) * suave(1.9, 1.85, rr)        # divisão escura
        cor_anel = np.array([0.95, 0.85, 0.7]) * 0.9
        atras = (ry_ < 0) & dentro
        a = (anel * (~atras))[..., None] * 0.85
        img = np.where(dentro[..., None], img * 0.15 + disco, img)
        img = img * (1 - a) + cor_anel * a * 1.1
        return img

    def sois(self, img, ox):
        for (fx, fy, r, cor, f) in ((0.18, 0.40, 9, (9, 8, 6.5), 1.0), (0.29, 0.47, 6, (8, 4.5, 2.2), 0.8)):
            sx, sy = W * fx - ox * 0.03, H * fy
            img = brilho(img, sx, sy, r * 0.6, cor, f)
            img = brilho(img, sx, sy, r * 6, np.array(cor) * 0.08, f)
            img = brilho(img, sx, sy, r * 30, np.array(cor) * 0.018, f)
        return img

    def altura(self, i, ox):
        base, amp, esc, par, _ = self.CAMADAS[i]
        xs = (np.arange(W) + ox * par) / esc + 2000
        return (base - amp * self.cristas[i][(xs.astype(int)) % 9000]) * H

    def render(self, ox, t, monolito=0.0, astronauta=None, oy=0.0):
        img = self.ceu(ox)
        img = self.sois(img, ox)
        img = self.gigante(img, ox)
        yy = np.arange(H, dtype=np.float32)[:, None]
        horiz = np.array([0.78, 0.47, 0.30], np.float32)
        for i, (base, amp, esc, par, nevoa) in enumerate(self.CAMADAS):
            topo = self.altura(i, ox) + oy * par
            m = suave(-1.2, 1.2, yy - topo[None])[..., None]
            xi = (np.arange(W) + int(ox * par * 1.3)) % (W * 2)
            tex = self.rocha[:, xi] * 0.7 + self.rocha2[:, xi] * 0.3
            rocha = (np.array([0.42, 0.22, 0.13]) * (0.55 + 0.9 * tex[..., None]))
            prof = np.clip((yy - topo[None]) / (H * 0.25), 0, 1)[..., None]
            rocha = rocha * (1.0 - 0.45 * prof)                          # base das encostas mais escura
            incl = np.convolve(np.gradient(topo), np.ones(25) / 25, mode="same")
            iluminada = np.clip(-incl * 1.2 + 0.3, 0, 1)[None, :, None]   # encostas viradas aos sóis (suavizado)
            aro = np.exp(-np.clip(yy - topo[None], 0, None) / 2.5)[..., None] * np.array([1.0, 0.7, 0.45]) * 0.9 * iluminada
            rocha = rocha * (0.75 + 0.5 * iluminada)
            cor = (rocha + aro * (1 - nevoa)) * (1 - nevoa) + horiz * nevoa * 0.9
            img = img * (1 - m) + cor * m
            if i == self.MORRO and monolito > 0:
                img = self.desenhar_monolito(img, ox, t, monolito, topo)
            if astronauta is not None and astronauta["camada"] == i:
                img = desenhar_astronauta(img, **{k: v for k, v in astronauta.items() if k != "camada"})
        return self.particulas(img, ox, t).astype(np.float32)

    def desenhar_monolito(self, img, ox, t, intensidade, topo):
        par = self.CAMADAS[self.MORRO][3]
        mx = int(W * 0.5 + 260 - ox * par)
        if not (0 <= mx < W):
            return img
        base = topo[mx] + 3
        h, w = 150, 20
        yy, xx = np.mgrid[:H, :W].astype(np.float32)
        corpo = suave(1.2, 0.0, np.abs(xx - mx) - w / 2) * suave(1.2, 0, yy - base) * suave(-1.2, 0, yy - (base - h))
        pulso = 0.65 + 0.35 * np.sin(t * 2.3)
        linhas = (np.abs(((yy - base) % 30) - 15) < 0.7) * corpo * pulso * intensidade * 0.6
        img = brilho(img, mx, base - h * 0.5, 160, (0.04, 0.15, 0.28), intensidade * pulso)   # halo atrás
        img = img * (1 - corpo[..., None] * 0.97) + corpo[..., None] * np.array([0.015, 0.015, 0.02])
        aresta = corpo * suave(w / 2 - 2.5, w / 2, np.abs(xx - mx)) * intensidade * pulso
        img += aresta[..., None] * np.array([0.15, 0.6, 1.0]) * 0.8
        img += linhas[..., None] * np.array([0.3, 1.1, 1.6])
        feixe = np.exp(-np.abs(xx - mx) / 2.5) * (yy < base - h) * intensidade * pulso * 0.35
        img += feixe[..., None] * np.array([0.2, 0.7, 1.2])
        return img

    def particulas(self, img, ox, t):
        for x, y, z, f in self.poeira:
            px = (x * W * 1.4 - ox * (0.4 + z) + t * 40 * (0.3 + z)) % (W * 1.4) - W * 0.2
            py = y * H + 8 * np.sin(t * 0.7 + f * 6)
            if 0 <= px < W and 0 <= py < H:
                r = 0.6 + 2.5 * z
                x0, x1, y0, y1 = int(max(px - 3 * r, 0)), int(min(px + 3 * r + 1, W)), int(max(py - 3 * r, 0)), int(min(py + 3 * r + 1, H))
                gy, gx = np.ogrid[y0:y1, x0:x1]
                g = np.exp(-((gx - px) ** 2 + (gy - py) ** 2) / (2 * r * r)) * (0.25 / (1 + z))
                img[y0:y1, x0:x1] += g[..., None] * np.array([1.0, 0.75, 0.5])
        return img


def desenhar_astronauta(img, x, y, escala, fase=0.0, andando=False):
    """Silhueta de astronauta (SDF 2D) com luz de contorno vinda dos sóis (à esquerda)."""
    s = escala
    x0, x1, y0, y1 = int(x - 30 * s), int(x + 30 * s), int(y - 110 * s), int(y + 4 * s)
    x0, y0 = max(x0, 0), max(y0, 0); x1, y1 = min(x1, W), min(y1, H)
    if x0 >= x1 or y0 >= y1:
        return img
    yy, xx = np.mgrid[y0:y1, x0:x1].astype(np.float32)
    u, v = (xx - x) / s, (yy - y) / s
    def cap(ax, ay, bx, by, r):
        px, py = u - ax, v - ay; dx, dy = bx - ax, by - ay
        h = np.clip((px * dx + py * dy) / (dx * dx + dy * dy), 0, 1)
        return np.sqrt((px - dx * h) ** 2 + (py - dy * h) ** 2) - r
    bal = np.sin(fase) * (14 if andando else 0)
    d = np.minimum.reduce([
        np.sqrt(u ** 2 + (v + 92) ** 2) - 13,                    # capacete
        cap(0, -76, 0, -42, 14),                                   # tronco
        cap(8, -78, 9, -50, 10),                                   # mochila
        cap(-4, -42, -4 + bal * 0.5, -2, 6.5), cap(4, -42, 4 - bal * 0.5, -2, 6.5),   # pernas
        cap(-11, -72, -14 - bal * 0.3, -46, 5), cap(11, -72, 13 + bal * 0.3, -46, 5),  # braços
    ])
    a = suave(1.0, -1.0, d * s)[..., None]
    nx = np.gradient(d, axis=1)
    aro = (np.clip(-nx * 3, 0, 1) * suave(-1.6 / s, 0, d))[..., None] * np.array([1.2, 0.8, 0.5])
    cor = np.array([0.025, 0.025, 0.03]) + aro
    reg = img[y0:y1, x0:x1]
    img[y0:y1, x0:x1] = reg * (1 - a) + cor * a
    vis = np.exp(-((u + 5) ** 2 + (v + 94) ** 2) / 18)[..., None] * np.array([1.2, 1.0, 0.8])
    img[y0:y1, x0:x1] += vis * a * 0.5
    return img
