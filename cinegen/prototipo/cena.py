"""H1 + H2 — Cena-programa e elenco persistente.

O mundo e os atores são feitos de "splats" gaussianos criados UMA vez (com semente fixa).
Cada quadro é apenas renderizado: nada é gerado por quadro.
Renderizador 2.5D em numpy (com paralaxe por profundidade). Em produção: 3D Gaussian Splatting na GPU.
"""
from dataclasses import dataclass
import numpy as np


@dataclass
class Splat:
    x: float
    y: float
    z: float            # profundidade (1 = plano dos atores; maior = mais longe)
    sx: float
    sy: float
    cor: tuple
    a: float = 1.0      # opacidade
    rot: float = 0.0
    p: float = 1.0      # >1 deixa a borda mais "dura" (super-gaussiana)
    add: bool = False   # luz aditiva (brilhos)


@dataclass
class Camera:
    x: float = 0.0
    y: float = 0.0
    zoom: float = 1.0


def renderizar(splats, cam, W, H, fundo=(0.02, 0.02, 0.03)):
    img = np.empty((H, W, 3), np.float32)
    img[:] = fundo
    # do mais distante para o mais próximo; ordem de inserção mantida (sort estável)
    for s in sorted(splats, key=lambda s: -s.z):
        k = cam.zoom * (1.0 + 0.15 * (1.0 / s.z - 1.0))   # objetos longe crescem menos no zoom
        cx = W / 2 + (s.x - cam.x / s.z) * k
        cy = H / 2 + (s.y - cam.y / s.z) * k
        sx, sy = max(s.sx * k, 0.4), max(s.sy * k, 0.4)
        r = 3.2 * max(sx, sy)
        x0, x1 = int(max(cx - r, 0)), int(min(cx + r + 1, W))
        y0, y1 = int(max(cy - r, 0)), int(min(cy + r + 1, H))
        if x0 >= x1 or y0 >= y1:
            continue
        xs = np.arange(x0, x1, dtype=np.float32) - cx
        ys = np.arange(y0, y1, dtype=np.float32)[:, None] - cy
        c, sn = np.cos(s.rot), np.sin(s.rot)
        u = (c * xs + sn * ys) / sx
        v = (-sn * xs + c * ys) / sy
        q = u * u + v * v
        if s.p != 1.0:
            q = q ** s.p
        al = (s.a * np.exp(-0.5 * q))[..., None]
        reg = img[y0:y1, x0:x1]
        cor = np.asarray(s.cor, np.float32)
        if s.add:
            reg += cor * al
        else:
            reg *= (1 - al)
            reg += cor * al
    return img


# ----------------------------------------------------------------- locação persistente
def criar_cozinha(seed=7):
    """Locação criada uma vez. Reusar a mesma semente = mesma locação em qualquer episódio."""
    rng = np.random.default_rng(seed)
    S = []
    # parede e textura de reboco
    S.append(Splat(0, -40, 1.25, 420, 170, (0.10, 0.12, 0.16), 1, p=3))
    for _ in range(220):
        S.append(Splat(rng.uniform(-450, 450), rng.uniform(-200, 110), 1.25,
                       rng.uniform(4, 22), rng.uniform(4, 22),
                       tuple(np.array([0.13, 0.15, 0.19]) * rng.uniform(0.7, 1.3)),
                       rng.uniform(0.1, 0.35), rng.uniform(0, 3)))
    # janela com lua
    S.append(Splat(-200, -70, 1.25, 62, 72, (0.05, 0.05, 0.06), 1, p=4))
    S.append(Splat(-200, -70, 1.25, 55, 65, (0.04, 0.07, 0.14), 1, p=4))
    S.append(Splat(-215, -100, 1.25, 12, 12, (0.85, 0.88, 0.95), 1, p=2))
    S.append(Splat(-215, -100, 1.25, 30, 30, (0.10, 0.14, 0.25), 0.5, add=True))
    S.append(Splat(-200, -70, 1.25, 2.5, 66, (0.05, 0.05, 0.06), 1, p=3))
    S.append(Splat(-200, -70, 1.25, 56, 2.5, (0.05, 0.05, 0.06), 1, p=3))
    # chão de madeira
    S.append(Splat(0, 175, 1.1, 480, 60, (0.16, 0.10, 0.06), 1, p=3))
    for i in range(18):
        S.append(Splat(-450 + i * 52, 175, 1.1, 1.2, 55, (0.08, 0.05, 0.03), 0.6, p=3))
    return S


def porta(tremor=0.0):
    """A porta é um objeto: o evento 'batida' a faz tremer (imagem) e soar (H6)."""
    dx = tremor * 2.5
    return [
        Splat(215 + dx, -10, 1.15, 48, 112, (0.06, 0.04, 0.03), 1, p=4),
        Splat(215 + dx, -10, 1.15, 42, 106, (0.22, 0.13, 0.07), 1, p=4),
        Splat(215 + dx, -55, 1.15, 30, 30, (0.18, 0.10, 0.05), 0.8, p=3),
        Splat(215 + dx, 40, 1.15, 30, 30, (0.18, 0.10, 0.05), 0.8, p=3),
        Splat(185 + dx, 0, 1.15, 4, 4, (0.75, 0.6, 0.3), 1, p=2),
        Splat(215, 98, 1.15, 50, 4, (0.9, 0.7, 0.4), 0.25 + 0.2 * tremor, add=True),  # luz por baixo
    ]


def mesa_e_abajur():
    return [
        Splat(0, 110, 0.95, 130, 14, (0.20, 0.12, 0.07), 1, p=4),
        Splat(0, 160, 0.95, 120, 40, (0.12, 0.07, 0.04), 1, p=4),
        Splat(0, 78, 0.95, 5, 22, (0.15, 0.15, 0.15), 1, p=3),
        Splat(0, 52, 0.95, 22, 13, (0.95, 0.75, 0.45), 1, p=3),
        Splat(0, 55, 0.95, 110, 90, (0.45, 0.28, 0.12), 0.12, add=True),
        Splat(0, 58, 0.95, 30, 22, (0.6, 0.45, 0.25), 0.25, add=True),
        Splat(-40, 92, 0.95, 9, 11, (0.75, 0.75, 0.72), 1, p=2),   # xícara
    ]


# ----------------------------------------------------------------- atores persistentes
ELENCO = {
    "LIA": dict(x=-95, pele=(0.78, 0.55, 0.42), cabelo=(0.10, 0.05, 0.03), roupa=(0.55, 0.15, 0.12),
                cabelo_longo=True, labio=(0.62, 0.28, 0.27)),
    "THEO": dict(x=95, pele=(0.55, 0.37, 0.26), cabelo=(0.05, 0.04, 0.03), roupa=(0.15, 0.25, 0.32),
                 cabelo_longo=False, labio=(0.45, 0.25, 0.22)),
}


def ator(nome, boca_abre=0.0, boca_redonda=0.0, labios=0.0, piscar=0.0, olhar=0.0, respira=0.0):
    e = ELENCO[nome]
    x, pele, ca = e["x"], e["pele"], e["cabelo"]
    hy = -60 + respira * 1.5
    esc = tuple(np.array(pele) * 0.8)
    S = []
    if e["cabelo_longo"]:
        S.append(Splat(x, hy + 22, 1.0, 38, 55, ca, 1, p=3))
    S.append(Splat(x, 62 + respira, 1.0, 50, 80, e["roupa"], 1, p=3.5))      # tronco
    S.append(Splat(x, hy + 42, 1.0, 10, 16, esc, 1, p=3))                    # pescoço
    S.append(Splat(x, hy, 1.0, 29, 35, pele, 1, p=3))                      # cabeça
    lado = -1 if x < 0 else 1                                                # luz do abajur vem do centro
    S.append(Splat(x - lado * 14, hy + 4, 1.0, 16, 30, (0.35, 0.22, 0.1), 0.12, add=True))
    S.append(Splat(x, hy + 6, 1.0, 4, 8, esc, 0.6))                          # nariz
    for d in (-11, 11):                                                      # olhos
        abre = max(1 - piscar, 0.08)
        S.append(Splat(x + d, hy - 6, 1.0, 5.5, 3.6 * abre, (0.92, 0.9, 0.86), 1, p=2.5))
        S.append(Splat(x + d + olhar * 2.8, hy - 6, 1.0, 2.4, 2.6 * abre, (0.08, 0.05, 0.04), 1, p=2.5))
        S.append(Splat(x + d, hy - 14, 1.0, 7, 1.4, ca, 0.9, rot=-0.15 * np.sign(d)))  # sobrancelha
    # boca: controlada pela MESMA trajetória que gera a voz (H5)
    largura = 8.5 * (1 - 0.45 * boca_redonda) + 1.5
    altura = 0.9 + 8.0 * boca_abre * (1 - labios)
    my = hy + 19
    S.append(Splat(x, my, 1.0, largura + 2.2, altura + 2.0, e["labio"], 1, p=2.2))
    S.append(Splat(x, my, 1.0, largura, altura, (0.18, 0.04, 0.05), 1, p=2.2))
    if boca_abre * (1 - labios) > 0.35:
        S.append(Splat(x, my - altura * 0.6, 1.0, largura * 0.7, 1.4, (0.85, 0.82, 0.78), 0.9))
    # cabelo por cima
    S.append(Splat(x, hy - 26, 1.0, 31, 13, ca, 1, p=3))
    if e["cabelo_longo"]:
        S.append(Splat(x - 27, hy + 8, 1.0, 8, 34, ca, 1, p=3))
        S.append(Splat(x + 27, hy + 8, 1.0, 8, 34, ca, 1, p=3))
    return S
