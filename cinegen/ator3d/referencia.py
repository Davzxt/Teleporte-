""""Sessão de fotos" de referência.

Sem pessoa real disponível aqui, uma cabeça 3D procedural (SDF + ray marching) faz o papel do
ator fotografado de vários ângulos. Em produção: 20–50 fotos reais (com consentimento) +
poses de câmera estimadas (COLMAP). O treino (treinar_ator.py) só vê as IMAGENS e as poses,
nunca a geometria daqui.
"""
import numpy as np

BG = np.array([0.08, 0.08, 0.09], np.float32)


def _elip(p, c, r):
    q = (p - c) / r
    k0 = np.linalg.norm(q, axis=-1)
    k1 = np.linalg.norm(q / r, axis=-1)
    return k0 * (k0 - 1) / np.maximum(k1, 1e-6)


def _sunion(a, b, k=0.12):
    h = np.clip(0.5 + 0.5 * (b - a) / k, 0, 1)
    return b * (1 - h) + a * h - k * h * (1 - h)


OLHOS = [np.array([s * 0.27, 0.12, 0.66]) for s in (-1, 1)]


def sdf(p):
    d = _elip(p, np.array([0, 0.02, 0]), np.array([0.74, 0.95, 0.84]))
    d = _sunion(d, _elip(p, np.array([0, -0.5, 0.18]), np.array([0.52, 0.48, 0.6])))
    d = _sunion(d, _elip(p, np.array([0, -1.25, -0.08]), np.array([0.33, 0.6, 0.33])), 0.15)
    d = _sunion(d, _elip(p, np.array([0, -0.1, 0.84]), np.array([0.11, 0.21, 0.17])), 0.06)
    for s in (-1, 1):
        d = _sunion(d, _elip(p, np.array([s * 0.74, -0.02, -0.02]), np.array([0.09, 0.2, 0.14])), 0.04)
    for c in OLHOS:
        d = np.minimum(d, np.linalg.norm(p - c, axis=-1) - 0.115)
    return d


def cor(p):
    x, y, z = p[..., 0], p[..., 1], p[..., 2]
    c = np.empty(p.shape, np.float32)
    c[:] = (0.80, 0.58, 0.46)                                      # pele
    c[y < -0.9] = (0.70, 0.50, 0.40)                               # pescoço
    cabelo = (y > 0.38 - 0.25 * np.clip(-z, 0, 1)) | ((z < -0.25) & (y > -0.55))
    c[cabelo] = (0.13, 0.07, 0.04)
    sob = (y > 0.25) & (y < 0.32) & (np.abs(x) > 0.12) & (np.abs(x) < 0.42) & (z > 0.5)
    c[sob] = (0.18, 0.10, 0.06)
    labio = (y > -0.5) & (y < -0.36) & (np.abs(x) < 0.21 - 0.5 * np.abs(y + 0.43)) & (z > 0.55)
    c[labio] = (0.66, 0.30, 0.30)
    c[labio & (np.abs(y + 0.43) < 0.012)] = (0.25, 0.07, 0.08)    # linha da boca
    for ce in OLHOS:
        l = p - ce
        r = np.linalg.norm(l, axis=-1)
        no_olho = r < 0.125
        frente = l[..., 2] / np.maximum(r, 1e-6)
        c[no_olho] = (0.93, 0.92, 0.9)
        c[no_olho & (frente > 0.82)] = (0.30, 0.18, 0.08)
        c[no_olho & (frente > 0.94)] = (0.03, 0.02, 0.02)
    return c


def camera(az_graus, el_graus, dist=4.2, alvo=(0, -0.2, 0)):
    """Câmera orbital. Retorna (R, t) mundo→câmera, convenção: +z para frente, +y para baixo."""
    az, el = np.radians(az_graus), np.radians(el_graus)
    alvo = np.array(alvo, np.float64)
    pos = alvo + dist * np.array([np.sin(az) * np.cos(el), np.sin(el), np.cos(az) * np.cos(el)])
    f = alvo - pos; f /= np.linalg.norm(f)
    r = np.cross(f, [0, 1, 0]); r /= np.linalg.norm(r)
    d = np.cross(f, r)
    R = np.stack([r, d, f])            # linhas: eixos da câmera no mundo
    return R, -R @ pos


def fotografar(R, t, res=96, foco=2.6):
    """Ray marching da cabeça. foco relativo à metade da imagem."""
    pos = -R.T @ t
    j, i = np.meshgrid(np.arange(res) + 0.5, np.arange(res) + 0.5)
    f = foco * res / 2
    dirs_cam = np.stack([(j - res / 2) / f, (i - res / 2) / f, np.ones_like(j)], -1)
    dirs = dirs_cam @ R
    dirs /= np.linalg.norm(dirs, axis=-1, keepdims=True)
    tt = np.full((res, res), 2.0)
    vivo = np.ones((res, res), bool)
    for _ in range(90):
        p = pos + dirs * tt[..., None]
        d = sdf(p)
        tt = np.where(vivo, tt + d * 0.8, tt)
        vivo &= (d > 1e-3) & (tt < 7)
    p = pos + dirs * tt[..., None]
    acerto = sdf(p) < 5e-3
    e = 1e-3
    n = np.stack([sdf(p + np.array(v) * e) - sdf(p - np.array(v) * e)
                  for v in ([1, 0, 0], [0, 1, 0], [0, 0, 1])], -1)
    n /= np.linalg.norm(n, axis=-1, keepdims=True) + 1e-9
    luz = np.array([-0.5, 0.6, 0.65]); luz /= np.linalg.norm(luz)
    amb = 0.32 + 0.15 * n[..., 1]
    dif = np.clip((n * luz).sum(-1), 0, 1) * 0.75
    aro = np.clip(1 + (n * dirs).sum(-1), 0, 1) ** 3 * 0.25
    img = cor(p) * (amb + dif)[..., None] + aro[..., None] * np.array([0.5, 0.55, 0.7])
    out = np.where(acerto[..., None], np.clip(img, 0, 1), BG)
    return out.astype(np.float32), acerto


def sessao(res=96):
    """Treino: 9 azimutes x 2 alturas. Teste: 3 ângulos NUNCA vistos no treino."""
    treino = [(az, el) for el in (-12, 18) for az in range(-80, 81, 20)]
    teste = [(-30, 3), (10, 3), (50, 3)]
    pega = lambda lista: [(camera(a, e), *fotografar(*camera(a, e), res)) for a, e in lista]
    return pega(treino), pega(teste)
