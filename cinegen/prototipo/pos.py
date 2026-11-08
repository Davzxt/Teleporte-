"""H9 — Ótica física e emulação de película: o 'look de cinema' sem rede neural."""
import numpy as np


def _blur(img, r):
    for ax in (0, 1):
        for _ in range(2):
            c = np.cumsum(np.pad(img, [(r + 1, r) if a == ax else (0, 0) for a in range(img.ndim)], mode="edge"), axis=ax)
            img = (np.take(c, range(2 * r + 1, c.shape[ax]), axis=ax) - np.take(c, range(0, c.shape[ax] - 2 * r - 1), axis=ax)) / (2 * r + 1)
    return img


class Pelicula:
    def __init__(self, W, H, seed=11):
        self.rng = np.random.default_rng(seed)
        yy, xx = np.mgrid[0:H, 0:W]
        d = np.sqrt(((xx - W / 2) / (W / 2)) ** 2 + ((yy - H / 2) / (H / 2)) ** 2)
        self.vinheta = (1 - 0.35 * np.clip(d - 0.35, 0, 1) ** 1.5)[..., None].astype(np.float32)
        barra = int(round((H - W / 2.39) / 2))   # proporção 2.39:1
        self.barra = max(barra, 0)

    def __call__(self, lin, exposicao=1.0):
        x = lin * exposicao * 1.1
        # halation: luz forte "vaza" vermelho-alaranjado na emulsão
        peq = x[::4, ::4]
        brilho = np.clip(peq.max(2, keepdims=True) - 0.75, 0, None)
        hal = _blur(brilho, 3)
        hal = np.repeat(np.repeat(hal, 4, 0), 4, 1)[: x.shape[0], : x.shape[1]]
        x = x + hal * np.array([0.9, 0.35, 0.12], np.float32) * 0.35
        x = x * self.vinheta
        # curva fílmica (ACES aproximada)
        x = np.clip((x * (2.51 * x + 0.03)) / (x * (2.43 * x + 0.59) + 0.14), 0, 1)
        # split-toning: sombras frias, altas luzes quentes
        lum = x.mean(2, keepdims=True)
        x = x + (1 - lum) * np.array([-0.012, 0.004, 0.025], np.float32) + lum * np.array([0.03, 0.01, -0.025], np.float32)
        # grão (mais forte nos meios-tons), novo a cada quadro
        g = self.rng.standard_normal(x.shape[:2]).astype(np.float32)[..., None]
        x = x + g * 0.035 * np.sqrt(lum * (1 - lum) + 0.02)
        x = np.clip(x, 0, 1) ** (1 / 1.1)
        if self.barra:
            x[: self.barra] = 0
            x[-self.barra:] = 0
        return (x * 255 + 0.5).astype(np.uint8)
