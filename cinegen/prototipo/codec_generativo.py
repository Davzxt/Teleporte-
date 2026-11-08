"""H4 — Codec generativo: quanto do vídeo cabe em vetores de movimento?

Mede, em quadros reais renderizados, quantos números um modelo precisaria prever
(quadro-chave + vetores de movimento por bloco) contra prever todos os pixels,
e quanta qualidade (PSNR) só o movimento já reconstrói.
"""
import numpy as np


def _cinza(f):
    return f.astype(np.float32).mean(2)


def vetores_movimento(ant, atual, bloco=8, raio=6):
    H, W = atual.shape
    H, W = H - H % bloco, W - W % bloco
    ant, atual = ant[:H, :W], atual[:H, :W]
    pad = np.pad(ant, raio, mode="edge")
    melhor = np.full((H // bloco, W // bloco), np.inf, np.float32)
    mv = np.zeros((H // bloco, W // bloco, 2), np.int8)
    for dy in range(-raio, raio + 1):
        for dx in range(-raio, raio + 1):
            cand = pad[raio + dy: raio + dy + H, raio + dx: raio + dx + W]
            sad = np.abs(atual - cand).reshape(H // bloco, bloco, W // bloco, bloco).sum((1, 3))
            m = sad < melhor
            melhor[m] = sad[m]
            mv[m] = (dy, dx)
    return mv


def prever(ant, mv, bloco=8, raio=6):
    H, W = mv.shape[0] * bloco, mv.shape[1] * bloco
    pad = np.pad(ant[:H, :W], [(raio, raio), (raio, raio), (0, 0)], mode="edge")
    out = np.empty((H, W, ant.shape[2]), ant.dtype)
    for by in range(mv.shape[0]):
        for bx in range(mv.shape[1]):
            dy, dx = int(mv[by, bx, 0]), int(mv[by, bx, 1])
            y, x = by * bloco, bx * bloco
            out[y:y + bloco, x:x + bloco] = pad[raio + dy + y: raio + dy + y + bloco, raio + dx + x: raio + dx + x + bloco]
    return out


def psnr(a, b):
    mse = np.mean((a.astype(np.float32) - b.astype(np.float32)) ** 2)
    return 99.0 if mse == 0 else 10 * np.log10(255.0 ** 2 / mse)


def avaliar(quadros, bloco=8):
    """quadros: lista de uint8 HxWx3 de UM plano contínuo. Retorna estatísticas."""
    pix = quadros[0].size
    ps, nums = [], []
    for ant, at in zip(quadros[:-1], quadros[1:]):
        mv = vetores_movimento(_cinza(ant), _cinza(at), bloco)
        pred = prever(ant, mv, bloco)
        ps.append(psnr(pred, at[: pred.shape[0], : pred.shape[1]]))
        nums.append(mv.size)
    return dict(numeros_por_quadro_pixels=pix, numeros_por_quadro_movimento=int(np.mean(nums)),
                reducao=pix / np.mean(nums), psnr_medio_so_movimento=float(np.mean(ps)))
