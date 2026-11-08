"""Carrega UTKFace (23.705 rostos reais alinhados) em 64x64. Licença: só pesquisa não comercial."""
import os
import numpy as np
from PIL import Image

PASTA = os.environ.get("UTK_DIR", "/tmp/claude-0/-home-user-Teleporte-/0ffc3a91-1c16-542a-a221-0372f80381d3/scratchpad/dados/utkf/data/UTKFace")
CACHE = os.path.join(os.path.dirname(PASTA), "utk64.npy")


def rotulos():
    """idade, sexo, etnia a partir do nome do arquivo (mesma ordem/embaralhamento de carregar)."""
    arqs = sorted(f for f in os.listdir(PASTA) if f.endswith(".jpg"))
    L = []
    for f in arqs:
        p = f.split("_")
        try:
            L.append((min(int(p[0]), 100), int(p[1]), int(p[2]) if p[2].isdigit() else 4))
        except (ValueError, IndexError):
            L.append((30, 0, 4))
    L = np.array(L, np.float32)[np.random.default_rng(0).permutation(len(L))]
    return L[:-3000], L[-3000:]


def carregar(res=64):
    cache = CACHE.replace("64", str(res))
    if os.path.exists(cache):
        X = np.load(cache)
    else:
        arqs = sorted(f for f in os.listdir(PASTA) if f.endswith(".jpg"))
        X = np.stack([np.asarray(Image.open(os.path.join(PASTA, f)).convert("RGB")
                                 .crop((20, 20, 180, 180)).resize((res, res), Image.LANCZOS)) for f in arqs])
        np.save(cache, X)
    rng = np.random.default_rng(0)
    idx = rng.permutation(len(X))
    X = X[idx].astype(np.float32) / 255
    return X[:-3000], X[-3000:]          # treino, teste (o teste nunca é visto pelos geradores)
