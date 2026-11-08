"""FFHQ (NVIDIA, CC BY-NC-SA 4.0): 4.890 rostos reais 512x512 obtidos de repositórios públicos no GitHub."""
import glob
import os
import numpy as np
from PIL import Image

D = os.environ.get("DADOS_DIR", "/tmp/claude-0/-home-user-Teleporte-/0ffc3a91-1c16-542a-a221-0372f80381d3/scratchpad/dados")


def carregar(res=256, n_teste=400):
    cache = f"{D}/ffhq_{res}.npy"
    if os.path.exists(cache):
        X = np.load(cache)
    else:
        fs = sorted(glob.glob(f"{D}/ffhq512/*/*.png")) + sorted(glob.glob(f"{D}/ffhq_extra/*.png"))
        X = np.stack([np.asarray(Image.open(f).convert("RGB").resize((res, res), Image.LANCZOS)) for f in fs])
        np.save(cache, X)
    idx = np.random.default_rng(0).permutation(len(X))
    X = X[idx]
    return X[:-n_teste], X[-n_teste:]      # uint8 (economiza memória)
