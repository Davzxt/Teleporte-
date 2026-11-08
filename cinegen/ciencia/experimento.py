"""Roda um ou mais métodos, mede tudo e registra no caderno de laboratório.

Uso: python3 experimento.py pca pex ...
"""
import json
import os
import sys
import time
import numpy as np
from PIL import Image

import dados
import metricas
import metodos

AQUI = os.path.dirname(os.path.abspath(__file__))
RES_DIR = os.path.join(AQUI, "resultados")

METODOS = {
    "pca": lambda: metodos.PCAGauss(150),
    "pex": lambda: metodos.PiramideExemplos(),
    "pex_d64": lambda: metodos.PiramideExemplos(doadores=64),
    "pex_d16": lambda: metodos.PiramideExemplos(doadores=16),
    "pex_gmm_d16": lambda: metodos.PiramideExemplos(doadores=16, base="gmm"),
    "pex_viz_d16": lambda: metodos.PiramideExemplos(doadores=16, base="viz"),
    "foto_viz": lambda: metodos.Fotomontagem(base="viz"),
    "foto_gmm": lambda: metodos.Fotomontagem(base="gmm"),
    "foto_gmm_alin": lambda: metodos.Fotomontagem(base="gmm", alinhar=3),
    "foto_gmm160": lambda: metodos.Fotomontagem(base="gmm", grupos=160),
    "foto_gmm400": lambda: metodos.Fotomontagem(base="gmm", grupos=400),
    "foto_gmm1000": lambda: metodos.Fotomontagem(base="gmm", grupos=1000),
    "foto_gmm1000_t07": lambda: metodos.Fotomontagem(base="gmm", grupos=1000, temp=0.7),
    "foto_gmm2000": lambda: metodos.Fotomontagem(base="gmm", grupos=2000),
}


def grade(X, nome, lin=3, col=8):
    X = np.clip(X[:lin * col], 0, 1)
    g = np.concatenate([np.concatenate(list(X[i * col:(i + 1) * col]), 1) for i in range(lin)], 0)
    Image.fromarray((g * 255).astype(np.uint8)).resize((g.shape[1] * 2, g.shape[0] * 2), Image.LANCZOS) \
        .save(os.path.join(RES_DIR, f"{nome}.png"))


def main(nomes):
    os.makedirs(RES_DIR, exist_ok=True)
    tr, te = dados.carregar()
    for nome in nomes:
        rng = np.random.default_rng(1)
        m = METODOS[nome]()
        t0 = time.time(); m.treinar(tr); tt = time.time() - t0
        t0 = time.time(); G = np.concatenate([m.gerar(500, rng) for _ in range(6)]); tg = (time.time() - t0) / len(G)
        r = metricas.avaliar(nome, tr, te, G, tt, tg)
        grade(G, nome)
        print(json.dumps(r, ensure_ascii=False))
        with open(os.path.join(RES_DIR, "caderno.jsonl"), "a") as fh:
            fh.write(json.dumps(r, ensure_ascii=False) + "\n")


if __name__ == "__main__":
    main(sys.argv[1:])
