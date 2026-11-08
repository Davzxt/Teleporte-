"""Upgrade HD: Fotomontagem em 256 px treinada em 4.490 rostos FFHQ; teste em 400 nunca vistos."""
import json, os, sys, time
import numpy as np
import torch
from PIL import Image
import dados_hd, metodos, metricas

RES = os.path.join(os.path.dirname(os.path.abspath(__file__)), "resultados")


def novidade(tr, te, G, n=300):
    d = lambda X: torch.tensor(X[:, ::8, ::8].reshape(len(X), -1).astype(np.float32))
    T = d(tr)
    g = torch.cdist(d(G[:n]), T).min(1).values.median()
    r = torch.cdist(d(te[:n]), T).min(1).values.median()
    return float(g / r)


VARIANTES = {
    "base": dict(),
    "A_k8": dict(doadores=8),
    "B_pares": dict(juntar=[[1, 2], [3, 4]]),
    "C_planta32": dict(base_res=32),
    "AB": dict(doadores=8, juntar=[[1, 2], [3, 4]]),
}


def main(grupos_lista):
    tr8, te8 = dados_hd.carregar(256)
    tr = tr8.astype(np.float32) / 255; te = te8.astype(np.float32) / 255
    for nome in grupos_lista:
        g = 150
        rng = np.random.default_rng(1)
        m = metodos.Fotomontagem(base="gmm", grupos=g, regioes=metodos.REGIOES_FFHQ, regioes_res=256, **VARIANTES[nome])
        t0 = time.time(); m.treinar(tr); tt = time.time() - t0
        t0 = time.time(); G = np.clip(np.concatenate([m.gerar(100, rng) for _ in range(4)]), 0, 1); tg = (time.time() - t0) / len(G)
        p = lambda X: metodos.reduzir(X, 64)
        c = [metricas.c2st(p(te), p(G), seed=s) for s in range(3)]
        r = dict(metodo=f"hd_{nome}", c2st_medio=float(np.mean(c)), c2st=[round(x, 3) for x in c],
                 espectro=metricas.espectro(te, G), novidade=novidade(tr8, te8, (G * 255).astype(np.uint8)),
                 treino_s=tt, gerar_ms=1000 * tg)
        print(json.dumps(r, ensure_ascii=False), flush=True)
        with open(os.path.join(RES, "caderno_hd.jsonl"), "a") as fh:
            fh.write(json.dumps(r, ensure_ascii=False) + "\n")
        grade = np.concatenate([np.concatenate(list(G[i * 6:(i + 1) * 6]), 1) for i in range(2)], 0)
        Image.fromarray((grade * 255).astype(np.uint8)).save(os.path.join(RES, f"hd_{nome}.png"))
        del m


if __name__ == "__main__":
    main(sys.argv[1:])
