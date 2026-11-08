"""Harmonizador: rede de 1 passo que desfaz os "remendos" da Fotomontagem.

Treino auto-supervisionado (sem rótulos): para um rosto real X, monta-se a versão remendada
  entrada = planta(X) + detalhes de regiões vindos de OUTRAS pessoas parecidas
e a rede aprende entrada → X. Em uso, recebe a Fotomontagem gerada (planta sorteada) e devolve
uma versão coerente. Saída residual (entrada + correção) para preservar a textura real quando ela já combina.
"""
import os
import sys
import time
import json
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as Fn
from PIL import Image

import dados_hd
import metodos
import metricas

AQUI = os.path.dirname(os.path.abspath(__file__))
RES = os.path.join(AQUI, "resultados")
R = 128
torch.set_num_threads(4)


# ------------------------------------------------------------------ rede
def bloco(i, o):
    return nn.Sequential(nn.Conv2d(i, o, 3, 1, 1), nn.GroupNorm(8, o), nn.SiLU(),
                         nn.Conv2d(o, o, 3, 1, 1), nn.GroupNorm(8, o), nn.SiLU())


class UNet(nn.Module):
    def __init__(self, c=(32, 64, 96, 128)):
        super().__init__()
        self.e = nn.ModuleList([bloco(3, c[0]), bloco(c[0], c[1]), bloco(c[1], c[2]), bloco(c[2], c[3])])
        self.d = nn.ModuleList([bloco(c[3] + c[2], c[2]), bloco(c[2] + c[1], c[1]), bloco(c[1] + c[0], c[0])])
        self.out = nn.Conv2d(c[0], 3, 1)
        nn.init.zeros_(self.out.weight); nn.init.zeros_(self.out.bias)   # começa como identidade

    def forward(self, x):
        s = []
        h = x
        for i, b in enumerate(self.e):
            h = b(h if i == 0 else Fn.avg_pool2d(h, 2))
            s.append(h)
        for i, b in enumerate(self.d):
            h = Fn.interpolate(h, scale_factor=2, mode="nearest")
            h = b(torch.cat([h, s[-2 - i]], 1))
        return x + self.out(h)


def lap(x):
    return x - Fn.avg_pool2d(Fn.pad(x, (2, 2, 2, 2), mode="replicate"), 5, 1)


# ------------------------------------------------------------------ dados de treino (remendos sintéticos)
class Remendos:
    def __init__(self, X, k=8, seed=0):
        self.X = torch.tensor(X).permute(0, 3, 1, 2).float()              # N,3,R,R em [0,1]
        b16 = Fn.interpolate(self.X, size=(16, 16), mode="area")
        self.base = Fn.interpolate(b16, size=(R, R), mode="bicubic", align_corners=False)
        self.det = (self.X - self.base).half()
        g = b16.reshape(len(X), -1)
        d = torch.cat([torch.cdist(g[i:i + 1000], g) for i in range(0, len(g), 1000)])
        d[torch.arange(len(g)), torch.arange(len(g))] = 1e9                # nunca o próprio rosto
        self.knn = d.topk(k, largest=False).indices
        yy, xx = np.mgrid[:R, :R]
        ms = []
        for cy, cx, ry, rx in ((c * R / 256 for c in r) for r in metodos.REGIOES_FFHQ.values()):
            dd = ((yy - cy) / ry) ** 2 + ((xx - cx) / rx) ** 2
            ms.append(1 / (1 + np.exp(np.clip((dd - 1) * 5, -50, 50))))
        M = np.stack(ms); M = np.concatenate([np.clip(1 - M.sum(0), 0, 1)[None], M])
        self.M = torch.tensor(M / M.sum(0, keepdims=True), dtype=torch.float32)[None, :, None]  # 1,R+1,1,H,W
        self.rng = np.random.default_rng(seed)

    def lote(self, n):
        idx = torch.tensor(self.rng.integers(0, len(self.X), n))
        nreg = self.M.shape[1]
        esc = torch.tensor(np.stack([self.rng.permutation(self.knn.shape[1])[:nreg] for _ in range(n)]))
        esc[:, 2] = esc[:, 1]; esc[:, 4] = esc[:, 3]                       # olhos juntos, nariz+boca juntos
        doad = self.knn[idx][torch.arange(n)[:, None], esc]                # n, R+1
        # o "resto" (cabelo/fundo) às vezes vem do próprio rosto, às vezes de doador
        proprio = torch.tensor(self.rng.random(n) < 0.5)
        doad[proprio, 0] = idx[proprio]
        D = self.det[doad].float()                                         # n, R+1, 3, H, W
        entrada = self.base[idx] + (D * self.M).sum(1)
        alvo = self.X[idx]
        if self.rng.random() < 0.5:
            entrada, alvo = entrada.flip(3), alvo.flip(3)
        return entrada.clamp(0, 1), alvo


def treinar(passos=3000, lote=12):
    tr8, te8 = dados_hd.carregar(R)
    tr = tr8.astype(np.float32) / 255
    rem = Remendos(tr)
    net = UNet()
    print("parâmetros:", sum(p.numel() for p in net.parameters()))
    opt = torch.optim.AdamW(net.parameters(), 2e-3, weight_decay=1e-4)
    sched = torch.optim.lr_scheduler.OneCycleLR(opt, 2e-3, total_steps=passos, pct_start=0.05)
    t0 = time.time(); hist = []
    for it in range(passos):
        x, y = rem.lote(lote)
        o = net(x)
        perda = (o - y).abs().mean() + 2.0 * (lap(o) - lap(y)).abs().mean()
        opt.zero_grad(); perda.backward(); opt.step(); sched.step()
        hist.append(perda.item())
        if it % 250 == 0 or it == passos - 1:
            with torch.no_grad():
                base_perda = ((x - y).abs().mean() + 2.0 * (lap(x) - lap(y)).abs().mean()).item()
            print(f"  passo {it:5d}  perda {np.mean(hist[-50:]):.4f}  (sem rede: {base_perda:.4f})  {time.time() - t0:.0f} s", flush=True)
    torch.save(net.state_dict(), os.path.join(RES, "harmonizador.pt"))
    return net


def avaliar(net, sufixo=""):
    tr8, te8 = dados_hd.carregar(R)
    tr = tr8.astype(np.float32) / 255; te = te8.astype(np.float32) / 255
    m = metodos.Fotomontagem(base="gmm", grupos=150, regioes=metodos.REGIOES_FFHQ, regioes_res=256,
                             doadores=8, juntar=[[1, 2], [3, 4]]).treinar(tr)
    rng = np.random.default_rng(5)
    G = np.clip(np.concatenate([m.gerar(100, rng) for _ in range(4)]), 0, 1)
    net.eval()
    with torch.no_grad():
        H = torch.cat([net(torch.tensor(G[i:i + 50]).permute(0, 3, 1, 2)) for i in range(0, len(G), 50)])
    H = H.clamp(0, 1).permute(0, 2, 3, 1).numpy()
    res = {}
    for nome, A in (("fotomontagem", G), ("harmonizada", H)):
        p = lambda X: metodos.reduzir(X, 64)
        c = [metricas.c2st(p(te), p(A), seed=s) for s in range(3)]
        d = lambda X: torch.tensor(X[:, ::4, ::4].reshape(len(X), -1))
        nov = float(torch.cdist(d(A[:300]), d(tr)).min(1).values.median() / torch.cdist(d(te[:300]), d(tr)).min(1).values.median())
        res[nome] = dict(c2st_medio=float(np.mean(c)), c2st=[round(x, 3) for x in c],
                         espectro=metricas.espectro(te, A), novidade=nov)
        print(nome, json.dumps(res[nome]))
    grade = np.concatenate([np.concatenate(list(G[:8]), 1), np.concatenate(list(H[:8]), 1)], 0)
    Image.fromarray((grade * 255).astype(np.uint8)).save(os.path.join(RES, f"harmonizador{sufixo}_antes_depois.png"))
    with open(os.path.join(RES, f"harmonizador{sufixo}.json"), "w") as fh:
        json.dump(res, fh, indent=1)


# ------------------------------------------------------------------ v2: corrige só a escala grossa
class Harmonizador2(nn.Module):
    """Vê a imagem em 32 px, prevê uma correção de 32 px (tom, cor, luz entre as partes) e a soma
    ampliada à imagem cheia: a textura fina real (> 32 px) fica intacta → não borra."""

    def __init__(self):
        super().__init__()
        self.u = UNet(c=(32, 64, 96, 128))

    def forward(self, x):
        b = Fn.interpolate(x, size=(32, 32), mode="area")
        corr = self.u(b) - b
        return x + Fn.interpolate(corr, size=x.shape[-2:], mode="bicubic", align_corners=False)


def treinar2(passos=3000, lote=32):
    tr8, _ = dados_hd.carregar(R)
    rem = Remendos(tr8.astype(np.float32) / 255)
    net = Harmonizador2()
    opt = torch.optim.AdamW(net.parameters(), 2e-3, weight_decay=1e-4)
    sched = torch.optim.lr_scheduler.OneCycleLR(opt, 2e-3, total_steps=passos, pct_start=0.05)
    d32 = lambda z: Fn.interpolate(z, size=(32, 32), mode="area")
    t0 = time.time(); h, h0 = [], []
    for it in range(passos):
        x, y = rem.lote(lote)
        o = net(x)
        perda = (d32(o) - d32(y)).abs().mean()
        opt.zero_grad(); perda.backward(); opt.step(); sched.step()
        h.append(perda.item()); h0.append((d32(x) - d32(y)).abs().mean().item())
        if it % 500 == 0 or it == passos - 1:
            print(f"  passo {it:5d}  erro grosso {np.mean(h[-200:]):.4f}  (sem rede, média: {np.mean(h0[-200:]):.4f})  {time.time() - t0:.0f} s", flush=True)
    torch.save(net.state_dict(), os.path.join(RES, "harmonizador2.pt"))
    return net


if __name__ == "__main__":
    if len(sys.argv) > 1 and sys.argv[1] == "v2":
        net = treinar2(int(sys.argv[2]) if len(sys.argv) > 2 else 3000)
        avaliar(net, sufixo="_v2")
    else:
        net = treinar(int(sys.argv[1]) if len(sys.argv) > 1 else 3000)
        avaliar(net)
