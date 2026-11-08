"""Métricas sem redes pré-treinadas.

c2st: um detector CNN é treinado para separar reais (teste) de gerados. 50% = indistinguível.
espectro: distância entre espectros de potência radiais (log). 0 = mesma textura/nitidez.
novidade: distância ao vizinho mais próximo do treino / mesma medida para fotos reais nunca vistas.
          ~1 = inventa tanto quanto uma pessoa nova real; << 1 = está copiando o treino.
"""
import os
import numpy as np
import torch
import torch.nn as nn


def _cnn():
    # 1ª camada sem passo: enxerga textura fina (borrão, ruído, emendas)
    bloco = lambda i, o, s: [nn.Conv2d(i, o, 3, s, 1), nn.BatchNorm2d(o), nn.LeakyReLU(0.2)]
    return nn.Sequential(*bloco(3, 32, 1), *bloco(32, 64, 2), *bloco(64, 128, 2), *bloco(128, 128, 2),
                         *bloco(128, 128, 2), nn.AdaptiveAvgPool2d(1), nn.Flatten(), nn.Linear(128, 1))


def c2st(reais, gerados, epocas=10, seed=0):
    torch.manual_seed(seed)
    n = min(len(reais), len(gerados))
    r = torch.tensor(reais[:n]).permute(0, 3, 1, 2)
    g = torch.tensor(np.clip(gerados[:n], 0, 1)).permute(0, 3, 1, 2).float()
    X = torch.cat([r, g]); y = torch.cat([torch.ones(n), torch.zeros(n)])
    perm = torch.randperm(2 * n); X, y = X[perm], y[perm]
    corte = int(1.6 * n)
    net = _cnn(); opt = torch.optim.Adam(net.parameters(), 1e-3)
    for _ in range(epocas):
        for i in range(0, corte, 128):
            l = nn.functional.binary_cross_entropy_with_logits(net(X[i:i + 128]).squeeze(1), y[i:i + 128])
            opt.zero_grad(); l.backward(); opt.step()
    net.eval()
    with torch.no_grad():
        p = net(X[corte:]).squeeze(1) > 0
    return float((p == (y[corte:] > 0.5)).float().mean())


def _espectro(X):
    F = np.abs(np.fft.fftshift(np.fft.fft2(X.mean(3)), axes=(1, 2))) ** 2
    n = X.shape[1]; c = n // 2
    yy, xx = np.mgrid[:n, :n]
    r = np.sqrt((yy - c) ** 2 + (xx - c) ** 2).astype(int)
    m = F.mean(0)
    return np.log(np.bincount(r.ravel(), m.ravel())[1:c] / np.bincount(r.ravel())[1:c])


def espectro(reais, gerados):
    return float(np.abs(_espectro(reais) - _espectro(np.clip(gerados, 0, 1))).mean())


def _nn(A, B):
    A = torch.tensor(A.reshape(len(A), -1)); B = torch.tensor(B.reshape(len(B), -1))
    return torch.cdist(A, B).min(1).values.numpy()


def novidade(treino, teste, gerados, n=500):
    d = lambda X: X[:, ::2, ::2].astype(np.float32)
    tr = d(treino)
    return float(np.median(_nn(d(np.clip(gerados[:n], 0, 1)), tr)) / np.median(_nn(d(teste[:n]), tr)))


def avaliar(nome, treino, teste, gerados, t_treino, t_gerar_por_img):
    return dict(metodo=nome, fd_utk=fd_utk(teste, gerados), c2st=c2st(teste, gerados), espectro=espectro(teste[:1000], gerados[:1000]),
                novidade=novidade(treino, teste, gerados), treino_s=t_treino, gerar_ms=1000 * t_gerar_por_img)


# ------------------------------------------------------------------ FD-UTK: "FID" com rede treinada em rostos reais
_EXTRATOR = None
_CAMINHO = os.path.join(os.path.dirname(os.path.abspath(__file__)), "resultados", "extrator_utk.pt")


def _extrator_rede():
    b = lambda i, o, s: [nn.Conv2d(i, o, 3, s, 1), nn.BatchNorm2d(o), nn.ReLU()]
    corpo = nn.Sequential(*b(3, 32, 1), *b(32, 64, 2), *b(64, 64, 1), *b(64, 128, 2), *b(128, 128, 1),
                          *b(128, 256, 2), *b(256, 256, 2), nn.AdaptiveAvgPool2d(1), nn.Flatten())
    return corpo, nn.Linear(256, 1 + 2 + 5)


def extrator(treino=None, rot=None, epocas=4):
    global _EXTRATOR
    if _EXTRATOR is not None:
        return _EXTRATOR
    corpo, cab = _extrator_rede()
    if os.path.exists(_CAMINHO):
        corpo.load_state_dict(torch.load(_CAMINHO))
    else:
        torch.manual_seed(0)
        X = torch.tensor(treino).permute(0, 3, 1, 2); R = torch.tensor(rot)
        opt = torch.optim.Adam(list(corpo.parameters()) + list(cab.parameters()), 2e-3)
        for ep in range(epocas):
            perm = torch.randperm(len(X))
            for i in range(0, len(X), 128):
                j = perm[i:i + 128]
                xb = X[j]
                if np.random.rand() < 0.5:
                    xb = xb.flip(3)
                o = cab(corpo(xb))
                l = (nn.functional.l1_loss(o[:, 0], R[j, 0] / 50) + nn.functional.cross_entropy(o[:, 1:3], R[j, 1].long())
                     + nn.functional.cross_entropy(o[:, 3:], R[j, 2].long()))
                opt.zero_grad(); l.backward(); opt.step()
            print(f"  extrator época {ep}: perda {l.item():.3f}")
        torch.save(corpo.state_dict(), _CAMINHO)
    corpo.eval()
    _EXTRATOR = corpo
    return corpo


def _feats(X):
    net = extrator()
    with torch.no_grad():
        return torch.cat([net(torch.tensor(np.clip(X[i:i + 500], 0, 1)).permute(0, 3, 1, 2).float())
                          for i in range(0, len(X), 500)]).double().numpy()


def fd_utk(reais, gerados):
    a, b = _feats(reais), _feats(gerados)
    m1, m2 = a.mean(0), b.mean(0)
    c1, c2 = np.cov(a, rowvar=False), np.cov(b, rowvar=False)
    ev = np.linalg.eigvals(c1 @ c2)
    return float(((m1 - m2) ** 2).sum() + np.trace(c1) + np.trace(c2) - 2 * np.sqrt(np.clip(ev.real, 0, None)).sum())
