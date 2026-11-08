"""Medida de abertura da boca robusta a barba, tom de pele e dentes (rosto alinhado 128x128)."""
import numpy as np


def abertura(X):
    """Altura (em linhas) da região central da boca que destoa da pele/lábio ao redor."""
    L = X.mean(3)
    perfil = L[:, 78:108, 56:72].mean(2)                      # perfil vertical no centro da boca
    ref = np.median(np.concatenate([perfil[:, :5], perfil[:, -5:]], 1), 1, keepdims=True)
    desvio = np.abs(perfil - ref) / (ref + 0.05)
    lateral = np.abs(L[:, 78:108, 40:50].mean(2) - ref) / (ref + 0.05)   # cantos: barba/sombra afetam igual
    return ((desvio - lateral) > 0.25).sum(1).astype(np.float32)
