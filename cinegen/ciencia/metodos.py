"""Geradores testados. Nenhum usa difusão nem GAN."""
import numpy as np
import torch
import torch.nn.functional as Fn


def _t(X):
    return torch.tensor(X).permute(0, 3, 1, 2).float()


def _np(T):
    return T.permute(0, 2, 3, 1).numpy()


def reduzir(X, res):
    return _np(Fn.interpolate(_t(X), size=(res, res), mode="area"))


def ampliar(X, res):
    return _np(Fn.interpolate(_t(X), size=(res, res), mode="bicubic", align_corners=False))


# ------------------------------------------------------------------ B0: PCA gaussiano
class PCAGauss:
    def __init__(self, k=150, res=64):
        self.k, self.res = k, res

    def treinar(self, X):
        Xr = reduzir(X, self.res) if X.shape[1] != self.res else X
        A = torch.tensor(Xr.reshape(len(Xr), -1))
        self.mu = A.mean(0)
        U, S, V = torch.pca_lowrank(A - self.mu, q=self.k, center=False)
        self.V = V
        self.sd = (S / np.sqrt(len(A) - 1))
        return self

    def gerar(self, n, rng):
        z = torch.tensor(rng.standard_normal((n, self.k)), dtype=torch.float32) * self.sd
        X = (z @ self.V.T + self.mu).numpy()
        return X.reshape(n, self.res, self.res, 3)


# ------------------------------------------------------------------ H1: Pirâmide de Exemplos
class PiramideExemplos:
    """Base paramétrica pequena (PCA 16x16) + detalhe não-paramétrico por posição, nível a nível."""

    def __init__(self, base_res=16, niveis=(32, 64), patch=8, passo=4, ctx=4, dim=24, k_escolha=3, doadores=None,
                 base="gauss"):
        self.tipo_base = base
        self.doadores = doadores          # H1b: só pega detalhe dos K rostos reais mais parecidos
        self.base_res, self.niveis = base_res, niveis
        self.patch, self.passo, self.ctx, self.dim, self.k = patch, passo, ctx, dim, k_escolha

    def _pos(self, res):
        return [(y, x) for y in range(0, res - self.patch + 1, self.passo)
                for x in range(0, res - self.patch + 1, self.passo)]

    def _janela(self, img, y, x, res):
        c = self.ctx
        y0, y1, x0, x1 = max(y - c, 0), min(y + self.patch + c, res), max(x - c, 0), min(x + self.patch + c, res)
        return img[:, y0:y1, x0:x1].reshape(img.shape[0], -1)

    def treinar(self, X):
        cls = {"gauss": PCAGauss, "gmm": PCAGMM, "viz": PCAVizinhos}[self.tipo_base]
        self.base = cls(150, self.base_res).treinar(X)
        self.bancos = []
        ant = reduzir(X, self.base_res)
        self.globais = torch.tensor(ant.reshape(len(ant), -1))
        for res in self.niveis:
            alvo = reduzir(X, res) if res != X.shape[1] else X
            up = ampliar(ant, res)
            resid = (alvo - up).astype(np.float16)            # o detalhe que falta neste nível
            up_t = torch.tensor(up)
            descr = {}
            for (y, x) in self._pos(res):
                D = self._janela(up_t, y, x, res)
                D = D - D.mean(1, keepdim=True)                # compara estrutura, não brilho
                U, S, V = torch.pca_lowrank(D, q=self.dim, center=True)
                descr[(y, x)] = (D.mean(0), V, (D - D.mean(0)) @ V)
            self.bancos.append((res, torch.tensor(resid), descr))
            ant = alvo
        return self

    def _detalhar(self, img, res, resid, descr, rng, doad=None):
        n = img.shape[0]
        up = torch.tensor(ampliar(img, res))
        acc = torch.zeros(n, res, res, 3); peso = torch.zeros(1, res, res, 1)
        jan = torch.hann_window(self.patch + 2, periodic=False)[1:-1]
        w2 = (jan[:, None] * jan[None, :])[None, :, :, None]
        for (y, x), (mu, V, Z) in descr.items():
            q = self._janela(up, y, x, res)
            q = ((q - q.mean(1, keepdim=True)) - mu) @ V
            if doad is None:
                d = torch.cdist(q, Z)
                top = d.topk(self.k, largest=False).indices
                esc = top[torch.arange(n), torch.tensor(rng.integers(0, self.k, n))]
            else:
                d = ((Z[doad] - q[:, None]) ** 2).sum(-1)              # (n, K)
                top = d.topk(min(self.k, doad.shape[1]), largest=False).indices
                pick = top[torch.arange(n), torch.tensor(rng.integers(0, top.shape[1], n))]
                esc = doad[torch.arange(n), pick]
            acc[:, y:y + self.patch, x:x + self.patch] += resid[esc, y:y + self.patch, x:x + self.patch].float() * w2
            peso[:, y:y + self.patch, x:x + self.patch] += w2
        return (up + acc / peso.clamp(min=1e-6)).numpy()

    def gerar(self, n, rng):
        img = self.base.gerar(n, rng)
        doad = None
        if self.doadores:
            g = torch.tensor(img.reshape(n, -1))
            doad = torch.cdist(g, self.globais).topk(self.doadores, largest=False).indices
        for res, resid, descr in self.bancos:
            img = self._detalhar(img, res, resid, descr, rng, doad)
        return img


# ------------------------------------------------------------------ H2: amostradores melhores para a "planta"
class PCAGMM(PCAGauss):
    """H2a: mistura de gaussianas no espaço PCA (k-means + covariância por grupo com encolhimento)."""

    def __init__(self, k=150, res=16, grupos=48, seed=0, temp=1.0):
        super().__init__(k, res)
        self.grupos, self.seed, self.temp = grupos, seed, temp

    def treinar(self, X):
        super().treinar(X)
        Xr = reduzir(X, self.res) if X.shape[1] != self.res else X
        Z = (torch.tensor(Xr.reshape(len(Xr), -1)) - self.mu) @ self.V
        g = torch.Generator().manual_seed(self.seed)
        C = Z[torch.randperm(len(Z), generator=g)[:self.grupos]].clone()
        for _ in range(25):
            lab = torch.cdist(Z, C).argmin(1)
            for j in range(self.grupos):
                if (lab == j).any():
                    C[j] = Z[lab == j].mean(0)
        self.comp = []
        for j in range(self.grupos):
            Zj = Z[lab == j]
            cov = torch.cov(Zj.T) if len(Zj) > 2 else torch.diag(self.sd ** 2)
            cov = 0.8 * cov + 0.2 * torch.diag(torch.diag(cov)) + 1e-5 * torch.eye(self.k)
            self.comp.append((len(Zj), Zj.mean(0), torch.linalg.cholesky(cov)))
        self.pesos = np.array([c[0] for c in self.comp], float); self.pesos /= self.pesos.sum()
        return self

    def gerar(self, n, rng):
        j = rng.choice(len(self.comp), n, p=self.pesos)
        e = torch.tensor(rng.standard_normal((n, self.k)), dtype=torch.float32) * self.temp
        z = torch.stack([self.comp[a][1] + self.comp[a][2] @ e[i] for i, a in enumerate(j)])
        return (z @ self.V.T + self.mu).numpy().reshape(n, self.res, self.res, 3)


class PCAVizinhos(PCAGauss):
    """H2b: combinação convexa aleatória de um rosto real e 2 vizinhos, no espaço PCA."""

    def __init__(self, k=150, res=16, viz=10, ruido=0.15):
        super().__init__(k, res)
        self.viz, self.ruido = viz, ruido

    def treinar(self, X):
        super().treinar(X)
        Xr = reduzir(X, self.res) if X.shape[1] != self.res else X
        self.Z = (torch.tensor(Xr.reshape(len(Xr), -1)) - self.mu) @ self.V
        Zn = self.Z / self.sd
        self.knn = torch.cat([torch.cdist(Zn[i:i + 2000], Zn).topk(self.viz + 1, largest=False).indices[:, 1:]
                              for i in range(0, len(Zn), 2000)])
        return self

    def gerar(self, n, rng):
        a = torch.tensor(rng.integers(0, len(self.Z), n))
        v = self.knn[a][torch.arange(n)[:, None], torch.tensor(rng.integers(0, self.viz, (n, 2)))]
        w = torch.tensor(rng.dirichlet([1.0, 1.0, 1.0], n), dtype=torch.float32)
        z = w[:, :1] * self.Z[a] + (w[:, 1:, None] * self.Z[v]).sum(1)
        z = z + self.ruido * self.sd * torch.tensor(rng.standard_normal((n, self.k)), dtype=torch.float32)
        return (z @ self.V.T + self.mu).numpy().reshape(n, self.res, self.res, 3)


# ------------------------------------------------------------------ H3: Fotomontagem por regiões
REGIOES = {  # (centro y, centro x, raio y, raio x) no rosto alinhado 64x64
    "olho_e": (15, 16, 7, 10), "olho_d": (15, 48, 7, 10),
    "nariz": (31, 32, 9, 7), "boca": (46, 32, 7, 13),
}


class Fotomontagem:
    """Planta 16x16 sorteada (H2) + TODO o detalhe acima de 16x16 vindo de regiões inteiras de
    pessoas reais compatíveis (doadores), fundidas com máscaras suaves (mistura laplaciana de 2 bandas)."""

    def __init__(self, base="viz", doadores=24, base_res=16, suave=2.5, grupos=48, alinhar=0, temp=1.0):
        self.temp = temp
        self.tipo_base, self.k, self.base_res, self.suave = base, doadores, base_res, suave
        self.grupos, self.alinhar = grupos, alinhar      # H2c / H4

    def treinar(self, X):
        if self.tipo_base == "gmm":
            self.base = PCAGMM(150, self.base_res, grupos=self.grupos, temp=self.temp).treinar(X)
        else:
            self.base = {"gauss": PCAGauss, "viz": PCAVizinhos}[self.tipo_base](150, self.base_res).treinar(X)
        baixo = reduzir(X, self.base_res)
        self.globais = torch.tensor(baixo.reshape(len(X), -1))
        self.baixo64 = torch.tensor(ampliar(baixo, X.shape[1])).half()
        self.detalhe = torch.tensor(X).half() - self.baixo64
        yy, xx = np.mgrid[:X.shape[1], :X.shape[1]]
        ms = []
        e = X.shape[1] / 64                        # regiões definidas em 64x64
        for cy, cx, ry, rx in ((c * e for c in r) for r in REGIOES.values()):
            d = ((yy - cy) / ry) ** 2 + ((xx - cx) / rx) ** 2
            ms.append(1 / (1 + np.exp((d - 1) * self.suave * 2)))
        M = np.stack(ms)
        resto = np.clip(1 - M.sum(0), 0, 1)
        M = np.concatenate([resto[None], M])
        self.M = torch.tensor(M / M.sum(0, keepdims=True), dtype=torch.float32)[..., None]   # (R+1, H, W, 1)
        return self

    def gerar(self, n, rng):
        b = self.base.gerar(n, rng)
        g = torch.tensor(b.reshape(n, -1))
        doad = torch.cdist(g, self.globais).topk(self.k, largest=False).indices        # (n, k)
        R = self.M.shape[0]
        esc = torch.stack([torch.tensor(rng.permutation(self.k)[:R]) for _ in range(n)])  # doador distinto por região
        ids = doad[torch.arange(n)[:, None], esc]                                         # (n, R)
        B = torch.tensor(ampliar(b, self.detalhe.shape[1]))
        D = self.detalhe[ids].float()                                                     # (n, R, H, W, 3)
        if self.alinhar:                     # H4: cada região do doador desliza para encaixar na planta
            L = self.baixo64[ids].float()
            a = self.alinhar
            melhor = torch.full((n, R), float("inf")); desl = torch.zeros(n, R, 2, dtype=torch.long)
            for dy in range(-a, a + 1):
                for dx in range(-a, a + 1):
                    e = (((L.roll((dy, dx), (2, 3)) - B[:, None]) ** 2) * self.M[None]).sum((2, 3, 4))
                    m = e < melhor
                    melhor[m] = e[m]; desl[m] = torch.tensor([dy, dx])
            for dy in range(-a, a + 1):
                for dx in range(-a, a + 1):
                    m = (desl[..., 0] == dy) & (desl[..., 1] == dx)
                    if m.any():
                        D[m] = D[m].roll((dy, dx), (1, 2))
        det = (D * self.M[None]).sum(1)
        return (B + det).numpy()
