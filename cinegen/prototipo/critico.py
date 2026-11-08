"""H10 — Críticos pequenos. Aqui: crítico de sincronia labial (áudio x boca)."""
import numpy as np


def sincronia(audio, sr, abertura_boca, fps, max_lag=6):
    """Correlação entre a energia da voz e a abertura da boca, testando atrasos de ±max_lag quadros."""
    n = len(abertura_boca)
    hop = sr / fps
    rms = np.array([np.sqrt(np.mean(audio[int(i * hop):int((i + 1) * hop)] ** 2) + 1e-12) for i in range(n)])
    melhor = (-2.0, 0)
    for lag in range(-max_lag, max_lag + 1):
        a = rms[max(lag, 0): n + min(lag, 0)]
        b = abertura_boca[max(-lag, 0): n + min(-lag, 0)]
        c = np.corrcoef(a, b)[0, 1]
        if c > melhor[0]:
            melhor = (c, lag)
    corr, lag = melhor
    return dict(correlacao=float(corr), atraso_ms=1000.0 * lag / fps, aprovado=bool(corr > 0.5 and abs(lag) <= 1))
