"""20 frases reais do TTS-Portuguese-Corpus (Edresson Casanova, CC BY 4.0), locutor masculino, PT-BR."""
import os
import wave
import numpy as np

PASTA = os.environ.get("VOZ_DIR", "/tmp/claude-0/-home-user-Teleporte-/0ffc3a91-1c16-542a-a221-0372f80381d3/scratchpad/dados/voz_real")
SR = 24000

FRASES = [
    "A inauguração da vila é quarta ou quinta-feira", "Vote se você tiver o título de eleitor",
    "Hoje é fundamental encontrar a razão da existência humana", "A temperatura é mais amena à noite",
    "Em muitas cidades a população está diminuindo.", "Nunca se deve ficar em cima do morro",
    "Para as pessoas estranhas o panorama é desolador", "É bom te ver colhendo flores menino",
    "Eu finjo me banhar num lago ao amanhecer", "Sua sensibilidade mostrará o caminho",
    "A Amazônia é a reserva ecológica do globo", "O ministério mudou demais com a eleição",
    "Novas metas surgem na informática", "O capital de uma empresa depende de sua produção",
    "Se não fosse ela tudo teria sido melhor", "A principal personagem no filme é uma gueixa",
    "Espere seu amigo em casa", "A juventude tinha que revolucionar a escola",
    "A cantora terá quatro meses para ensaiar seu canto", "Esse tema foi falado no congresso.",
]


def reamostrar(x, sr_de, sr_para):
    n = int(round(len(x) * sr_para / sr_de))
    X = np.fft.rfft(x)
    m = n // 2 + 1
    Y = np.zeros(m, complex)
    k = min(m, len(X))
    Y[:k] = X[:k]
    return np.fft.irfft(Y, n) * (n / len(x))


def carregar():
    audios = []
    for i in range(1, 21):
        with wave.open(os.path.join(PASTA, f"{i}.wav")) as w:
            x = np.frombuffer(w.readframes(w.getnframes()), "<i2").astype(np.float32) / 32768
            sr = w.getframerate()
        audios.append(reamostrar(x, sr, SR).astype(np.float32))
    return audios
