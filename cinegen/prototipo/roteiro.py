"""Programa de cena (H1). Em produção, um LLM local escreve isto a partir de uma ideia,
consultando a memória de série (H8). Aqui, um curta de exemplo escrito à mão."""

ROTEIRO = {
    "titulo": "A Porta",
    "locacao": {"nome": "cozinha", "semente": 7},   # mesma semente = mesma locação em toda a série
    "elenco": ["LIA", "THEO"],
    "eventos": [
        {"tipo": "abertura", "dur": 2.2},
        {"tipo": "batida", "n": 2, "forca": 0.45},
        {"tipo": "fala", "quem": "LIA", "texto": "Você ouviu isso?"},
        {"tipo": "fala", "quem": "THEO", "texto": "Foi só o vento."},
        {"tipo": "pausa", "dur": 0.9},
        {"tipo": "batida", "n": 3, "forca": 1.0},
        {"tipo": "fala", "quem": "LIA", "texto": "O vento não bate na porta."},
        {"tipo": "final", "dur": 3.0},
    ],
}
