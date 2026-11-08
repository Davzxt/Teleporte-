# CINEGEN-2: arquitetura leve, rápida e de alta qualidade (baseada nos experimentos)

## O que os experimentos provaram (ver `ciencia/CADERNO.md`)

1. **Separar estrutura de textura funciona.** Um modelo minúsculo sorteia só a "planta" (forma,
   luz, pose: ~150 números) e a textura de foto vem de exemplos reais. Treina em segundos e gera
   centenas de imagens por segundo em CPU.
2. **A qualidade sobe com os dados, não com o tamanho do modelo:**
   - rostos de 128 → 256 px com FFHQ, com o detector caindo de ~88% para ~60–66%;
   - voz de 70 s → 97 min, ficando 27% mais próxima da real e com 35% menos emendas.
3. **O teto da recombinação pura é a coerência.** Juntar partes de pessoas diferentes gera olhar
   torto, bigode em rosto feminino e dentes que não combinam. Falta um componente que **aprenda a
   harmonizar** as partes.
4. **Direção por crítico funciona.** Várias tomadas mais um crítico pequeno levaram a sincronia
   labial de −167 ms para dentro de 1 quadro.
5. **Renderizar é barato.** O mundo procedural ou 3D custa milissegundos por quadro.

## A arquitetura

```
TEXTO/ROTEIRO ──► [Diretor: LLM pequeno local] ──► programa de cena (planos, falas, eventos)
                                                        │
      ┌───────────────────────────┬─────────────────────┼──────────────────────┐
      ▼                           ▼                     ▼                      ▼
 PLANTA (paramétrica,       MEMÓRIA DE EXEMPLOS    ATORES/MUNDO 3D         VOZ
 pequena, 1–50M param.)     (não-paramétrica,      (Gaussian Splatting,    seleção de unidades
 sorteia estrutura:         em DISCO: retalhos de  criados 1x e            em horas de fala real
 pose, luz, forma,          pele, cabelo, fundo,   reutilizados)           + suavizador neural
 identidade                 movimento, som)                                curto nas emendas
      │                           │                     │                      │
      └──────────► HARMONIZADOR (rede pequena, 1 passo, ~20–100M) ◄────────────┘
                   aprende a juntar planta + exemplos sem emendas
                   (coerência de identidade, gênero, olhar, luz)
                                   │
                    VÍDEO NO ESPAÇO DO CODEC (H4): quadros-chave + vetores de movimento
                                   │
                       ÓTICA/PELÍCULA FÍSICA (sem rede) ─► CRÍTICOS (sincronia, identidade,
                                                            realismo) ─► refaz só o que reprovar
```

### Por que cada peça atende "leve, rápido e com pouco treino"

| Peça | Aprendido? | Treino | Tamanho | Velocidade |
|---|---|---|---|---|
| Planta | sim, pequeno | minutos | 1–50M param. | milhares/s |
| Memória de exemplos | não (índice) | só indexar | disco (GB–TB), 0 de VRAM | busca em ms |
| **Harmonizador** | **sim (peça nova)** | **horas** (1 GPU alugada) | 20–100M | 1 passo, ms |
| Mundo/atores 3D | ajuste por cena | minutos por ator | MB por ator | 100+ qps |
| Voz | não (seleção) + suavizador pequeno | minutos | banco de fala + ~5M | tempo real |
| Críticos | sim, pequenos | minutos | 1–10M | ms |

O conhecimento visual e sonoro fica **na memória em disco**, não nos pesos. Por isso o modelo é
leve e o treino é curto. Em troca, **a qualidade final depende da qualidade da memória**: para
cinema, a memória precisa ter material de cinema (alta resolução, iluminação profissional, horas de
voz atuada).

## O que falta para "qualidade de cinema" (honesto)

| Requisito | Status aqui | O que destrava |
|---|---|---|
| Rosto HD coerente | 256 px, com remendos | **harmonizador** treinado (horas de GPU) + memória maior (FFHQ completo, 70 mil) |
| Corpo e atuação | não existe | memória de captura de movimento + vídeo de pessoas |
| Voz natural | boa, com algumas emendas | mais horas de voz atuada + suavizador neural nas emendas |
| Mundo fotorrealista | procedural "pintado" | memória de filmagens reais de paisagens (Mosaico de Película, H3) |
| Tempo de geração de um filme | horas em CPU | já atende |

## Próximo experimento recomendado

**Treinar o harmonizador.** Uma rede pequena de 1 passo recebe a fotomontagem "remendada" e
devolve o rosto coerente.

- **Pares de treino gerados de graça:** pegar um rosto real, remendar com partes de outros
  (exatamente o que o modelo atual faz) e treinar a rede para desfazer o remendo, mantendo a
  identidade. Não precisa de rótulo nenhum.
- **Custo:**
  - em CPU (este ambiente), só dá para um teste em 64–128 px;
  - em 256 px ou mais, de 2 a 6 horas numa GPU alugada (poucos dólares).
- **Como medir:** detector C2ST, novidade e taxa de defeitos visíveis em amostras cegas.
