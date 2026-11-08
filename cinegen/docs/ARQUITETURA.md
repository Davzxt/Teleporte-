# Arquitetura — do roteiro ao filme

```
 IDEIA ──► [LLM local] ──► ROTEIRO ──► DECUPAGEM (programa de cena por plano)
                │                              │
                ▼                              ▼
        MEMÓRIA DE SÉRIE (H8) ◄──────► ELENCO + LOCAÇÕES persistentes (H2, H3)
                                               │
          ┌────────────────────┬───────────────┼─────────────────────┐
          ▼                    ▼               ▼                     ▼
   ATUAÇÃO (H7)        FALA: trajetória   EVENTOS FÍSICOS      FX não-3D
   motion matching     articulatória (H5)  (passos, portas)    codec generativo (H4)
          │              │          │           │                     │
          ▼              ▼          ▼           ▼                     ▼
      RENDER 3DGS  ◄── rosto     vocoder    foley modal (H6)  composição no render
          │                         │           │
          ▼                         ▼           ▼
    ÓTICA + PELÍCULA (H9)        MIXAGEM ESPACIAL (posição em relação à câmera)
          │                         │
          └──────────► CRÍTICOS (H10) ◄─┘  ──► refaz só os planos reprovados
                              │
                              ▼
                      FILME / EPISÓDIO (.mp4)
```

## Componentes e tamanho estimado

| Componente | Tipo | Tamanho | Roda em |
|---|---|---|---|
| Roteirista/diretor | LLM local quantizado (7–14B) | 5–9 GB | GPU |
| Programa de cena | JSON/DSL validado | — | CPU |
| Atores e locações | Gaussianas 3D + rig | 50–500 MB cada (disco) | GPU (render) |
| Mosaico de Película | índice ANN + rede de harmonização | 0,5–2 TB SSD + ~100M param. | SSD + GPU |
| Codec generativo | transformer AR em vetores de movimento | ~200–400M param. | GPU |
| Atuação | motion matching + seletor pequeno | base mocap + ~20M param. | CPU |
| Voz | trajetória articulatória → vocoder neural | ~30–80M param. | GPU/CPU |
| Foley | síntese modal + biblioteca de sons | — | CPU |
| Críticos | redes pequenas especializadas | ~10–300M cada | GPU |

## Orçamento de tempo (estimativa para um filme de 100 min, 1080p, 24 qps)

Hardware de referência: **uma GPU de 24 GB (RTX 3090/4090)**, 64 GB de RAM e SSD NVMe de 2 TB.
Com 12 GB de VRAM funciona, mas mais devagar. Considera que todos os modelos já estão treinados.

| Etapa | Tempo |
|---|---|
| Roteiro + decupagem + bíblia (LLM) | 20–40 min |
| Criação do elenco e das locações (só na 1ª vez da série) | 1–2 h |
| Atuação (motion matching) e trajetórias de fala | 10–20 min |
| Render 3DGS de 144 mil quadros (100–200 qps) | 12–25 min |
| FX por codec generativo (depende do filme) | 10–40 min |
| Voz (vocoder ~50× tempo real) + foley + mixagem | 10–20 min |
| Pós óptica + críticos + re-renders (~+40%) | 30–60 min |
| **Total** | **~2,5 a 5 h** (episódios seguintes da série: ~1–2 h) |

Para comparação: 144 mil quadros por difusão, a ~5 s por quadro, dariam **~200 horas** na mesma GPU.

## Roteiro de construção (fases)

1. **Fase 0, já feita aqui:** protótipo de brinquedo em CPU (`prototipo/`) que prova o fluxo
   roteiro → cena → fala articulatória → boca sincronizada → foley por eventos → look de cinema →
   crítico.
2. **Fase 1:** trocar o renderizador 2.5D por um rasterizador 3DGS em GPU (gsplat) e criar 1 ator
   real a partir de fotos.
3. **Fase 2:** vocoder neural condicionado na trajetória articulatória (treinar com um corpus de fala
   em português e articulação estimada a partir de vídeo do rosto).
4. **Fase 3:** motion matching com uma base de mocap aberta e um LLM gerando o programa de cena.
5. **Fase 4:** Mosaico de Película (índice + rede de harmonização) e codec generativo para FX.
6. **Fase 5:** críticos treinados, um curta de 5 min e depois um episódio.
