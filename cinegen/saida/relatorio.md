# Relatório do CINEGEN (protótipo)

Duração: 14.4 s · 344 quadros · 640x360 @ 24 qps

## Tempos (CPU, sem GPU)
- voz (síntese articulatória): 0.2 s
- render (gaussianas, CPU): 73.6 s
- pós (película): 7.6 s
- som (foley + trilha + mix): 0.5 s
- velocidade de render: 4.7 qps (renderizador numpy em CPU)

## Crítico de sincronia labial (H5 + H10)
- LIA: "Você ouviu isso?" → correlação 0.66, atraso 0 ms, APROVADO
    controle (boca atrasada 208 ms de propósito) → atraso detectado -208 ms, REPROVADO
- THEO: "Foi só o vento." → correlação 0.88, atraso 0 ms, APROVADO
    controle (boca atrasada 208 ms de propósito) → atraso detectado -208 ms, REPROVADO
- LIA: "O vento não bate na porta." → correlação 0.48, atraso 0 ms, REPROVADO
    controle (boca atrasada 208 ms de propósito) → atraso detectado -208 ms, REPROVADO

## Codec generativo (H4) — plano final com câmera em movimento (320x180)
- números por quadro em pixels: 172800
- números por quadro em vetores de movimento: 1760
- redução: 98x menos números para o modelo prever
- qualidade só com movimento (sem resíduo): PSNR 46.2 dB
