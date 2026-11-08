# Caderno de laboratório: gerador de rostos sem difusão e sem GAN, treinado em segundos

## 1. Objetivo científico

Gerar **rostos humanos novos e realistas** (pessoas que não existem):
- **sem difusão e sem GAN**;
- com **treino em minutos** em hardware doméstico (aqui: CPU de 4 núcleos, sem GPU).

**Critérios de sucesso, medidos:**

| Métrica | O que mede | Real × real (calibração) |
|---|---|---|
| **FD-UTK** ↓ | Distância de Fréchet nas características de uma rede treinada em rostos reais (o "FID" deste laboratório) | **0,06** |
| **Detector (C2ST)** ↓ | Acerto de uma CNN treinada para separar real de gerado. 50% = indistinguível | **50%** |
| **Espectro** ↓ | Diferença de textura e nitidez (espectro de potência) | 0,04 |
| **Novidade** ≈1 | Distância ao rosto de treino mais próximo, relativa a uma pessoa real nova. <1 = está copiando | 1,00 |

**Controles das métricas:** fotos reais borradas para 32 px dão FD 0,29 e detector 100%. Borradas
para 16 px dão FD 5,59. As réguas distinguem o que deveriam distinguir.

**Dados:** UTKFace, 23.705 rostos reais alinhados (20.705 de treino e 3.000 de teste nunca vistos),
em 64×64. Licença só para pesquisa não comercial. Pré-treinados (Inception/FID) ficaram
indisponíveis por causa da rede do ambiente, por isso as métricas são próprias.

## 2. Estado da arte (de memória, não verificado online nesta sessão)

- Geradores dominantes: difusão (Stable Diffusion, Seedance etc.), GANs (StyleGAN) e
  autorregressivos (VAR). Todos exigem dias a meses de GPU.
- **Prior art direto** das peças usadas aqui:
  - *face hallucination*: modelo global + retalhos (Liu, Shum & Freeman, 2007), para **super-resolução**;
  - *example-based super-resolution* (Freeman et al., 2002);
  - fotomontagem e troca de partes do rosto;
  - modelos de mistura gaussiana (GMM).
- **O que se testa aqui** é usar essas peças como **gerador do zero** e medir onde fica a fronteira
  entre qualidade e novidade.

## 3. Hipóteses e resultados (64×64, CPU, uma semente salvo indicação)

| # | Método | FD-UTK ↓ | Detector ↓ | Espectro ↓ | Novidade | Treino | Gerar/img |
|---|---|---|---|---|---|---|---|
| B0 | PCA gaussiano ("eigenfaces"), referência | 2,42 | 97% | 0,57 | 1,05 | 2 s | 0,04 ms |
| H1 | Pirâmide de Exemplos (retalhos 8×8 por posição) | 2,45 | 99% | 0,46 | 1,04 | 28 s | 8 ms |
| H1b | H1 + doadores coerentes (16) | 2,37 | 100% | 0,45 | 1,03 | 25 s | 1,0 ms |
| H2a | H1b + planta por **GMM** (48 grupos) | 1,74 | 97% | 0,43 | 0,97 | 31 s | 1,0 ms |
| H2b | H1b + planta por vizinhança convexa | 2,44 | 96% | 0,84 | **0,59** | 24 s | 0,9 ms |
| H3 | **Fotomontagem** por regiões (GMM 48) | 1,74 | 95% | 0,26 | 0,99 | 3 s | 0,4 ms |
| H4 | H3 + alinhamento local ±3 px | 2,24 | 96% | 0,27 | 0,99 | 3 s | 14 ms |
| H2c | H3, GMM 160 grupos | 1,63 | 86% | 0,23 | 0,97 | 3 s | 0,45 ms |
| H2c | H3, GMM 400 grupos | 1,52 | 91% (média 4 sementes: ~95%) | 0,24 | 0,95 | 4 s | 0,4 ms |
| **H2c** | **H3, GMM 1000 grupos** | **1,31** | 73% (média 4 sementes: **~88%**) | 0,22 | **0,93** | **5 s** | **0,4 ms** |
| H2d | H3, GMM 1000, temperatura 0,7 | 1,85 | 98% | 0,35 | 0,81 | 6 s | 0,4 ms |
| H2c | H3, GMM 2000 grupos | **1,23** | 83% | **0,21** | 0,89 | 9 s | 0,4 ms |

Imagens: `resultados/<método>.png`. Controle de cópia: `resultados/controle_copia.png`.

## 4. Análise crítica

**O que funcionou:**
- **Separar estrutura de textura.** Um modelo paramétrico pequeno sorteia só a "planta" (16×16,
  150 números) e toda a textura de foto vem de regiões inteiras de pessoas reais compatíveis (H3).
  O espectro foi de 0,57 para 0,21, quase no nível real.
- **A qualidade da planta é o gargalo.** Melhorar só a planta (Gaussiana → GMM 48 → 1000 grupos)
  levou a FD de 2,42 para 1,31 (−46%) e o detector de 97% para ~88% em média.
- **Custo:** treino de 5 segundos em CPU e 0,4 ms por rosto, ou seja, ~2.500 rostos por segundo
  em CPU.
- **Não é cópia:** no controle, cada rosto gerado é visivelmente outra pessoa em relação ao rosto
  de treino mais próximo.

**O que falhou e por quê:**
- **H1 (retalhos 8×8):** a textura vira ruído, porque retalhos vizinhos vêm de pessoas diferentes.
  Doadores coerentes ajudam pouco.
- **H2b (vizinhança):** "melhora" apenas copiando o treino (novidade 0,59). É um falso positivo que a
  métrica de novidade pegou.
- **H4 (alinhamento):** piorou. Causas prováveis: o `roll` dá a volta nas bordas e o encaixe pela
  imagem borrada é impreciso. Vale refazer com borda preenchida e encaixe por marcos faciais.
- **H2d (temperatura):** menos variedade deixa a planta "média" e borrada.

**Limite encontrado:** com mais grupos a qualidade sobe, mas a novidade cai (1000 → 0,93;
2000 → 0,89). Com 20 mil fotos, o método chega a uma **fronteira entre qualidade e cópia**. Para
passar dela é preciso mais dados ou um modelo de planta melhor que a GMM.

**Incertezas:**
- O detector varia ±10 pontos entre sementes, por isso a FD é a métrica principal.
- 64×64 está muito longe de cinema.
- O FD-UTK usa uma rede própria, então não é comparável ao FID publicado.

## 5. Conclusão honesta

Um gerador **sem difusão e sem GAN, treinado em 5 segundos em CPU**, produz rostos novos com
textura de foto e engana o detector em parte das vezes (melhor semente: 73%). **Ainda não é
indistinguível** (real = 50%), e a resolução é baixa. A hipótese central ficou apoiada:
*estrutura sorteada + textura real recombinada* resolve a textura de graça. O problema restante é
**gerar plantas coerentes**.

## 6. Próximos experimentos (em ordem de prioridade)

1. **Planta melhor que GMM, ainda treinável em minutos:** fluxo normalizante pequeno ou
   autorregressivo sobre os 150 coeficientes PCA. A meta é FD < 1,0 com novidade ≥ 0,95.
2. **Mais dados no seu PC:** FFHQ (70 mil fotos) ou CelebA (200 mil) devem empurrar a fronteira
   qualidade × cópia.
3. **Mais resolução:** níveis 128 e 256 com regiões menores (cílios, pele, cabelo).
4. **Alinhamento corrigido** (H4 sem dar a volta nas bordas, por marcos faciais).
5. **Ligar ao CINEGEN:** rosto gerado → várias vistas → ator 3D (`../ator3d/`) → fala sincronizada.

---

# Parte 2: voz humana real e vídeo de rosto falando

## 7. Voz por recombinação de fala real (`../voz_real/`)

**Dados:** 20 frases reais (70 s, locutor masculino PT-BR) do TTS-Portuguese-Corpus, de Edresson
Casanova, sob licença CC BY 4.0. Foi o único áudio de fala real acessível pela rede deste ambiente.

**Método:**
1. **Alinhamento fonético sem anotação:** HMM de 3 sub-estados por fonema, com pausas opcionais
   entre palavras e auto-treino por Viterbi a partir de início plano. Convergiu em 8 iterações
   (12 s), gerando 727 unidades.
2. **Validação física do alinhamento:**
   - "s" com centroide em 4,1 kHz e "f" em 3,3 kHz;
   - "u" em 463 Hz e "o" em 602 Hz;
   - silêncio a −64 dB contra −35 dB nas vogais.
3. **Seleção de unidades:** Viterbi sobre custo de alvo (contexto fonético) + custo de emenda
   (salto espectral). Trechos contínuos da mesma gravação custam zero.

**Resultado (deixa-um-de-fora: a frase testada nunca está no banco):**

| Voz | MCD (variante MFCC, dB) ↓ |
|---|---|
| Articulatória/formantes antiga | 57,5 ± 7,9 |
| **Recombinação de voz real** | **20,8 ± 1,9** (melhor em 20/20 frases) |

Os pesos de alvo e emenda quase não mudam o resultado: o método é robusto.

**Limites:**
- 70 s de banco geram emendas audíveis e prosodia pouco variada.
- Um só locutor.
- O "ʃ" tem só 2 exemplos.

## 8. Vídeo: rosto novo falando (`../video_real/falante.py`)

**Etapas:**
1. **Identidade:** Fotomontagem em 128 px sobre 7.177 homens de 20 a 55 anos (planta GMM com 300
   grupos). Um crítico (detector treinado na hora) ordena 64 candidatos.
   - O filtro automático de "boca neutra" **falhou** (rostos superexpostos enganam a medida), e o
     rosto final foi **escolhido por um humano** entre os 8 melhores do crítico.
   - Novidade de 1,07: não é cópia.
2. **Animação por deformação:** o queixo e o lábio inferior descem, o lábio superior sobe e o vão
   recebe interior com gradiente, dentes superiores suaves e desfoque óptico local. Inclui
   piscadas, micro-movimento e acenos ligados à energia da voz.
3. **Tentativa descartada:** escolher bocas reais por "abertura". A medida de abertura não passou
   no controle visual (barba e dentes a confundiam).

**Sincronia:**
- Trajetória × áudio: atraso de 0 ms nas 3 frases.
- Medida **nos pixels do vídeo final**: correlação de 0,62 e atraso de 0 ms.
- Uma primeira medição deu −125 ms por erro da régua (ela procurava pixels pretos depois que o
  interior da boca passou a ser marrom). Foi corrigida.

**Limites:**
- 128 px ampliados (macio); o UTKFace tem no máximo 200 px.
- Cabeça quase frontal; sem corpo nem cenário.
