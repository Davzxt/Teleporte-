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

---

# Parte 3: cena completa em outro planeta ("O Monólito", `../filme/`)

**Objetivo:** uma cena de filme inteira (planos, falas, som, montagem) em outro planeta, gerada sem
difusão e sem GAN.

**Fontes de cada elemento:**
- **Mundo:** procedural e determinístico, sem dados (não existem fotos de outros planetas para
  treinar):
  - planeta em órbita com atmosfera e nascer do sol com reflexo anamórfico;
  - céu com dois sóis;
  - gigante gasoso com anéis em ordem de profundidade correta;
  - cinco camadas de montanhas com névoa atmosférica e luz de contorno dependente da inclinação;
  - poeira e monólito pulsante.
- **Humano:**
  - rosto gerado (Fotomontagem), falando com animação por deformação, dentro de um capacete com
    visor dourado, reflexo curvo físico, HUD e embaçado da respiração;
  - silhueta de astronauta com ciclo de caminhada.
- **Som:**
  - voz humana recombinada (rádio com banda telefônica e bips, ou seca dentro do capacete);
  - respiração, vento e passos granulares gerados nos eventos de passo;
  - zumbido sincronizado com o pulso visual do monólito;
  - trilha de cordas graves com crescendo.

**Defeitos encontrados e corrigidos nos testes visuais:**
- emenda de textura no planeta;
- terreno em listras retas;
- máscara de luz de contorno invertida (astronauta branco);
- brilho cobrindo o monólito;
- rosto grande demais no visor;
- reflexo cobrindo o rosto;
- HUD em cima da boca.

**Direção por crítico (H10), medida:** na primeira versão, "Eu vou até lá" ficou com atraso de −167 ms
(unidades de voz com marcação imprecisa). Com até 21 tomadas por fala, cada uma proibindo uma
gravação do banco, e escolha pelo crítico:

| Fala no capacete | v1 | v2 (tomadas + crítico), medido nos pixels |
|---|---|---|
| "O céu tem dois sóis…" | 0 ms | 0 ms (correlação 0,74) |
| "Espere. Você ouviu isso?" | 0 ms | 0 ms (correlação 0,80) |
| "Eu vou até lá." | **−125 ms** | **−42 ms** (1 quadro; correlação 0,72) |

**Custo:** 45 s de cena e 1.082 quadros em 960×540, renderizados em ~10 min numa CPU de 4 núcleos.

**Limites honestos:**
- O mundo tem estilo de pintura digital fotográfica, não fotorrealismo de filmagem.
- O rosto tem 128 px (o visor disfarça).
- A silhueta do astronauta é simples.
- A voz tem um só locutor e 70 s de banco.
- Não há corpo humano realista em plano aberto.

---

# Parte 4: upgrade de dados (busca no GitHub) e Voz HD

## 9. Dados novos encontrados

- **Rostos FFHQ 512×512** (NVIDIA, CC BY-NC-SA 4.0): 703 de `mmshaifur/FFHQ_dataset` e 4.187
  extraídos de 80 das 997 partes de um zip dividido em `TrevisanoEros/ffhq-dataset`. A extração
  leu o índice zip64 e baixou só as partes necessárias. Total: **4.890 rostos HD**.
- **Voz FalaBrasil hts-br** (UFPA, BSD-2): **97,5 min**, 1.000 frases de um locutor brasileiro, com
  marcação fonética pronta (40 fonemas).

## 10. Rostos HD (256 px; detector = média de 3 sementes, 400 de teste)

| Variante | Detector ↓ | Novidade |
|---|---|---|
| Fotomontagem HD (GMM 150) | 66% | 0,86 |
| A: doadores mais parecidos (k=8) | **60%** | 0,83 |
| B: olhos/boca do mesmo doador | 66% | 0,86 |
| C: planta 32 px | 74% (refutada) | 0,89 |
| A+B | 66% | 0,82 |

**Atenção:** com 400 fotos de teste o detector tem menos poder que no experimento de 64 px
(3.000), então a comparação com ~88% é otimista.

**Defeitos visíveis que persistem:**
- olhar incoerente;
- dentes e boca de outro doador;
- bigode em rosto feminino.

Esse é o **teto da recombinação pura**. A peça que falta é um harmonizador treinado (ver
`docs/ARQUITETURA_V2.md`).

## 11. Voz HD (`../voz_hd/`)

**Componentes:**
- Conversor texto→fonema com regras do português brasileiro: tS/dZ, nasais, semivogais,
  vocalização de "l", "r" de coda.
- Seleção de unidades em 50 mil unidades.
- Emenda com custo de entonação (F0).
- Emenda no ponto de maior correlação da onda.

**Bugs encontrados pelo próprio teste:**
1. O corte do silêncio inicial usava o fim do trecho inteiro e cortava o começo das frases. O bug
   também existia na voz antiga e foi corrigido nas duas.
2. Havia uma frase gravada duas vezes no banco (vazamento no teste); duplicatas agora são excluídas.
3. A poda aleatória de candidatos destruía as continuações naturais; agora é priorizada por contexto.

**Resultado (mesma régua para as duas, deixa-um-de-fora):**

| Voz | MCD ↓ | Emendas/s ↓ |
|---|---|---|
| Antiga (70 s) | 22,3 ± 2,2 | 5,2 |
| **Voz HD (97 min)** | **16,3 ± 1,3** | **3,4** |

Ressalva: os locutores e os bancos são diferentes, então a comparação é aproximada.

---

# Parte 5: harmonizador (2 tentativas, ambas refutadas) e "O Interrogatório"

## 12. Harmonizador v1: U-Net residual de 1 passo (0,9M parâmetros), 128 px

**Treino:** auto-supervisionado em pares (rosto real → versão remendada com partes de 8 vizinhos),
perda L1 + L1 de alta frequência, 2.500 passos (~22 min de CPU).

| | Detector ↓ | Espectro ↓ | Novidade |
|---|---|---|---|
| Fotomontagem | 62% | 0,32 | 0,84 |
| v1 harmonizada | 64% | **1,00 (borrada)** | 0,85 |

**Refutada.** Uma rede determinística treinada por erro médio não consegue inventar o detalhe certo
(ele é imprevisível a partir de partes de outras pessoas). Ela converge para a média, e a média é
borrão. O risco tinha sido previsto antes do resultado.

## 13. Harmonizador v2: corrige só a escala grossa (32 px), preservando a textura real

- **No treino funciona:** o erro grosso cai de 0,0285 para 0,0213 (−25%).
- **Nos rostos gerados falha:** o detector passa de 67% para **86%**, o espectro de 0,30 para 0,42 e
  aparecem manchas escuras. Causa: **mudança de distribuição**. O treino usa plantas reais; a
  geração usa plantas sorteadas pela GMM, que têm "fantasmas" nunca vistos no treino.

**Lição:** o harmonizador precisa ser **gerador** (sortear detalhe coerente) e ser treinado com as
próprias saídas do modelo. Isso exige mais computação do que a CPU daqui oferece.

## 14. Cena "O Interrogatório" (`../filme/drama.py`)

**Elenco e som:**
- dois rostos gerados (Fotomontagem HD, doadores k=8, olhos e nariz/boca coerentes), escolhidos à
  mão pelo diretor entre 48 candidatos;
- duas vozes humanas recombinadas (FalaBrasil hts-br e TTS-Portuguese-Corpus), cada fala com 12
  tomadas escolhidas pelo crítico de sincronia;
- animação por deformação na geometria FFHQ;
- campo/contracampo com claro-escuro;
- chuva, relógio, lâmpada, trovão sincronizado com relâmpago, trilha de tensão.

**Sincronia medida nos pixels:** 6/6 falas dentro de 1 quadro (0–42 ms), correlação de 0,75 a 0,90.

**Fotografia:** a primeira versão foi reprovada pelo próprio autor (queixo cortado, vazio no quadro,
borda da foto visível, rosto estourado) e foi recomposta.

**Limite visível:** os remendos nos rostos (boca e pele).
