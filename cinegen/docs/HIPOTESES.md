# Hipóteses — CINEGEN: cinema gerado sem difusão

## A ideia central: recombinar, não alucinar

Os geradores de vídeo atuais (difusão) **inventam cada pixel de cada quadro**, partindo de ruído e
limpando aos poucos, em dezenas de passos. Isso custa caro e causa os defeitos clássicos: o rosto do
personagem muda entre planos, a mão ganha um dedo e a física "derrete". Também deixa impossível
manter 2 horas de filme coerentes.

A proposta do CINEGEN é outra:

> **Um filme não é uma sequência de imagens. É um mundo, atores, uma câmera e um som.**
> Gere o mundo **uma vez**. Depois **renderize**, que é barato, determinístico e consistente.
> Onde precisar de "humanidade" (textura, movimento, voz), **recombine dados humanos reais**
> em vez de alucinar do zero.

Daí saem 10 hipóteses. Cada uma diz o que resolve, por que deve ser mais rápida e leve, e como testá-la.

---

## H1 — CENA-PROGRAMA: o filme é um código, os pixels saem de um renderizador

**Hipótese.** Um modelo de linguagem (LLM local) escreve cada plano como um **programa de cena**:
quem está lá, onde está a câmera, lente, luz, falas, ações e eventos físicos. Os pixels não são
"gerados". Eles são **renderizados** a partir de primitivas 3D (Gaussian Splatting 3D, que roda a
100–300 quadros por segundo em uma GPU de consumo).

**Por que é rápido.** Gerar um quadro por difusão leva segundos. Rasterizar gaussianas leva
milissegundos. O custo "gerativo" acontece **uma vez por ator ou locação**, e não uma vez por quadro.

**Por que é consistente.** O ator do episódio 1 e o do episódio 8 são **o mesmo objeto**. A
continuidade é garantida pela forma como o sistema é construído, sem depender da sorte.

**Teste.** Medir quadros por segundo e a identidade facial entre planos (similaridade de embedding
facial). A meta é similaridade acima de 0,9 entre planos, contra cerca de 0,6–0,7 nos modelos de
difusão longos.

*Protótipo:* `prototipo/cena.py` (renderizador de gaussianas 2.5D em numpy) e `roteiro.py`.

---

## H2 — ELENCO PERSISTENTE: atores neurais como "arquivos de ator"

**Hipótese.** Cada personagem é um **arquivo de ator**: gaussianas 3D com rig de corpo e de rosto,
blendshapes e materiais (pele com subsurface, cabelo em fios). O arquivo é criado a partir de fotos
de referência (de pessoas reais que consentiram) ou do gerador da H3, e é reutilizado na série
inteira. Figurinos são camadas trocáveis.

**Ganho.** Um elenco de 10 pessoas custa minutos a poucas horas **uma única vez**. Uma temporada
inteira reusa tudo.

**Teste.** Criar o ator a partir de 20–50 fotos e medir a fidelidade (PSNR e LPIPS) em poses que não
apareciam nas fotos.

---

## H3 — MOSAICO DE PELÍCULA: imagem por recuperação + harmonização em 1 passo

**Hipótese.** Para criar texturas, cenários e "a cara de cinema", não se gera do ruído. Monta-se uma
**biblioteca vetorial de retalhos de filmagens reais licenciadas** (bilhões de recortes de 32×32 até
256×256, indexados por CLIP/DINO e por estatística de luz). Um gerador **pequeno, de passagem única**
(sem iterar):

1. recebe um layout grosseiro (do programa de cena da H1);
2. busca os retalhos reais mais compatíveis (busca vetorial ANN: milissegundos);
3. costura e **harmoniza** luz, cor e grão com uma rede pequena (tipo U-Net de 1 passo, ~50–100M
   parâmetros).

**Por que parece humano.** O grão, a textura de pele, os tecidos e o desfoque óptico **vêm de
película real**. A rede só corrige as emendas, e o "plástico" típico da IA some porque a textura não
foi inventada.

**Por que é leve.** O conhecimento visual fica no **índice** (disco), e não nos pesos (VRAM). Um
índice de 1 TB em SSD com uma rede de 100M parâmetros cabe em PC doméstico.

**Teste.** FID e preferência humana contra difusão com o mesmo tempo de GPU.

---

## H4 — CODEC GENERATIVO: gerar no espaço de um codec de vídeo, não no de pixels

**Hipótese.** O que não cabe bem em 3D (água, fogo, fumaça, multidões, folhagem ao vento) é gerado
**na mesma estrutura de um H.264/AV1**: um quadro-chave mais **vetores de movimento** e **resíduos
esparsos** para os quadros seguintes. Um modelo autorregressivo pequeno prevê só esses números.

**Ganho.** Os vetores de movimento são de **~50× a 200× menos números** do que os pixels. Prever
menos números significa um modelo menor e mais rápido, e o movimento fica temporalmente estável
porque é movimento de verdade, e não ruído limpo quadro a quadro.

**Teste.** O protótipo `prototipo/codec_generativo.py` mede quantos números o modelo precisaria
prever e quanta qualidade só o movimento já recupera.

---

## H5 — ARTICULAÇÃO ÚNICA: a mesma trajetória move a boca e gera a voz

**Hipótese.** A sincronia labial dos sistemas atuais é feita **depois** (gera a voz e tenta acertar a
boca nela). Aqui, existe **um único sinal de controle**: a trajetória articulatória (abertura da
mandíbula, arredondamento dos lábios, posição da língua, fechamento labial, vozeamento e nasalidade).
Ela alimenta **ao mesmo tempo**:

- o **sintetizador de trato vocal** (um vocoder neural condicionado nessa trajetória, que dá o
  realismo);
- o **rig facial** do ator.

**Resultado.** A sincronia é **estrutural**: não existe como a boca e o som se desalinharem, porque
os dois vêm do mesmo número. Dá para dublar em outro idioma gerando uma nova trajetória, e a boca
acompanha automaticamente.

**Teste.** O crítico de sincronia (`prototipo/critico.py`) mede a correlação e o atraso entre a
energia do áudio e a abertura da boca. Meta: atraso de 0 ms.

*Protótipo:* `prototipo/voz.py`, com texto em português → fonemas → articulação → síntese
fonte-filtro → boca do personagem.

---

## H6 — FOLEY POR FÍSICA: o som nasce dos eventos da cena

**Hipótese.** O programa de cena **sabe** quando uma porta é batida, um pé toca o chão ou um copo cai.
Cada evento dispara uma **síntese modal** (as frequências de ressonância do material: madeira, vidro,
metal) misturada com recuperação de gravações reais da biblioteca. A posição do objeto em relação à
câmera define a **espacialização** (pan e reverberação; em produção, ambisonics/HRTF).

**Ganho.** Sincronia perfeita sem ajuste manual. Custo quase zero em CPU.

*Protótipo:* `prototipo/som.py`. A batida na porta faz a porta tremer na imagem **e** gera o som, a
partir do mesmo evento.

---

## H7 — ATUAÇÃO POR RECOMBINAÇÃO: corpo humano real, não movimento inventado

**Hipótese.** O movimento do corpo vem de **motion matching** (a técnica de jogos AAA) sobre uma
grande base de captura de movimento de atores reais, escolhido e misturado por um modelo pequeno que
lê a intenção do roteiro ("hesita, olha para a porta, recua"). As micro-expressões vêm da mesma
lógica, com clipes faciais reais combinados com a trajetória da H5.

**Por que parece humano.** Cada gesto **foi feito por um humano**. O modelo só escolhe e emenda, e
ninguém precisa aprender a "inventar" um andar natural.

**Por que é rápido.** Motion matching roda em tempo real em CPU.

---

## H8 — MEMÓRIA DE SÉRIE: a "bíblia viva"

**Hipótese.** Um banco de estado da história (personagens, figurinos por cena, ferimentos, objetos,
horário e clima, quem sabe o quê) é lido e escrito pelo LLM a cada cena. O LLM não precisa "lembrar"
de 10 episódios. Ele **consulta**.

**Ganho.** Coerência em séries longas: o arranhão do episódio 2 continua no 3, a xícara não troca de
mão entre planos.

---

## H9 — ÓTICA FÍSICA: o "look de cinema" é física, não aprendizado

**Hipótese.** Boa parte do que o olho chama de "cinema" é determinística: lente (profundidade de
campo, bokeh, aberração, vinheta), película (curva de tons, halation, grão), proporção 2.39:1, 24 qps
e obturador de 180°. Isso pode ser **simulado fisicamente** a custo baixo, sem nenhuma rede neural.

*Protótipo:* `prototipo/pos.py` (curva fílmica, halation, split-toning, vinheta, grão, letterbox).

---

## H10 — DIREÇÃO POR CRÍTICO: qualidade por seleção, não por força bruta

**Hipótese.** Como renderizar é barato (H1), dá para gerar **várias versões de cada plano** e deixar
**críticos pequenos e especializados** escolherem: sincronia labial, continuidade, identidade facial,
exposição, ritmo de montagem e, no fim, um crítico estético treinado com preferências humanas. Só os
planos reprovados são refeitos.

**Ganho.** A qualidade sobe com amostragem barata, e não com um modelo gigante.

*Protótipo:* `prototipo/critico.py` (crítico de sincronia).

---

## Comparação com difusão (expectativa, não medição)

| Critério | Difusão de vídeo | CINEGEN (hipótese) |
|---|---|---|
| Custo por quadro | segundos (dezenas de passos) | milissegundos (renderização) |
| Consistência de personagem | frágil entre planos | garantida pela construção (mesmo objeto) |
| Duração máxima coerente | segundos a ~1 min | ilimitada (filme, série) |
| Sincronia labial | ajuste posterior | estrutural (H5) |
| Som | modelo separado | nasce dos eventos (H6) |
| Controle do diretor | prompt | câmera, lente, luz, atuação editáveis |
| VRAM | 24–80 GB | ~8–24 GB (o conhecimento fica no índice em disco) |
| Ponto fraco | física, mãos, duração | criar atores e cenários fotorreais; fenômenos não-3D (ver H4) |

## O que é original e o que já existe

Partes que **já existem** na pesquisa e servem como blocos: Gaussian Splatting 3D, avatares de
gaussianas, motion matching, síntese articulatória/fonte-filtro, síntese modal e geração
autorregressiva por escalas. O que é **proposto aqui como novo**:

- a **arquitetura inteira** "filme = programa de cena + elenco persistente + renderização",
  orientada a obras longas;
- o **Mosaico de Película** (H3): recuperação de retalhos reais com harmonização de 1 passo como
  substituto da difusão para textura cinematográfica;
- o **Codec Generativo** (H4): gerar diretamente vetores de movimento e resíduos de codec;
- a **Articulação Única** (H5): uma trajetória que governa voz e rosto, com a sincronia estrutural;
- **Foley por eventos do programa de cena** (H6) junto com **direção por críticos** (H10).

## Riscos e o que não está resolvido

- **Fotorrealismo de humanos em close** é o ponto mais difícil. Pele, olhos e cabelo exigem atores
  digitais de alta qualidade (H2), e a fidelidade em closes ainda é um problema aberto.
- **Voz realista**: o protótipo usa síntese de formantes (soa robótica). O realismo depende de
  **treinar** um vocoder neural condicionado em trajetórias articulatórias, o que exige dados de fala
  com medições articulatórias ou estimadas.
- **Treinar** os componentes (H3, H4, vocoder, crítico estético) exige datasets grandes e licenciados
  e dias ou semanas de GPU. As estimativas de "algumas horas" valem para **gerar** depois que tudo
  estiver treinado.
- **Ética e direito**: usar apenas rostos e vozes com consentimento e filmagens licenciadas.
