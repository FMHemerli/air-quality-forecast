# Estudo futuro: limiar de 24 h num sistema de previsão horária

Status: questão levantada, não decidida. Este documento registra o problema e as
direções candidatas, não uma solução escolhida.

## Problema identificado

A detecção de excedência implementada hoje compara **valores horários** contra
`HEALTH_THRESHOLD_UGM3 = 15.0`, tratando o número como se fosse um limiar horário.

Ele não é. A diretriz da OMS de 15 µg/m³ para PM2.5 é definida sobre a **média de
24 horas**. Aplicar esse valor diretamente a leituras horárias equipara duas grandezas
diferentes: uma concentração instantânea e uma média agregada.

## Por que isso é um problema

**1. Viés sistemático por hora do dia.** O PM2.5 tem ciclo diurno pronunciado, dirigido
pela altura da camada limite atmosférica: à noite e de madrugada a camada comprime e as
concentrações sobem; à tarde a mistura vertical dilui. Um limiar fixo aplicado a todas as
horas superdetecta nas horas naturalmente sujas e subdetecta nas horas naturalmente limpas
— mesmo em dias cuja média de 24 h é idêntica.

**2. O alvo não corresponde ao que a diretriz regula.** O evento de interesse sanitário é
"a média do dia estourou 15". O rótulo atual é "esta hora estourou 15". São eventos
distintos, com taxas-base distintas, e o modelo está sendo otimizado para o segundo
enquanto o primeiro é o que importa.

**3. A taxa-base fica inflada.** Aplicado hora a hora na Califórnia, o limiar de 15 é
ultrapassado com muita frequência. Isso torna a "excedência" um evento comum em vez de
raro, o que infla o F-beta por construção e reduz o valor informativo da métrica de
detecção — o número parece bom sem que o modelo tenha resolvido o problema difícil.

## Direções candidatas

### A. Perfil diurno multiplicativo (limiar tabelado por hora)

Concentrações de poluentes são aproximadamente lognormais, o que favorece um modelo
multiplicativo:

$$x_t = m_d \cdot r_{h(t)} \cdot \eta_t$$

com $m_d$ = média do dia, $r_h$ = perfil diurno climatológico (razão entre a hora $h$ e a
média diária, $\overline{r} = 1$) e $\eta$ = ruído multiplicativo de mediana 1. Segue

$$\tau_h = 15 \cdot r_h$$

isto é, um limiar horário **estático tabelado por hora do dia** — 24 constantes por site,
estimadas uma vez offline. Corrige o viés (1) de forma direta e fisicamente interpretável:
nas horas de camada limite comprimida $r_h > 1$ e o limiar sobe; nas horas de mistura
$r_h < 1$ e desce.

Para alinhar com a assimetria de custo já adotada ($\beta = 2$, recall pesa mais), o
limiar pode ser deslocado por um quantil do ruído:

$$\tau_h = 15 \cdot r_h \cdot q_\alpha(\eta), \qquad \alpha < 0{,}5$$

Um único parâmetro interpretável, controlando o trade-off alarme falso vs. excedência
perdida.

### B. Probabilidade condicional calibrada

Alternativa mais expressiva: estimar diretamente

$$p_t = P\left(m_d > 15 \mid x_t,\; h,\; \text{site},\; \text{histórico recente}\right)$$

por regressão logística ou pelo próprio XGBoost, seguida de **calibração** (Platt ou
isotônica), validada com reliability diagram e Brier score. O ponto de corte operacional
sai da assimetria de custo, não de uma escolha arbitrária.

Vantagem sobre (A): usa o histórico recente além da hora do dia. Custo: precisa de
calibração explícita para que a probabilidade seja utilizável como número.

Nota: as features de janela já existentes (`roll_24h_mean`, `roll_24h_max`, lags)
fornecem o histórico necessário — não é preciso novo estado na inferência.

### C. Caracterizar a agregação (pré-requisito das duas anteriores)

Pela forte autocorrelação horária do PM2.5, a média de 24 h **não** equivale a 24
amostras independentes. O tamanho amostral efetivo

$$n_{\text{ef}} = \frac{24}{1 + 2\sum_{k=1}^{23}\left(1 - \frac{k}{24}\right)\rho_k}$$

quantifica quanta suavização de fato ocorre. Quanto menor $n_{\text{ef}}$, mais uma hora
isolada informa sobre a média do dia — e mais viável é qualquer limiar horário. Medir
isso é barato e condiciona a expectativa de sucesso das direções A e B.

## Faz sentido? Como validar

O teste decisivo é avaliar **no nível diário**, que é o nível que a diretriz regula:
o esquema recupera os dias cuja média real excedeu 15?

- Comparar limiar fixo (atual) vs. $\tau_h$ por hora vs. probabilidade calibrada, com
  recall/precision/F-beta calculados sobre **dias**, no split de teste.
- Estabelecer o teto de desempenho: um detector que enxerga o dia inteiro. A distância
  até ele é a skill irrecuperável — uma hora isolada não determina uma média de 24 h, e
  esse limite deve ser reportado, não escondido.
- O harness de `scripts/ablate.py` serve para essa comparação com pouca adaptação.

## Questões em aberto

- **Qual janela de 24 h?** Dia-calendário (como a diretriz normalmente é aplicada) ou
  média móvel de 24 h? A escolha muda o rótulo e a taxa-base.
- **$r_h$ precisa ser por estação do ano?** Provável no Vale Central, por causa da
  inversão térmica de inverno.
- **$r_h$ precisa ser por site?** Quase certo — Fresno (vale, alta linha de base) e
  Monterey (costeiro, limpo) têm regimes opostos.
- **Vale manter também o limiar horário puro** como métrica secundária, para não perder
  comparabilidade com os resultados já registrados em `models/metrics.json`?
- **O limiar de 35 µg/m³** (fronteira USG do US AQI) faria mais sentido como limiar
  horário, por ser definido sobre janelas mais curtas? Vale medir as duas opções.

## Impacto se implementado

Muda a **definição do alvo de detecção**, portanto muda `metrics.py`, a métrica objetivo
do Optuna e os números já publicados. Exige retreino e atualização do README. Não afeta
o pipeline de features nem a garantia de causalidade.
