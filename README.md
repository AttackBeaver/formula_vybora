# Формула выбора — хакатон «Академия искусственного интеллекта» (МГПУ, 2026)

Задача: для каждого студента из `test.jsonl` собрать ранжированный **топ-5**
дисциплин из числа доступных ему (`available_modules`), упорядоченных по
убыванию ожидаемой заинтересованности.

## 1. Метрика

`pNDCG@5` — priority-weighted NDCG с **фиксированной нормой 1.0**:

```txt
pNDCG@5 = min(1.0, DCG@5)
DCG@5   = Σ_{i=1..5} weight(priority_i) / log2(i + 1)

weights = {1: 1.00, 2: 0.80, 3: 0.60, 4: 0.45, 5: 0.35}
```

Это **не классический NDCG**. Score обрезается сверху единицей, поэтому
идеальный ответ — модуль priority-1 на первой позиции.

Официальный чекер организаторов лежит в корне: `metric.py`.

## 2. Данные (`data/`)

| Файл | Содержимое |
| --- | --- |
| `train.jsonl` | 4774 студента, история P1–P2 (ключ `student_id`) |
| `val.jsonl` | 3043 студента, есть `available_modules` (159 модулей, ключ `id`) |
| `test.jsonl` | 4306 студентов, `available_modules` = 158 модулей (ключ `id`) |
| `modules.jsonl` | 322 модуля: `module_id`, `title`, `description`, `available_in_periods` |
| `val_solution.csv` | `id, ground_truth` = `'m_001:1 m_002:3 ...'` |

### Ключевая особенность: val и test не совпадают по составу

| Сегмент | test | val |
| --- | --- | --- |
| cold (нет истории и профиля) | **35.2%** (1516) | 2.9% (88) |
| warm P1 (1 период истории) | 2.0% (87) | 50.3% (1531) |
| warm P2 (2 периода) | 32.4% (1393) | 46.8% (1424) |
| warm P3 (3 периода) | 30.4% (1310) | — |

Дополнительно, по данным `diag_val_vs_test_available.py`:

- `available_modules` **одинаков для всех студентов внутри сплита**:
  все val-студенты видят одни и те же 159 модулей, все test-студенты — те
  же 158. Значит, `pop_rank_in_available` — не per-student фича,
  а «альтернативный глобальный рейтинг на своём пуле».
- `Jaccard(val_avail, test_avail) = 0.4676`.
- `Jaccard(train_pool_P2, test_avail) = 0.42`.
- 57 test-модулей отсутствуют в train-пуле; 88 train-модулей не встретятся
  в test.
- Популярность и co-occurrence в test **слабее** на 10–27% из-за этого.

## 3. Структура проекта

```txt
formula_vybora/
├── data/                          # данные (вне git)
├── src/
│   ├── data_loader.py             # JSONL + pickle-кэш
│   ├── metric.py                  # локальный pNDCG@5
│   ├── features.py                # FeatureStats + build_dataset (content_backend)
│   ├── ltr_model.py               # LightGBM Ranker + pndcg5_eval + FEATURE_SETS
│   ├── content.py                 # TF-IDF по title+description
│   ├── content_embed.py           # эмбеддинги (MiniLM, 384d, кэш npz)
│   ├── popularity.py              # global / by-institute популярность
│   ├── cold_start.py              # fallback для cold (global pop)
│   ├── submit.py                  # запись submission.csv + валидация
│   ├── kfold_eval.py              # 5-fold честная оценка по warm-val
│   ├── eval_clean.py              # honest eval без val в обучении
│   ├── estimate_test_score.py     # композиционная оценка public по val
│   ├── diag_*.py                  # диагностика
│   └── run_ltr_v*.py              # эксперименты
├── submissions/                   # сгенерированные сабмиты
├── models/                        # сохранённые бустеры
├── metric.py                      # официальный чекер
└── README.md
```

## 4. Champion

**Public-чемпион: v7 = 0.3348** (15-е место, подтверждён).

**Локальный чемпион (ожидает подтверждения): v37 / feature-set `v12`** —
cold raw 0.3308, warm raw 0.4760, `test_estimate (raw)` 0.4249.

### 4.1. Конфигурация v7 (базовый чемпион)

- **31 признак** `FEATURE_COLS_V4`.
- Срезы: `p0` (0.5), `p1` (1.0), `pmask` (1.0), `p2` (3.0).
- LightGBM `lambdarank`, `truncation_level=15`, `lr=0.05`,
  `num_leaves=31`, `min_data_in_leaf=100`.
- Multi-seed 5 (42..46), усреднение скоров.
- Fallback на cold через global pop.
- Test: `max_history_periods=2` + нормализация номеров периодов.

### 4.2. Конфигурация v37 (кандидат)

- **31 признак** `FEATURE_COLS_V12` = V7-набор (без `pop_rank_in_available`,
  `pop_rank_frac`, `pop_ratio_in_institute`) **+ 3 глобальные rank-фичи**:
  `pop_rank_global`, `pop_rank_global_in_avail`, `pop_rank_global_frac`.
- Срезы: `p0` (0.3), `p1` (1.0), `pmask` (1.0), **`p0_mask` (3.0)**, `p2` (3.0).
- Всё остальное как в v7.
- Инференс **без fallback** — LTR на cold теперь лучше (0.3308 > 0.2816).

## 5. Наборы признаков (`FEATURE_SETS` в `ltr_model.py`)

| Набор | # | Content | Особенность | Совместим с `--content-backend` |
| --- | --- | --- | --- | --- |
| `v4` | 31 | TF-IDF | базовый чемпион | `tfidf`, `both`, `none` |
| `v5` | 31 | embed | заменяет TF-IDF на эмбеддинги | `embed`, `both` |
| `v6` | 33 | оба | TF-IDF + embed | `both` |
| `v7` | 28 | TF-IDF | v4 без 3 rank-фич | `tfidf`, `both`, `none` |
| `v8` | 30 | оба | v6 без 3 rank-фич | `both` |
| `v11` | 34 | TF-IDF | v4 + 3 глобальные rank | `tfidf`, `both`, `none` |
| `v12` | 31 | TF-IDF | v7 + 3 глобальные rank | `tfidf`, `both`, `none` |

Несовместимые комбинации падают с явной ошибкой — см.
`COMPATIBLE_FEATURE_SETS`.

`content_backend` в `FeatureStats.fit` управляет тем, какие content-модели
считать. `build_dataset` **всегда** пишет 4 колонки
(`tfidf_cos_mean/max`, `embed_cos_mean/max`); неактивные = нули.

### Глобальные rank-фичи (v11/v12)

Добавлены в v37. Идея: `pop_rank_in_available` использует
`pop_by_institute`, который для cold-студентов вырождается (нет института →
`inst_pop = {}` → алфавитная сортировка `module_id`). Три новые фичи:

- `pop_rank_global` — ранг модуля по `pop_weighted` среди всех 322 модулей.
  Одинаков для всех студентов. Строится один раз в `FeatureStats.fit`.
- `pop_rank_global_in_avail` — ранг модуля по `pop_weighted` среди
  candidates конкретного студента (per-student).
- `pop_rank_global_frac` — нормализованный (`rank / n_available`).

`pop_rank_global` фиксируется на train и **не пересчитывается** в
`add_val_solution_popularity` — утечки нет.

### Кэш эмбеддингов

`content_embed.py` кэширует в `data/module_embeddings.npz`
(`shape=(322, 384)`, `paraphrase-multilingual-MiniLM-L12-v2`).
Первый запуск считает, дальше читает из кэша.

## 6. Оценка качества

| Метод | Что даёт | Честность |
| --- | --- | --- |
| **Holdout** (284 val_p2) | быстрый A/B в схеме v7 | шумит ±0.0075 |
| **KFold** (`kfold_eval.py`) | 5-fold по warm-val | переоценивает `weight_valp1` |
| **eval_clean** (`eval_clean.py`) | обучение **только на train**, оценка на всей val | калиброван: даёт ≈ public − 0.011 (для v4) |
| **estimate_test_score** | композиционная оценка по готовому val-сабмиту | завышает p2, если сабмит in-sample |

### Композиция test (для composite)

```txt
cold = 0.352   ← val cold
p1   = 0.020   ← val p1
p2   = 0.324   ← val p2
p3   = 0.304   ← val p2 (proxy: после max_history_periods=2 совпадает)
```

**Правило калибровки:** `eval_clean` даёт абсолют, сопоставимый с public
с точностью ±0.011 (проверено на v4: 0.3238 vs public v7 = 0.3348). Но
**как только меняется схема обучения** (добавляются val-срезы, p0_mask,
глобальные rank), этот оффсет съезжает. `holdout` и `estimate_test_score`
на in-sample сабмите остаются переоценёнными.

### Эмпирические калибровки

| Модель | eval_clean (или схема) | Public | Ratio |
| --- | --- | --- | --- |
| v4 (V4, tfidf) | 0.3238 | 0.3348 | 0.967 |
| v30 (V4 без rank, без val_p2) | 0.3181 | 0.3181 | 1.000 (случайно) |
| v29 (V6, both) | 0.3299 | 0.3218 | 1.025 |
| v33 (cold-opt ranking) | 0.4244 (in-sample) | 0.2949 | 1.44 |

**Вывод:** локальные метрики — только A/B **в одной и той же схеме
обучения**. Как только схема меняется, сравнивать числа между схемами
нельзя.

## 7. Эксперименты

### 7.1. Warm-направление (все — регресс)

| Версия | Идея | Holdout | Public | Δ vs v7 |
| --- | --- | --- | --- | --- |
| **v7** | **champion** | **0.5134** | **0.3348** | — |
| v28 | V4 −3 rank-фичи | 0.5214 | 0.3094 | −0.0254 |
| v29 | V6 (TF-IDF + embed) | 0.5147 | 0.3218 | −0.0130 |
| v30 | V4 без rank, без val_p2 | — | 0.3181 | −0.0167 |
| v34 (all) | train-пул = все 322 | 0.4707 | — | — |

### 7.2. Cold-направление

| Версия | Идея | cold raw | Public | Комментарий |
| --- | --- | --- | --- | --- |
| v7 (fallback) | global pop | 0.2256 | **0.3348** | fallback даёт 0.2816 |
| v32 | greedy по train P1 | 0.1078 | — | train P1 ≠ cold |
| v33 | ranking из val cold | 0.3424 (in-sample) | 0.2949 | overfit на 88 val cold |
| v35 | + срез `p0_mask` | 0.2524 | — | первый сдвиг |
| v36 (p0_mask=3.0) | усиленные веса | 0.2658 | — | насыщение |
| **v37 / v11** | + глобальные rank | **0.3384** | (ждём) | прорыв |
| **v37 / v12** | + глобальные rank | **0.3308** | (ждём) | warm выше |

**Главный вывод:**

- Warm-направление исчерпано. Ни одна модификация v7 не улучшила public.
- Cold-направление: единственный сигнал, который переносится — **честные
  глобальные rank-фичи** + `p0_mask` (настоящий cold в обучении).
- «Фиксированный top-5 по val cold» (v33) не переносится — val cold n=88
  слишком мал и дрейфует относительно test cold.

## 8. Что проверено и НЕ работает

- Cold-start fallback на v7 (устарело: v37 raw его обходит, 0.331 > 0.282).
- Правка периодов через `pop_by_period[3]` — утечка через `val_solution`.
- Срез `val_p1` с большим весом — val_distribution ≠ test.
- Удаление rank-3 фич (v28, v30) — **rank-3 это сигнал**, их удаление
  стоит −0.025.
- Расширение train-пула до всех 322 модулей (v34) — holdout −0.043.
- Embeddings (v5, v6, v8, v29) — вредят на public, помогают только
  на отравленной метрике.
- Фиксированный ranking по val cold targets (v33) — overfit на 88.
- CatBoost YetiRank — игнорирует веса.
- Фильтр `already_chosen` — метрика поощряет повторы.
- Ансамбли v23+v24 — модели коррелированы, +0.001.

## 9. Правила работы

1. **Лимит сабмитов — 2 в день.** Беречь.
2. **Сабмит — только с уверенным приростом** (не шум). Локальные
   метрики дают A/B **внутри одной схемы**, не абсолют.
3. **Один ответ = один файл/действие.** Код приводится полностью.
4. **Существующие сигнатуры не менять** без явной причины.
5. **Новые файлы называть явно** — пользователь отслеживает структуру.

## 10. Быстрый старт

```bash
# активировать venv (пример под Windows)
.venv\Scripts\activate

# champion v7 (public 0.3348)
python -m src.run_ltr_v7

# кандидат v37 / v12 (raw, без fallback)
python -m src.run_ltr_v37 --feature-set v12

# кандидат v37 / v11
python -m src.run_ltr_v37 --feature-set v11

# honest eval без val в обучении
python -m src.eval_clean --feature-set v4 --content-backend tfidf --n-seeds 3

# композиционная оценка public по готовому val-сабмиту
python -m src.estimate_test_score submissions/val_submission_ltr_v7.csv

# диагностика: val vs test available
python -m src.diag_val_vs_test_available

# диагностика: используются ли embed-фичи
python -m src.diag_content_importance
```

## 11. Текущий статус

- **Public-чемпион:** v7 = 0.3348 (15-е место).
- **Цель:** public ≥ 0.43 для топ-1.
- **Локальный кандидат:** v37 / v12 — `test_estimate (raw) = 0.4249`.
  Ожидание public 0.33–0.36.
- **Что работает:** глобальные rank-фичи (`pop_rank_global*`),
  `p0_mask`-срез для cold.
- **Что не работает:** всё, что трогает warm-фичи и train-пул.

### Следующие шаги

1. **Подтвердить v37/v12 на public.** Если ≥ 0.34 — новый чемпион.
2. **Если < 0.34** — диагностировать перенос `pop_rank_global*` на test;
   вероятно, попробовать ансамбль v7 + v37/v12.
3. **Параллельно** — попробовать v37 с `p0_mask=5.0` (в v36 не помогло,
   но с глобальными rank может дать ещё +0.005 на cold).

## 12. Известные проблемы (post-mortem)

1. **val ≠ test по составу.** Cold 2.9% vs 35.2%, warm P1 50% vs 2%.
2. **Jaccard(val_avail, test_avail) = 0.4676.** Rank-фичи считаются в
   разных контекстах.
3. **Jaccard(train_pool_P2, test_avail) = 0.42.** Модель учится на другом
   пуле кандидатов (v34 показал: расширять пул бесполезно).
4. **Holdout шумит ±0.0075** (n=284). Все приросты ниже ±0.003 —
   статистически незначимы.
5. **KFold переоценивает `weight_valp1`.** Val_p1 в val = 50%, в test = 2%.
6. **Утечка через `val_solution`.** `pop_by_period[3]` и `p2`-срез (w=3.0)
   построены на val-таргетах.
7. **`estimate_test_score` завышает p2 in-sample.** Если сабмит получен
   по схеме, где p2-срез обучался на 80% val_p2, скрипт считает p2 по
   всем 100%, что даёт +0.03–0.06 к реальному composite.
8. **`pop_rank_in_available` для cold вырождается** в алфавитную
   сортировку (нет института → `pop_by_institute = 0`). Компенсировано
   в v37 через `pop_rank_global*`.
