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

Следствия:

- Любая метрика на val систематически смещена.
- `Jaccard(val_avail, test_avail) = 0.4676` — фичи, зависящие от состава
  `available` (`pop_rank_in_available`, `pop_rank_frac`, `pop_ratio_in_institute`),
  считаются в разных контекстах.
- `Jaccard(train_pool_P2, test_avail) = 0.42` — модель учится на другом пуле
  кандидатов, чем работает.

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
│   ├── content_embed.py           # мультиязычные эмбеддинги (MiniLM, 384d)
│   ├── popularity.py              # global / by-institute популярность
│   ├── cold_start.py              # fallback для cold (в v7+ отключён)
│   ├── submit.py                  # запись submission.csv + валидация
│   ├── kfold_eval.py              # 5-fold честная оценка (--content-backend)
│   ├── estimate_test_score.py     # композиционная оценка public по val
│   ├── diag_*.py                  # диагностика
│   └── run_ltr_v*.py              # эксперименты
├── submissions/                   # сгенерированные сабмиты
├── models/                        # сохранённые бустеры
├── metric.py                      # официальный чекер
└── README.md
```

## 4. Champion: v7 (public = 0.3348)

### 4.1. Признаки — 31 (`FEATURE_COLS_V4`)

**S — студент (7):**
`institute`, `specialty`, `n_periods`, `n_choices`, `n_unique_modules`,
`avg_history_priority`, `is_cold`

**M — модуль (10):**
`pop_weighted_global`, `pop_raw_global`, `pop_by_institute`,
`pop_by_specialty`, `pop_last_period`, `pop_prev_period`, `pop_first_period`,
`n_periods_available`, `title_len`, `desc_len`

**S×M — взаимодействие (14):**
`already_chosen`, `cooc_score`, `cooc_max`, `cooc_norm`,
`cooc_ppmi_score`, `cooc_ppmi_max`, `cooc_ppmi_norm`,
`tfidf_cos_mean`, `tfidf_cos_max`,
`pop_ratio_in_institute`, `pop_rank_in_available`, `pop_rank_frac`,
`n_available`, `was_available_before`

### 4.2. Обучающие срезы

| Срез | Строк | Вес | Что моделирует |
| --- | --- | --- | --- |
| `p1` | 830K | 1.0 | P1 → target P2 (из train) |
| `p0` | 652K | 0.5 | [] → target P1 (из train) |
| `pmask` | 830K | 1.0 | P1 со стёртым профилем → target P2 (cold-like) |
| `p2` | 181K | 3.0 | P1,P2 → target P3 (80% val_p2) |

### 4.3. Модель

- LightGBM `lambdarank`, `truncation_level=15`, `learning_rate=0.05`,
  `num_leaves=31`, `min_data_in_leaf=100`.
- `label_gain = [0.0, 0.35, 0.45, 0.60, 0.80, 1.00]` — веса приоритетов.
- Custom `feval = pndcg5_eval` — правильная метрика для early stopping.
- Multi-seed: 5 моделей (42..46), усреднение скоров.

### 4.4. Инференс

- Train / val: вся история.
- **Test**: `max_history_periods=2` + нормализация номеров периодов
  (2 и 3 → 1 и 2). Устраняет сдвиг `pop_last_period` / `pop_prev_period`.
- Cold-start fallback **отключён** — LTR на них работает лучше.

## 5. Наборы признаков (`FEATURE_SETS` в `ltr_model.py`)

| Набор | # | Content-сигнал | Совместим с `--content-backend` |
| --- | --- | --- | --- |
| `v4` | 31 | TF-IDF (`tfidf_cos_mean/max`) | `tfidf`, `both`, `none` |
| `v5` | 31 | Embeddings (`embed_cos_mean/max`) | `embed`, `both` |
| `v6` | 33 | Оба (`tfidf_cos_*` + `embed_cos_*`) | `both` |

Несовместимые комбинации (`--content-backend tfidf --feature-set v5` и т.п.)
падают с явной ошибкой в `kfold_eval.py` — см. `COMPATIBLE_FEATURE_SETS`.

`content_backend` в `FeatureStats.fit` управляет тем, какие content-модели
считать. `build_dataset` **всегда** пишет 4 колонки
(`tfidf_cos_mean/max`, `embed_cos_mean/max`); неактивные заполняются нулями.
Это позволяет переключать `FEATURE_SETS` без ветвлений в коде.

### Кэш эмбеддингов

`content_embed.py` кэширует эмбеддинги в `data/module_embeddings.npz`
(`shape = (322, 384)`, модель `paraphrase-multilingual-MiniLM-L12-v2`).
Первый запуск считает, дальше читает из кэша.

## 6. Оценка качества

| Метод | Что даёт | Честность |
| --- | --- | --- |
| **Holdout** (284 val_p2) | быстрый A/B | шумит ±0.0075 |
| **KFold** (`kfold_eval.py`) | 5-fold по warm-val | переоценивает `weight_valp1` |
| **Composite** (`estimate_test_score.py`) | test-взвешивание сегментов | p2 в `val_submission_*.csv` измеряется **in-sample** |

### Композиция test (для composite)

```txt
cold = 0.352   ← val cold
p1   = 0.020   ← val p1
p2   = 0.324   ← val p2 (использовать честный holdout!)
p3   = 0.304   ← val p2 (proxy: после max_history_periods=2 структурно совпадает)
```

**Важно:** `val_submission_*.csv` генерируется `run_ltr_v29.py` на **всех**
val-студентах, но p2-срез (w=3.0) обучался на 80% из них. Поэтому
композиционная оценка по этому файлу даёт **завышенный** p2 (на v29 —
0.5716 против честного holdout 0.5147). Правильная процедура:

1. Взять `holdout p2` из логов `run_ltr_v*` (шаг `[8/9]`).
2. Подставить в composite:

   ```txt
   correct = 0.352·cold + 0.020·p1 + 0.628·p2_holdout
   ```

3. Для v29 это даёт ≈ **0.430** против наивных 0.408 (скрипт) / 0.466 (со всеми p2).

### Gap val → test

Даже «правильный» composite v29 = 0.430 против public v7 = 0.3348.
**Gap ≈ 0.095 — структурный**, композиционной поправкой не убирается.
Правило: **любой прирост < ±0.003 считать шумом** до тех пор, пока не
найдена причина gap.

## 7. Эксперименты

| Версия | Идея | Holdout | Public |
| --- | --- | --- | --- |
| **v7** | **champion** | 0.5134 | **0.3348** |
| v10/v10b | p1-классификатор | 0.50 | — |
| v11 | cold-ranker | — | — |
| v12 | +3 rank-фичи | 0.5148 | 0.2838 |
| v13 | −rank, +content-pop | 0.5089 | — |
| v14 | CatBoost YetiRank | 0.4708 | — |
| v15 | −2 rank-фичи | 0.5128 | 0.2909 |
| v16 | absolute-period фичи | 0.4974 | — |
| v17 | фильтр `already_chosen` | 0.3371 | — |
| v18/v22 | v7 + фильтр | — | — |
| v23 | fix периодов + val_p1 w=1 | 0.5225 | 0.2980 |
| v24 | fix only | 0.5126 | — |
| v25 | z-ensemble v23+v24 | 0.5138 | — |
| v26 | fix + val_p1 w=5 | 0.4972 | 0.3291 |
| v27 | no fix + val_p1 w=5 | 0.5008 | 0.3224 |
| v28 | v7 −3 rank-фичи | 0.5214 | 0.3094 |
| **v29** | **v7 + эмбеддинги (`v6`, `both`)** | **0.5147** | — |

**Итог**: ни одна модификация v7 не побила её public.
Локальные улучшения — шум или переобучение на val.

### v29 — детали

- `content_backend = "both"`, `feature_set = "v6"` (33 признака).
- Диагностика `diag_content_importance.py`: `embed gain = 266 / ~28000 = 0.95%`.
  Эмбеддинги реально используются, но LightGBM опирается в основном на
  rank/pop-фичи (`pop_rank_frac = 38901`, `pop_weighted_global = 35527`).
- Holdout +0.0013 при шуме ±0.0075 → статистически = v7.
- **Не сабмитить** — это шум.

## 8. Что уже проверено и не работает

- Cold-start fallback (LTR сам лучше: 0.28 vs 0.19).
- Правка периодов через `pop_by_period[3]` (утечка через `val_solution`).
- Срез `val_p1` с большим весом (val_distribution ≠ test).
- Rank-фичи (`pop_rank_in_available` и др.) — не переносятся val↔test.
- CatBoost YetiRank (игнорирует веса).
- Фильтр `already_chosen` (метрика поощряет повторы).
- Ансамбли v23+v24 (модели коррелированы, +0.001).
- Эмбеддинги как замена TF-IDF (v29 = v7 + шум).

## 9. Правила работы

1. **Лимит сабмитов — 2 в день.** Беречь.
2. **Сабмит — только с уверенным приростом** (не шум).
3. **Один ответ = одно действие** — файл, эксперимент, диагностика.
4. **Код приводится полностью**, без сокращений.
5. **Существующие сигнатуры не менять** без явной причины.
6. **Новые файлы называть явно** — пользователь отслеживает структуру.

## 10. Быстрый старт

```bash
# активировать venv (пример под Windows)
.venv\Scripts\activate

# champion v7 → submissions/submission.csv
python -m src.run_ltr_v7

# v29 (v7 + эмбеддинги) → submissions/submission_v29.csv
python -m src.run_ltr_v29

# 5-fold честная оценка: v4 + tfidf (baseline)
python -m src.kfold_eval --content-backend tfidf --feature-set v4

# 5-fold: v6 + both (эмбеддинги подключены)
python -m src.kfold_eval --content-backend both --feature-set v6

# композиционная оценка public для готового val-сабмита
python -m src.estimate_test_score submissions\val_submission_ltr_v29.csv

# диагностика: используются ли embed-фичи
python -m src.diag_content_importance
```

## 11. Текущий статус

- **Champion**: v7, public **0.3348** (15-е место).
- **Цель**: public ≥ 0.43 для топ-1.
- **В работе**: анализ структурного gap val→test (~0.095);
  аудит `available_modules` val vs test (`diag_val_vs_test_available`);
  оценка вклада cold-сегмента (35.2% test).
- **Ожидания**: тюнинг признаков исчерпан (v12–v29 — все ниже v7);
  дальнейший рост — только через (а) честную валидацию без val-утечек,
  (б) прицельный cold-сегмент, (в) переносимые content-сигналы,
  не зависящие от состава `available`.

## 12. Известные проблемы (post-mortem)

1. **val ≠ test по составу.** Cold 2.9% vs 35.2%, warm P1 50% vs 2%.
   Любая оценка на val смещена.
2. **Jaccard(val_avail, test_avail) = 0.4676.** Rank-фичи считаются в
   разных контекстах.
3. **Jaccard(train_pool_P2, test_avail) = 0.42.** Модель учится на другом
   пуле кандидатов, чем работает.
4. **Holdout шумит ±0.0075** (n=284). Все приросты +0.001..+0.010 —
   статистически незначимы.
5. **KFold переоценивает `weight_valp1`.** Val_p1 в val = 50%, в test = 2%.
6. **Утечка через `val_solution`.** `pop_by_period[3]` и `p2`-срез (w=3.0)
   построены на val-таргетах. `estimate_test_score` тоже завышает p2,
   если читает in-sample сабмит.
