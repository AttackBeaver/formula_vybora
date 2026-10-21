# Формула выбора — хакатон «Академия искусственного интеллекта» (МГПУ, 2026)

Задача: для каждого студента из `test.jsonl` собрать ранжированный **топ-5**
дисциплин из числа доступных ему (`available_modules`), упорядоченных по
убыванию ожидаемой заинтересованности.

---

## 1. Метрика

`pNDCG@5` — priority-weighted NDCG с **фиксированной нормой 1.0**:

```txt
pNDCG@5 = min(1.0, DCG@5)
DCG@5   = Σ_{i=1..5} weight(priority_i) / log2(i + 1)

weights = {1: 1.00, 2: 0.80, 3: 0.60, 4: 0.45, 5: 0.35}
```

Это **не классический NDCG**. Score обрезается сверху единицей, идеальный
ответ — priority-1 модуль на позиции 1.

Официальный чекер (`metric.py` в корне) и локальный (`src/metric.py`)
проверены на совпадение: оба дают одинаковый score на v7 val-сабмите
(0.4803 при in-sample оценке).

**ВАЖНО:** `src/metric.py` **НЕ в git** (был случайно в `.gitignore`
паттерном `metric.py`). Восстановить можно из self-test-версии в
`src/metric.py` — она содержит `parse_ground_truth`, `pndcg_at_k`,
`evaluate_pndcg5`, `hit_at_k`, `recall_at_k`. **Перед следующим клоном**
заменить в `.gitignore` `metric.py` → `/metric.py` и закоммитить `src/metric.py`.

---

## 2. Данные (`data/`)

| Файл | Содержимое |
|---|---|
| `train.jsonl` | **5002** студента, история P1–P2 (ключ `student_id`) |
| `val.jsonl`   | 3043 студента, есть `available_modules` (159 модулей, ключ `id`) |
| `test.jsonl`  | 4306 студентов, `available_modules` = 158 модулей (ключ `id`) |
| `modules.jsonl` | 322 модуля: `module_id`, `title`, `description`, `available_in_periods` |
| `val_solution.csv` | `id, ground_truth` = `'m_001:1 m_002:3 ...'` |

### Ключевая особенность: val и test не совпадают по составу

| Сегмент | test | val |
|---|---|---|
| cold (нет истории и профиля) | **35.2%** (1516) | 2.9% (88) |
| warm P1 (1 период истории) | 2.0% (87) | 50.3% (1531) |
| warm P2 (2 периода) | 32.4% (1393) | 46.8% (1424) |
| warm P3 (3 периода) | 30.4% (1310) | — |

Дополнительно, по данным `diag_val_vs_test_available.py`:

- `available_modules` **одинаков для всех студентов внутри сплита**:
  все val-студенты видят одни и те же 159 модулей, все test-студенты — 158.
- `Jaccard(val_avail, test_avail) = 0.4676`.
- 57 test-модулей отсутствуют в train-пуле; 88 train-модулей не встретятся
  в test.
- Популярность и co-occurrence в test **слабее** на 10–27% из-за этого.

---

## 3. Структура проекта

```
formula_vybora/
├── data/                          # данные (вне git)
├── src/
│   ├── data_loader.py             # JSONL + pickle-кэш
│   ├── metric.py                  # локальный pNDCG@5 (НЕ В GIT!)
│   ├── features.py                # FeatureStats + build_dataset
│   ├── ltr_model.py               # LightGBM + pndcg5_eval + FEATURE_SETS
│   ├── content.py                 # TF-IDF по title+description
│   ├── content_embed.py           # эмбеддинги MiniLM (кэш npz)
│   ├── popularity.py              # global / by-institute популярность
│   ├── cold_start.py              # fallback для cold (global pop)
│   ├── evaluate.py                # CLI-сверка с официальным metric.py
│   ├── submit.py                  # запись submission.csv + валидация
│   ├── kfold_eval.py              # 5-fold OOF
│   ├── eval_clean.py              # honest eval (без val в обучении)
│   ├── eval_clean_p0mask.py       # honest eval с p0mask
│   ├── estimate_test_score.py     # композиционная оценка public
│   ├── swap_cold_raw.py           # заменить cold-строки на raw-LTR
│   ├── diag_*.py                  # диагностика
│   └── run_ltr_v*.py              # эксперименты
├── submissions/                   # сгенерированные сабмиты (не в git)
├── models/                        # сохранённые бустеры (не в git)
├── metric.py                      # официальный чекер (НЕ В GIT!)
└── README.md
```

**Восстановление после клона:**
1. `python -m venv .venv && .venv\Scripts\activate`
2. `pip install lightgbm pandas numpy scikit-learn scipy sentence-transformers`
3. Положить `data/` (5 файлов — не в git).
4. Восстановить `src/metric.py` (см. §1).
5. `python -c "from src.data_loader import load_all; load_all()"` — построит кэш.

---

## 4. Champion: v41 raw = **0.3604** (public)

### 4.1. Конфигурация

- **34 признака** `FEATURE_COLS_V11` = V4 + 3 глобальные rank-фичи
  (`pop_rank_global`, `pop_rank_global_in_avail`, `pop_rank_global_frac`).
- Срезы: `p0` (0.5), `p1` (1.0), `pmask` (1.0), `p2` (3.0).
- LightGBM `lambdarank`, `truncation_level=15`, `lr=0.05`, `num_leaves=31`,
  `min_data_in_leaf=100`.
- Multi-seed 5 (42..46), усреднение скоров.
- Test: `max_history_periods=2` + нормализация номеров периодов.
- **Инференс: raw-LTR на cold (БЕЗ fallback).** Это ключевое отличие от v7.

### 4.2. Почему сработало

Три фактора вместе:

1. **Глобальные rank-фичи осмысленны для cold.** `pop_rank_in_available`
   использует `pop_by_institute`, который для cold (нет профиля) вырождается
   в алфавитную сортировку. Глобальные rank работают через `pop_weighted`,
   который осмыслен для всех.
2. **p2=3.0 (не 7.0, не без него).** Warm-transfer держится.
3. **Raw-LTR на cold вместо fallback.** В v41 cold raw = 0.3372 (в val),
   fallback = 0.2816. Fallback **вредит** успешной модели на −0.056.
   На public cold_raw-LTR ≈ 0.307 > fallback 0.2816.

### 4.3. Разложение public

```
0.3604 = 0.352·cold + 0.020·p1 + 0.628·warm
Если warm ≈ 0.39, p1 ≈ 0.35:
cold ≈ (0.3604 − 0.007 − 0.245) / 0.352 ≈ 0.307
```

**Впервые cold на test выше fallback** (0.2816) на +0.025.

---

## 5. Наборы признаков (`FEATURE_SETS` в `ltr_model.py`)

| Набор | # | Content | Особенность | Public (если сабмитили) |
|---|---|---|---|---|
| `v4`  | 31 | TF-IDF | базовый (v7) | 0.3240 |
| `v5`  | 31 | embed  | TF-IDF→embed | — |
| `v6`  | 33 | оба    | TF-IDF + embed | 0.3218 (v29, старая) |
| `v7`  | 28 | TF-IDF | v4 без rank-3 | 0.3094 (v28, старая) |
| `v8`  | 30 | оба    | v6 без rank-3 | — |
| **`v11`** | **34** | **TF-IDF** | **v4 + 3 глобальные rank** | **0.3604 (v41 raw)** |
| `v12` | 31 | TF-IDF | v7 + 3 глобальные rank | 0.3168 (v38) |

### Глобальные rank-фичи (v11)

- `pop_rank_global` — ранг по `pop_weighted` среди всех 322 модулей.
- `pop_rank_global_in_avail` — ранг среди candidates студента.
- `pop_rank_global_frac` — нормализованный.

`pop_rank_global` фиксируется на train и **не пересчитывается** в
`add_val_solution_popularity` — утечки нет.

**Ключевое наблюдение:** V11 (rank-3 сохранены + глобальные rank) выигрывает
у V4 честно (+0.009 в eval_clean_p0mask). V12 (rank-3 удалены + глобальные
rank) — падает на cold (0.12), потому что без rank-3 модель не может
отличить «странных» студентов от нормы.

### Кэш эмбеддингов

`content_embed.py` кэширует в `data/module_embeddings.npz`
(`shape=(322, 384)`, `paraphrase-multilingual-MiniLM-L12-v2`).

---

## 6. Оценка качества

| Метод | Что даёт | Честность на новой data |
|---|---|---|
| **Holdout** (284 val_p2) | proxy warm | **не коррелирует** с public при p2-тюнинге |
| **KFold** | 5-fold warm-val | переоценивает `weight_valp1` |
| **eval_clean** | обучение **только на train**, оценка на val | V4: 0.3203 ≈ public 0.3240 |
| **eval_clean_p0mask** | honest + p0mask срез | V11: 0.3331 — но на public v38/v39 дал регресс |
| **estimate_test_score** | композиция по val-сабмиту | завышает p2, если in-sample |

### Калибровка honest → public (новая data)

| Модель | Honest | Public | Оффсет |
|---|---|---|---|
| v4 без p0mask | 0.3203 | 0.3240 (v7) | **+0.004** |
| V12+p0mask | 0.3529 | 0.3168 (v38) | **−0.036** |
| V11+p0mask | 0.3464 | 0.3142 (v39) | **−0.032** |
| **V11 без p0mask (v41)** | — | **0.3604** | — |

**Honest eval нестабилен между семьями.** Оффсет +0.004 для V4, −0.032 для
V11/V12 без p2. **Honest предсказывает направление, но не величину.**

### Holdout → public: **не коррелирует**

| p2-вес | Holdout | Public |
|---|---|---|
| 1.5 | 0.4958 | — |
| 3.0 | 0.5069 | 0.3240 (v7) |
| 5.0 | 0.5152 | — |
| 7.0 | 0.5222 | 0.3205 (v40) |

**Не использовать holdout на p2-срезе для решения о сабмите.**

### Что реально коррелирует с public: **cold raw**

| Модель | cold raw (val) | cold на public |
|---|---|---|
| v7 (fallback) | 0.2482 | 0.2816 (fallback) |
| v40 (fallback) | 0.2468 | 0.2816 (fallback) |
| **v41 (raw)** | **0.3372** | **~0.307 (raw)** |

Если cold raw на val > fallback (0.2816) — **raw-LTR на cold даёт win**.
Если < — **fallback спасает**.

---

## 7. Эксперименты

### 7.1. Новая data (train=5002)

| Модель | Что | Public | Δ vs v7 |
|---|---|---|---|
| v7 | V4, p2=3.0, fallback | 0.3240 | — |
| v38 | V12 + p0mask, без p2 | 0.3168 | −0.0072 |
| v39 | V11 + p0mask, без p2 | 0.3142 | −0.0098 |
| v40 (p2=7.0) | V4, p2=7.0 | 0.3205 | −0.0035 |
| **v41 raw** | **V11, p2=3.0, raw cold** | **0.3604** | **+0.0364** |

### 7.2. Старая data (train=4774) — для истории

| Версия | Идея | Public |
|---|---|---|
| **v7** | **champion старой data** | **0.3348** |
| v28 | V4 −3 rank | 0.3094 |
| v29 | V6 (TF-IDF+embed) | 0.3218 |
| v30 | V4 без rank, без p2 | 0.3181 |
| v33 | cold-opt ranking | 0.2949 |

---

## 8. Что проверено и НЕ работает

- Удаление rank-3 — **реальный сигнал**, потеря −0.025.
- Embeddings (V5, V6, V8) — на public вредят.
- Расширение train-пула до 322 (v34) — holdout −0.043.
- Глобальные rank **без rank-3** (V12) — cold падает до 0.12.
- Глобальные rank **с p2 + p0mask, но без raw cold** (v38, v39) — регресс.
- Увеличение p2-веса 3.0→7.0 — holdout растёт, public падает (v40).
- Cold через фиксированный ranking (v33) — overfit на 88 cold.
- Ансамбль v7 + v39 (z-score) — хуже обоих.
- Fallback на cold **при успешной raw-модели** — вредит (−0.056 в val).

---

## 9. Текущий статус

- **Public-чемпион (новая data):** **v41 raw = 0.3604**.
- **Цель:** public ≥ 0.43 для топ-1.
- **Доступно сабмитов:** 1 (после сабмита v41).

### Что знаем

1. **V11 + p2 + raw cold — рабочий рецепт.** Все три компонента нужны.
2. **Raw-LTR на cold > fallback**, если cold raw в val > 0.2816.
3. **Local метрики (holdout, honest) не предсказывают public** для
   V11/V12-семьи; корректно предсказывают только для V4-семьи.
4. **p2-вес = 3.0 — оптимум**, 7.0 overfits.

### Следующие шаги

1. **v42 = V11 + p2=3.0 + p0mask=3.0 + raw cold.** Единственное отличие от
   v41 — p0mask=3.0 (настоящий cold в обучении). Гипотеза: cold raw вырастет
   с 0.3372 до 0.35–0.36, public +0.005..+0.010.
2. **Если v42 выиграет** — v43 = V11 + p2=3.0 + p0mask=5.0 (в v36 p0mask=5.0
   дал насыщение, но для V11 может быть иначе).
3. **Если v42 не выиграет** — второй слот на ансамбль v41 + v7 (два
   feature-set, страховка).

---

## 10. Известные проблемы (post-mortem)

1. **val ≠ test по составу.** Cold 2.9% vs 35.2%, warm P1 50% vs 2%.
2. **available одинаков внутри сплита.** Rank-фичи через `pop_by_institute`
   на cold вырождаются в алфавит.
3. **Holdout шумит ±0.0075** и **не коррелирует с public** при p2-тюнинге.
4. **Honest eval нестабилен между семьями** (+0.004 для V4, −0.032 для V11/V12).
5. **`src/metric.py` не в git** — `.gitignore` паттерн `metric.py` без слэша.
6. **Новая data (5002) сдвинула baseline** −0.011.
7. **Fallback — не всегда друг.** Он поднимает слабую модель (v7: cold 0.22→0.28),
   но вредит сильной (v41: cold 0.34→0.28). Решение: сравнивать raw vs fallback
   на val перед сабмитом.
