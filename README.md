# Формула выбора — хакатон «Академия искусственного интеллекта» (МГПУ, 2026)

Задача: для каждого студента из `test.jsonl` собрать ранжированный **топ-5**
дисциплин из числа доступных ему (`available_modules`), упорядоченных по
убыванию ожидаемой заинтересованности.

**Финальный результат: public = 0.3604** (v41, V11-семья).

## 1. Метрика

`pNDCG@5` — priority-weighted NDCG с **фиксированной нормой 1.0**:

```txt
pNDCG@5 = min(1.0, DCG@5)
DCG@5   = Σ_{i=1..5} weight(priority_i) / log2(i + 1)
weights = {1: 1.00, 2: 0.80, 3: 0.60, 4: 0.45, 5: 0.35}
```

Score обрезается сверху единицей. Идеальный ответ — priority-1 модуль на позиции 1.

## 2. Данные (`data/`)

| Файл | Содержимое |
| --- | --- |
| `train.jsonl` | 5002 студента (P1:P2 = 1899:3103) |
| `val.jsonl` | 3043 студента, 159 модулей |
| `test.jsonl` | 4306 студентов, 158 модулей |
| `modules.jsonl` | 322 модуля |
| `val_solution.csv` | 3043 строк |

### Состав сплитов

| Сегмент | test | val |
| --- | --- | --- |
| cold | **35.2%** (1516) | 2.9% (88) |
| warm P1 | 2.0% (87) | 50.3% (1531) |
| warm P2 | 32.4% (1393) | 46.8% (1424) |
| warm P3 | 30.4% (1310) | — |

**Ключевые особенности:**

- `available_modules` одинаков для всех студентов внутри сплита.
- `Jaccard(val_avail, test_avail) = 0.4676`.
- 57 test-модулей нет в train-пуле; 88 train-модулей не встретятся в test.

## 3. Champion: v41 raw = **0.3604** public

### Конфигурация

- **34 признака** `FEATURE_COLS_V11` = V4 + 3 глобальные rank-фичи.
- Срезы: `p0` (0.5), `p1` (1.0), `pmask` (1.0), `p2` (3.0).
- LightGBM `lambdarank`, `truncation_level=15`, `lr=0.05`, `num_leaves=31`.
- Multi-seed **5** (42..46).
- Test: `max_history_periods=2` + нормализация периодов.
- **Инференс: raw-LTR на cold (БЕЗ fallback).**

## 4. Наборы признаков

| Набор | # | Особенность | Public |
| --- | --- | --- | --- |
| `v4` | 31 | V7 baseline | 0.3240 |
| `v7` | 28 | V4 без rank-3 | 0.3094 |
| **`v11`** | **34** | **V4 + 3 глобальные rank** | **0.3604** |
| `v12` | 31 | V7 + 3 глобальные rank | 0.3168 |
| `v14` | 37 | V11 + pop_delta | LGBM игнорирует |
| `v15` | 33 | V11 − pop_rank_global_frac | cold −0.053 |
| `v16` | 32 | V11 − title_len − desc_len | нейтрально |

## 5. Финальная таблица экспериментов

### Успех

| Модель | Public |
| --- | --- |
| **v41 (V11, p2=3, raw cold)** | **0.3604** |

### Регрессы

| Модель | Public | Δ vs v41 |
| --- | --- | --- |
| v7 (V4, fallback) | 0.3240 | −0.0364 |
| v40 (V4, p2=7) | 0.3205 | −0.0399 |
| v38 (V12 + p0mask) | 0.3168 | −0.0436 |
| v39 (V11 + p0mask) | 0.3142 | −0.0462 |
| v43 (гибрид cold v43/warm v41) | 0.2617 | **−0.0987** |

### Локальные провалы

| Эксперимент | Результат |
| --- | --- |
| v45 XGBoost-ансамбль | warm 0.4467 (хуже LGBM) |
| v46 two-stage (classifier → ranker) | recall@30=0.68, warm 0.4094 |
| v47 label_gain=1.5, trunc=30 | cold 0.3063 (−0.031) |
| v50 BPR-MF | warm 0.2374 |
| v14 pop_delta | 0/3 в топ-10 importance |
| v15 ablation | cold 0.2840 (−0.053) |
| v16 ablation | нейтрально |
| p2-тюнинг (5.0, 7.0) | holdout ↑, public ↓ |
| p0mask=1, p0mask=3 | cold на test = 0.028 |
| noGRA (V11 − pop_rank_global_in_avail) | warm +0.004, cold −0.040 |

## 6. Post-mortem

### Данные

1. **val ≠ test по составу.** Cold 2.9% vs 35.2%, warm P1 50% vs 2%.
2. **available_modules одинаков внутри сплита.** Rank-фичи через `pop_by_institute` вырождаются для cold в алфавитную сортировку.
3. **Jaccard(train_avail, test_avail) = 0.42.** 57 test-модулей нет в train-пуле.

### Метрики

1. **Holdout на p2-срезе токсичен.** Растёт при p2=5→7, но public падает. Не использовать для решения о сабмите.
2. **Honest eval нестабилен между семьями.** +0.004 для V4, −0.032 для V11/V12 с p0mask. Направление корректно, величина — нет.
3. **`p0_mask` — яд для cold.** На val cold +0.008, на public cold −0.32 (0.34 → 0.028). Все модели с p0mask нельзя выпускать на cold-сегмент.

### Архитектуры

1. **XGBoost `rank:pairwise` — на 0.09 хуже LGBM.** Не знает clip-1.0. Не помогает в ансамбле.
2. **Two-stage classifier → ranker не работает** при 3.2% positive rate. Recall@30 = 0.68 — target теряется.
3. **BPR-MF не подходит** для 5002 users × 322 items. Разреженность убивает collaborative patterns.

### Feature engineering

1. **`rank-3` — реальный сигнал**, а не мусор. Удаление = −0.025.
2. **Глобальные rank-фичи (V11)** — единственное, что работает на cold. Три фичи вместе → +0.036.
3. **`pop_rank_global_frac` — не дубль** `pop_rank_global_in_avail` на практике (ablation v15 = −0.053).
4. **`title_len`/`desc_len` — нейтральны**, не ID-прокси (ablation v16).
5. **`pop_delta` — LGBM игнорирует.** Популярность по периодам уже покрыта `pop_last/prev_period`.

### Fallback и seeds

1. **Fallback — не всегда друг.** Слабую модель поднимает (v7: 0.22→0.28), сильную топит (v41: 0.34→0.28).
2. **5 seeds минимум** для финальной модели. 3 seeds — шум ±0.004.

## 7. Структура проекта

```txt
formula_vybora/
├── data/                          # данные (нет в архиве)
├── src/
│   ├── data_loader.py             # JSONL + кэш
│   ├── metric.py                  # pNDCG@5 (самодостаточный, self-test)
│   ├── features.py                # FeatureStats + build_dataset
│   ├── ltr_model.py               # LightGBM + FEATURE_SETS
│   ├── content.py / content_embed.py
│   ├── popularity.py / cold_start.py
│   ├── evaluate.py                # CLI-сверка с официальным metric.py
│   ├── submit.py
│   ├── kfold_eval.py / eval_clean.py / eval_clean_p0mask.py
│   ├── estimate_test_score.py
│   ├── swap_cold_raw.py / swap_cold_hybrid.py
│   ├── diag_*.py                  # диагностика
│   └── run_ltr_v*.py              # эксперименты
├── submissions/                   # сабмиты (нет в архиве)
├── models/                        # бустеры (нет в архиве)
├── metric.py                      # официальный чекер (нет в архиве)
└── README.md
```

## 8. Быстрый старт

```bash
# активировать venv
.venv\Scripts\activate

# champion v41 (public 0.3604)
python -m src.run_ltr_v42 --feature-set v11 --weight-p2 3.0 --n-seeds 5
python -m src.swap_cold_raw   # raw-LTR на cold вместо fallback

# проверка метрики
python -m src.metric
python -m src.evaluate --submission submissions/val_submission_ltr_v7.csv
```

**Восстановление из архива:**

1. `python -m venv .venv && .venv\Scripts\activate`
2. `pip install -r requirements.txt`
3. Положить `data/` (5 файлов).
4. Восстановить `src/metric.py`
5. `python -c "from src.data_loader import load_all; load_all()"` — построит кэш.

## 9. Итог

- **Champion: v41 = 0.3604 public.**
- Потолок архитектуры — 0.365–0.370.
- Все направления исчерпаны (V11-семья, embeddings, XGBoost, two-stage, BPR-MF, p2-тюнинг, p0mask, pop_delta, ablation).
