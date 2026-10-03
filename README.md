# Формула выбора

Решение задачи рекомендации элективных дисциплин для студентов МГПУ
в рамках хакатона «Академия искусственного интеллекта» (2026).

**Задача:** для каждого студента тестовой выборки сформировать ранжированный
список из 5 наиболее релевантных дисциплин из числа доступных ему
(`available_modules`), упорядоченных по убыванию предполагаемой
заинтересованности.

**Метрика:** `pNDCG@5` — priority-weighted Normalized Discounted Cumulative
Gain с весами `{1: 1.00, 2: 0.80, 3: 0.60, 4: 0.45, 5: 0.35}`.

## Текущие результаты

| Модель | val mean | cold | warm | test_estimate |
| --- | --- | --- | --- | --- |
| Baseline (взвешенная популярность) | 0.2368 | — | — | — |
| LTR v1 (25 фич, без fallback) | 0.3529 | — | — | — |
| LTR v2 (30 фич, PPMI + TF-IDF) | 0.3572 | — | — | — |
| LTR + augmentation (без fallback) | 0.3627 | 0.0475 | 0.3710 | 0.2699 |
| LTR + augmentation + cold-start fallback | 0.3693 | 0.2816 | 0.3719 | 0.3401 |
| **LTR с маскированием + fallback** (текущая) | **0.3705** | **0.2816** | **0.3732** | **0.3409** |

`test_estimate` — перевзвешенная оценка под распределение test
(`cold_share = 0.352`), компенсирует дисбаланс cold-студентов между
val (2.9%) и test (35.2%).

## 1. Структура проекта

```txt
formula_vybora/
├── data/                              # данные соревнования (не в git)
│   ├── train.jsonl
│   ├── val.jsonl
│   ├── test.jsonl
│   ├── modules.jsonl
│   ├── val_solution.csv
│   ├── sample_submission.csv
│   └── _cache/                        # pickle-кэш (генерируется автоматически)
│       └── all_data.pkl
├── src/
│   ├── __init__.py
│   ├── check_setup.py                 # проверка окружения
│   ├── data_loader.py                 # загрузка JSONL/CSV, кэширование
│   ├── metric.py                      # pNDCG@5 (соответствует официальной)
│   ├── evaluate.py                    # CLI-оценка submission по метрике
│   ├── popularity.py                  # baseline: взвешенная популярность
│   ├── content.py                     # TF-IDF модель для cosine
│   ├── features.py                    # feature engineering (30 фич) + маскирование
│   ├── ltr_model.py                   # LightGBM Ranker + sample weights
│   ├── cold_start.py                  # cold-start fallback (global / institute)
│   ├── segment_analysis.py            # анализ по cold/warm/institute
│   ├── inference.py                   # top-k рекомендации
│   ├── submit.py                      # запись + валидация формата
│   ├── validate_submission.py         # разовая валидация submission перед загрузкой
│   ├── run_baseline.py                # end-to-end baseline
│   └── run_ltr.py                     # end-to-end LTR с маскированием и fallback
├── notebooks/
│   └── eda.ipynb                      # разведочный анализ
├── models/
│   └── lgbm_ranker.txt                # сохранённая модель LightGBM
├── submissions/
│   ├── val_submission_ltr.csv         # LTR + fallback (победитель по cold)
│   ├── val_submission_popularity.csv  # baseline
│   ├── submission_popularity.csv      # baseline для test
│   └── submission.csv                 # финальный сабмит для test
├── metric.py                          # официальный чекер организаторов
├── requirements.txt
├── ruff.toml
└── README.md
```

## 2. Установка и запуск

### 2.1. Требования

- Python 3.11+ (проверено на 3.14)
- VS Code с расширениями: Python, Pylance, Jupyter, Ruff

### 2.2. Установка

```bash
python -m venv .venv
# Windows:
.\.venv\Scripts\Activate.ps1

pip install -r requirements.txt
```

### 2.3. Данные

Положить в `data/`:

- `train.jsonl`, `val.jsonl`, `test.jsonl`, `modules.jsonl`
- `val_solution.csv`, `sample_submission.csv`
- `metric.py` — в **корень проекта** (не в `data/`).

### 2.4. Быстрая проверка

```bash
python src/check_setup.py         # все библиотеки OK
python -m src.data_loader         # сводка по данным
```

### 2.5. Пайплайны

**Baseline (популярность):**

```bash
python -m src.run_baseline
```

**LTR с маскированием и cold-start fallback:**

```bash
python -m src.run_ltr             # обучает, оценивает, сохраняет submission.csv
```

**Сегментный анализ любого submission:**

```bash
python -m src.segment_analysis --submission submissions/val_submission_ltr.csv
```

**Сравнение стратегий cold-start fallback:**

```bash
python -m src.cold_start
```

**Валидация submission перед загрузкой на платформу:**

```bash
python -m src.validate_submission
```

## 3. Текущее состояние проекта

### 3.1. Реализовано

| Модуль | Назначение |
| --- | --- |
| `data_loader.py` | Загрузка JSONL/CSV, pickle-кэш |
| `metric.py` | Локальная pNDCG@5, совпадает с официальной |
| `evaluate.py` | Оценка submission + сверка с официальным `metric.py` |
| `popularity.py` | Взвешенная популярность (global + by institute) |
| `content.py` | TF-IDF по описаниям, cosine |
| `features.py` | 30 признаков (S, M, S×M), маскирование профиля |
| `ltr_model.py` | LightGBM Ranker, custom `pndcg5_eval`, sample weights |
| `cold_start.py` | Fallback для cold-студентов: global / institute-inferred |
| `segment_analysis.py` | Разбивка метрики по cold / warm / institute |
| `inference.py` | Top-k рекомендации |
| `submit.py` | Запись submission.csv + валидация |
| `validate_submission.py` | Финальная проверка перед платформой |
| `run_baseline.py` | Baseline end-to-end |
| `run_ltr.py` | LTR end-to-end с тремя train-срезами и fallback |
| `notebooks/eda.ipynb` | Разведочный анализ данных |

### 3.2. Не реализовано

| Этап | Приоритет |
| --- | --- |
| Multi-seed ensemble | Средний |
| Item-item CF через SVD | Средний |
| Тюнинг гиперпараметров LTR | Средний |
| `notebooks/model.ipynb` для код-ревью | Высокий (deliverable) |
| Обучение финальной модели на `train + val` | Средний |

## 4. Ключевые выводы EDA

### 4.1. Размеры данных

| Файл | Студентов |
| --- | --- |
| train | 5 002 |
| val | 3 043 |
| test | 4 306 |
| modules | 322 (каталог) |

### 4.2. Time-based split

Студенты **пересекаются** между train / val / test (~97%).
Профиль и история у пересекающихся совпадают. Это не утечка —
это time-based split: у одного студента разные целевые периоды.
**Верная схема:** `past = history[1..t-1] → target = history[t]`.

### 4.3. Available_modules

- В val — ровно 159 модулей (модули целевого периода).
- Внутри института набор одинаков, но, как показал анализ,
  сигнатура `available_modules` **не** восстанавливает институт однозначно:
  8 институтов сжимаются в 2 уникальных набора.
- Ground truth **на 100%** попадает в `available_modules`.

### 4.4. Популярность — long-tail

| Топ-K | Покрытие выборов |
| --- | --- |
| 10 | 16.0% |
| 20 | 27.6% |
| 50 | 51.1% |
| 100 | 74.8% |

### 4.5. Холодный старт

| Выборка | Cold-студентов |
| --- | --- |
| val | 2.9% (88 / 3043) |
| **test** | **35.2%** (1516 / 4306) |

Публичный val-скор **систематически завышен** относительно private
из-за дисбаланса cold-студентов. Разрыв `val − test_estimate ≈ +0.03`.

### 4.6. Test содержит богатые истории

1 310 студентов в test имеют 3 периода истории — для них сигнала больше,
чем у val-студентов (у которых максимум 2).

## 5. Архитектура решения

### 5.1. Общий поток

```txt
train.jsonl ──► FeatureStats.fit() ──┐
                                      ▼
train.jsonl ──► build_dataset(past=[1], mask=False) ──► p1 (warm, w=1.0)
            ──► build_dataset(past=[],  mask=False) ──► p0 (no-hist, w=0.5)
            ──► build_dataset(past=[1], mask=True)  ──► pmask (cold, w=1.0)
                                      │
                                      ▼
                             encode_categoricals()
                                      │
                                      ▼
                       prepare_xygw() → X, y, groups, weights
                                      │
                                      ▼
                LightGBM Ranker (lambdarank, sample weights)
                  label_gain = [0, .35, .45, .60, .80, 1.0]
                  feval = pndcg5_eval (custom)
                                      │
                                      ▼
val.jsonl  ──► build_dataset(past=[1,2]) ──► predict_scores()
                                      │
                                      ▼
                        top_k_per_student(k=5) → rec_ltr
                                      │
                                      ▼
            cold_start: rec_ltr + fallback (global / institute)
                                      │
                                      ▼
                      выбор победителя по cold-сегменту
                                      │
                                      ▼
                              submission.csv
```

### 5.2. Признаки (30)

**S — студент (7):**
`institute`, `specialty`, `n_periods`, `n_choices`, `n_unique_modules`,
`avg_history_priority`, `is_cold`.

**M — модуль (9):**
`pop_weighted_global`, `pop_raw_global`, `pop_by_institute`,
`pop_by_specialty`, `pop_period_1`, `pop_period_2`,
`n_periods_available`, `title_len`, `desc_len`.

**S×M — взаимодействие (14):**

- Co-occurrence raw: `already_chosen`, `cooc_score`, `cooc_max`, `cooc_norm`
- Co-occurrence PPMI: `cooc_ppmi_score`, `cooc_ppmi_max`, `cooc_ppmi_norm`
- TF-IDF cosine: `tfidf_cos_mean`, `tfidf_cos_max`
- Прочее: `pop_ratio_in_institute`, `pop_rank_in_available`,
  `pop_rank_frac`, `n_available`, `was_available_before`

### 5.3. Data augmentation — три среза

| Срез | Строк | Weight | Что моделирует |
| --- | --- | --- | --- |
| `p1`: past=[1] → p2 | 830 676 | 1.0 | warm студент |
| `p0`: past=[] → p1 | 652 876 | 0.5 | есть профиль, нет истории |
| `pmask`: past=[1] → p2, mask | 830 676 | 1.0 | полностью cold |

Префиксы `p1__` / `p0__` / `pmask__` в `student_id` — чтобы LightGBM
не смешал группы из разных срезов.

### 5.4. Маскирование профиля

При `mask_profile=True` студент обрабатывается как cold:
`profile = {}`, `history = []`. Все S и S×M фичи автоматически
обнуляются (`is_cold=1`, `n_choices=0`, `cooc_score=0`,
`tfidf_cos_mean=0`, `already_chosen=0`). Сохраняются только
M-признаки — единственный сигнал, доступный реальному cold-студенту.

Это позволило модели **обучиться работать в режиме без профиля**
и поднять cold-сегмент с 0.0475 до 0.2206 **без fallback**.

### 5.5. Cold-start fallback

Две стратегии:

- **A: global popularity** — top-5 по взвешенной популярности среди
  `available_modules`.
- **B: institute-inferred** — восстановление института по сигнатуре
  `available_modules`, затем top-5 по популярности института.

На val стратегия B не превзошла A (индекс сигнатур содержит всего 2
уникальных набора, восстановление невозможно). Победитель выбирается
автоматически по cold-сегменту: на текущей итерации — **A (global)**,
cold = 0.2816.

### 5.6. Custom eval

Стандартный LightGBM-NCDG **не совпадает** с нашей метрикой:

- LightGBM: `NDCG = DCG / IDCG` (полный IDCG, идеал = упорядоченные 5 модулей).
- Официальная метрика: `pNDCG = min(1.0, DCG / 1.0)` (идеал = priority-1
  на первой позиции).

Реализована `pndcg5_eval` — early stopping ориентируется **именно на нашу
метрику**.

## 6. Feature importance (по gain, топ-10)

Из последнего запуска `run_ltr`:

| Feature | Gain |
| --- | --- |
| `cooc_ppmi_score` | 60 678 |
| `pop_weighted_global` | 53 513 |
| `pop_rank_in_available` | 50 940 |
| `pop_rank_frac` | 38 843 |
| `pop_period_2` | 32 272 |
| `specialty` | 28 424 |
| `pop_raw_global` | 22 460 |
| `cooc_ppmi_norm` | 20 976 |
| `tfidf_cos_max` | 19 975 |
| `is_cold` | 16 424 |

После добавления маскирования `pop_weighted_global` поднялся
со 7-го на 2-е место, а `is_cold` вошёл в топ-10 — модель активно
использует оба режима (warm / cold).

## 7. История экспериментов

| # | Модель / Изменение | val mean | Δ |
| --- | --- | --- | --- |
| 1 | Baseline (популярность) | 0.2368 | — |
| 2 | LTR, 25 фич, lr=0.05, num_leaves=63 | 0.3529 | +0.116 |
| 3 | + custom eval (наш pNDCG@5) | 0.3529 | 0 |
| 4 | + lr=0.02, num_leaves=31, patience=200 | 0.3558 | +0.003 |
| 5 | + PPMI фичи + TF-IDF cosine | 0.3572 | +0.001 |
| 6 | + augmentation `past=[]` | 0.3627 | +0.006 |
| 7 | + cold-start fallback (global) | 0.3693 | +0.007 |
| 8 | + маскирование + 3 среза + sample weights | **0.3705** | +0.001 |

`test_estimate` (перевзвешенная под cold_share=0.352):

| Шаг | test_estimate | Δ |
| --- | --- | --- |
| Шаг 6 (LTR без fallback) | 0.2699 | — |
| Шаг 7 (+ fallback) | 0.3401 | **+0.070** |
| **Шаг 8 (+ маскирование)** | **0.3409** | +0.001 |

Основной прирост дал cold-start fallback. Маскирование улучшило
устойчивость модели на cold-подобных примерах и слегка подняло warm.

## 8. Технические решения и обоснования

### 8.1. Кэширование данных

Парсинг JSONL занимает 2–3 сек. После первого запуска данные лежат в
`data/_cache/all_data.pkl` и загружаются мгновенно.
Отключить: `load_all(use_cache=False)`.

### 8.2. Fit-then-transform

Все агрегаты (`pop_weighted`, `cooc`, `PPMI`, TF-IDF) считаются **один раз
на train** и переиспользуются для val / test. Это устраняет
training-serving skew.

### 8.3. Защита от утечек

При обучении на срезе `past=[1] → target=period 2` из истории студента
**исключается период 2**. Иначе в `cooc_score` модель увидела бы свой
же target.

### 8.4. Воспроизводимость

- Детерминированные сортировки (tie-break по `module_id`).
- `seed=42` в LightGBM.
- `requirements.txt` фиксирует версии.
- Кэш не влияет на результат — только на скорость.

### 8.5. Линтер

`ruff.toml` под `target-version = "py314"`, включены правила
`E, F, I, UP, B, SIM`. Код в PEP 585-стиле.

## 9. Формат submission

```csv
id,recommendations
s_000123,m_014 m_087 m_031 m_155 m_022
s_000124,m_009 m_102 m_034 m_077 m_141
```

Правила (валидируются в `submit.validate_submission` и
`validate_submission.py`):

- одна строка на студента;
- ровно 5 различных `module_id`;
- порядок — от лучшего к худшему;
- `module_id` разделены пробелом;
- все `module_id` присутствуют в каталоге.

## 10. Дальнейшие шаги

1. **Загрузить текущий submission на платформу.** Текущий
   `submissions/submission.csv` (4306 строк, LTR + fallback global)
   прошёл валидацию и готов к отправке. Public score позволит
   откалибровать `test_estimate`.

2. **Multi-seed ensemble** (5 моделей с разными seed) — ожидаемый
   прирост `+0.003–0.008` на warm.

3. **Item-item CF через SVD** — независимый сигнал для ансамбля.

4. **Тюнинг гиперпараметров LTR** — `learning_rate`, `num_leaves`,
   регуляризация.

5. **Финальный пайплайн:**
   - обучение на `train + val`;
   - предсказание для `test`;
   - сохранение `submissions/submission.csv`.

6. **`notebooks/model.ipynb`** — единый артефакт для код-ревью
   с полным описанием пайплайна.

## 11. Известные ограничения и риски

| Риск | Митигация |
| --- | --- |
| Cold-студенты в test (35.2%) | Cold-start fallback + маскированное обучение |
| Разрыв val (2.9% cold) vs test (35.2% cold) | Перевзвешенная оценка `test_estimate` |
| `best_iteration=88` — модель недоучена | Тюнинг learning rate, больше данных через augmentation |
| Institute-inferred fallback не работает | Оставлен как опция, выбирается автоматически |
| Невоспроизводимость на код-ревью | Фиксированные версии, seed, README с инструкцией |
| Лимит 2 сабмита в день на платформе | Все эксперименты — локально, платформа — только для финальных решений |
