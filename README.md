# Формула выбора

Решение задачи рекомендации элективных дисциплин для студентов МГПУ
в рамках хакатона «Академия искусственного интеллекта» (2026).

**Задача:** для каждого студента тестовой выборки сформировать ранжированный
список из 5 наиболее релевантных дисциплин из числа доступных ему
(`available_modules`), упорядоченных по убыванию предполагаемой
заинтересованности.

**Метрика:** `pNDCG@5` — priority-weighted Normalized Discounted Cumulative
Gain с весами `{1: 1.00, 2: 0.80, 3: 0.60, 4: 0.45, 5: 0.35}`.

## Результаты

| Модель | Holdout | Public |
| --- | --- | --- |
| Baseline (взвешенная популярность) | — | — |
| LTR v1 (30 фич, без fallback) | — | 0.3129 |
| LTR v4 (относительные периоды) | 0.4987 | 0.3189 |
| LTR v6 (обрезка test до 2 периодов) | 0.4987 | **0.3326** |
| **LTR v7 (multi-seed + w_p2=3.0)** | **0.5134** | в очереди |

**Holdout** — 284 студента val, извлечённых как 20% стратифицированной
выборки из группы `periods=2`. Модель их не видела при обучении.

**Public** — публичный скор на платформе.

## Структура проекта

```txt
formula_vybora/
├── data/                              # данные соревнования (не в git)
│   ├── train.jsonl
│   ├── val.jsonl
│   ├── test.jsonl
│   ├── modules.jsonl
│   ├── val_solution.csv
│   ├── sample_submission.csv
│   └── _cache/all_data.pkl            # pickle-кэш
├── src/
│   ├── data_loader.py                 # загрузка JSONL/CSV, кэширование
│   ├── metric.py                      # pNDCG@5
│   ├── evaluate.py                    # CLI-оценка submission
│   ├── popularity.py                  # baseline и fallback
│   ├── content.py                     # TF-IDF + cosine
│   ├── features.py                    # feature engineering (31 признак)
│   ├── ltr_model.py                   # LightGBM Ranker + custom eval
│   ├── cold_start.py                  # cold-start fallback
│   ├── segment_analysis.py            # разбивка метрики по сегментам
│   ├── test_diagnostic.py             # структура test по группам
│   ├── test_period3_diagnostic.py     # сравнение periods=2 vs periods=3
│   ├── validate_submission.py         # проверка формата submission
│   ├── submit.py                      # запись submission.csv
│   ├── inference.py                   # top-k рекомендации
│   ├── run_baseline.py                # baseline end-to-end
│   ├── run_ltr.py                     # LTR v1
│   ├── run_ltr_v4.py                  # LTR v4
│   ├── run_ltr_v5.py                  # LTR v5 (transition, отброшен)
│   ├── run_ltr_v6.py                  # LTR v6
│   └── run_ltr_v7.py                  # LTR v7 (multi-seed + w_p2)
├── notebooks/
│   └── eda.ipynb                      # разведочный анализ
├── models/                            # сохранённые бустеры
├── submissions/                       # сгенерированные сабмиты
├── metric.py                          # официальный чекер организаторов
├── requirements.txt
├── ruff.toml
└── README.md
```

## Установка и запуск

```bash
python -m venv .venv
.\.venv\Scripts\Activate.ps1     # Windows
source .venv/bin/activate         # macOS/Linux
pip install -r requirements.txt
```

Данные в `data/`, `metric.py` — в корне проекта.

```bash
python src/check_setup.py                          # проверка окружения
python -m src.data_loader                          # сводка по данным
python -m src.run_ltr_v7                           # основной пайплайн
python -m src.validate_submission                  # валидация submission
python -m src.segment_analysis                     # метрики по сегментам
```

## Архитектура

### Признаки (31)

**Студент (7):** `institute`, `specialty`, `n_periods`, `n_choices`,
`n_unique_modules`, `avg_history_priority`, `is_cold`.

**Модуль (10):** `pop_weighted_global`, `pop_raw_global`, `pop_by_institute`,
`pop_by_specialty`, `pop_last_period`, `pop_prev_period`, `pop_first_period`,
`n_periods_available`, `title_len`, `desc_len`.

**Взаимодействие (14):** `already_chosen`, `cooc_score/max/norm`,
`cooc_ppmi_score/max/norm`, `tfidf_cos_mean/max`, `pop_ratio_in_institute`,
`pop_rank_in_available`, `pop_rank_frac`, `n_available`, `was_available_before`.

### Обучающие срезы

| Срез | Строк | Weight | Что моделирует |
| --- | --- | --- | --- |
| p1: `[P1]` → target P2 | 830 676 | 1.0 | студент с 1 периодом истории |
| p0: `[]` → target P1 | 652 876 | 0.5 | студент с профилем, без истории |
| pmask: `[P1]` → target P2, профиль стёрт | 830 676 | 1.0 | холодный студент |
| p2: `[P1,P2]` → target P3 (90% val) | 181 260 | 3.0 | формат test periods=2 |

**Holdout:** 20% val periods=2, извлечённых стратифицированно (seed=42),
не участвуют в обучении.

**Early stopping** — на holdout по кастомной метрике `pndcg5_eval`, не по
стандартной NDCG LightGBM (формулы различаются: официальная метрика
обрезает DCG сверху единицей).

### Инференс

1. **Train и val:** вся доступная история (не более 2 периодов).
2. **Test:** обрезка истории до последних 2 периодов (`max_history_periods=2`).
   Выравнивает формат test-студентов с 3 периодами под формат обучающих срезов.
3. **Cold-start fallback:** студенты без профиля и истории получают top-5
   по взвешенной популярности среди `available_modules`.
4. **Multi-seed:** 5 моделей с seed 42..46, усреднение скоров.

## Ключевые находки

### Структура test

| Группа | N | Доля | История | Target |
| --- | --- | --- | --- | --- |
| periods=0 | 1516 | 35.2% | `[]` | P1 |
| periods=1 | 87 | 2.0% | `[P1]`, `[P2]`, `[P3]` | P2/P3/P4 |
| periods=2 | 1393 | 32.4% | `[P2,P3]` | P4 |
| periods=3 | 1310 | 30.4% | `[P1,P2,P3]` | P4 |

Все четыре группы имеют одинаковую сигнатуру `available_modules`
(158 модулей), то есть фильтр по институту не работает.

### Разрыв val/test

В val холодных студентов 2.9%, в test — 35.2%. Val-скор систематически
завышен на ≈0.05–0.08. Пересчёт под test-распределение:
`test_estimate = 0.352 × cold + 0.648 × warm`.

### Реконструкция по группам из public

Из public v6 = 0.3326:

- periods=0 ≈ 0.28 (fallback)
- periods=1 ≈ 0.32
- periods=2 ≈ 0.50
- **periods=3 ≈ 0.22** — самый проблемный сегмент

### Сдвиг формата периодов

Модель обучена на строго 1–2-периодных срезах. С 3 периодами встречается
только на test. Обрезка `max_history_periods=2` подняла public v4 → v6
с 0.3189 до 0.3326 (+0.014).

## История экспериментов

| # | Изменение | Holdout | Public | Δ public |
| --- | --- | --- | --- | --- |
| 1 | LTR v1: 30 фич, augmentation, fallback | — | 0.3129 | baseline |
| 2 | v4: относительные периоды | 0.4987 | 0.3189 | +0.006 |
| 3 | v5: transition-фичи (утечка, отброшено) | — | — | — |
| 4 | v6: обрезка test до 2 периодов | 0.4987 | 0.3326 | +0.014 |
| 5 | **v7: multi-seed (5) + w_p2=3.0** | **0.5134** | в очереди | ожидается +0.01 |

**Отброшенные гипотезы:**

- Institute-inferred cold fallback — сигнатуры `available_modules` сжимаются
  в 2 уникальных набора, восстановление института невозможно.
- Transition-матрица между соседними периодами — тестовая диагностика
  показала 0.5805, но после устранения утечки — 0.4999 (= v4). Не используется.
- Ансамбль v4 + v6 — v6 строго лучше v4 на periods=3, усреднение тянет
  качество вниз.

## Ограничения

| Ограничение | Комментарий |
| --- | --- |
| Test periods=3 (30.4%) | Даёт ≈0.22 против ≈0.50 у periods=2. Структурный сдвиг формата. |
| Cold-студенты в test (35.2%) | Fallback по популярности даёт ≈0.28 — потолок для этого подхода. |
| Разрыв val и test | ≈0.05–0.08 из-за дисбаланса cold. Локальные метрики завышены. |
| Лимит платформы | 2 сабмита в день. Все эксперименты — локально на holdout. |

## Зависимости

Python 3.11+, LightGBM, pandas, numpy, scikit-learn, scipy, matplotlib,
seaborn, tqdm, joblib, jupyter.

Версии зафиксированы в `requirements.txt`. Линтер — ruff, конфигурация
в `ruff.toml`.
