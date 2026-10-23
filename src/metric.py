"""
Локальная метрика pNDCG@5 — согласована с официальным чекером.

pNDCG@5 = min(1.0, DCG@5)
DCG@5   = Σ_{i=1..5} weight(priority_i) / log2(i + 1)
weights = {1: 1.00, 2: 0.80, 3: 0.60, 4: 0.45, 5: 0.35}

ВАЖНО: score ОБРЕЗАЕТСЯ сверху единицей. Это не классический NDCG.
Идеальный ответ — priority-1 модуль на позиции 1 (даёт ровно 1.0).

Публичный API:
    parse_ground_truth(raw)               → dict[str, int]
    pndcg_at_k(rec, target, k=5)          → float
    evaluate_pndcg5(rec_all, gt_raw, k=5) → (mean, per_student)
    hit_at_k(rec, target, k=5)            → int (0/1)
    recall_at_k(rec, target, k=5)         → float (|∩| / |target|)
"""
from __future__ import annotations

import math

PRIORITY_WEIGHTS: dict[int, float] = {
    1: 1.00, 2: 0.80, 3: 0.60, 4: 0.45, 5: 0.35,
}


def parse_ground_truth(raw: str | None) -> dict[str, int]:
    """
    'm_001:1 m_002:3 ...' → {module_id: priority}.

    Устойчив к обрамляющим кавычкам (CSV после pandas). None/пустое → {}.
    Битые токены (без ':' или с непарсящимся приоритетом) пропускаются.
    """
    if raw is None:
        return {}
    s = str(raw).strip()
    if not s:
        return {}
    if len(s) >= 2 and s.startswith('"') and s.endswith('"'):
        s = s[1:-1].strip()
    result: dict[str, int] = {}
    for tok in s.split():
        if ":" not in tok:
            continue
        mid, prio_s = tok.split(":", 1)
        try:
            prio = int(prio_s)
        except ValueError:
            continue
        result[mid] = prio
    return result


def _dcg_at_k_raw(rec: list[str], target: dict[str, int], k: int = 5) -> float:
    """DCG@k БЕЗ clip — для диагностики. Не использовать напрямую в метрике."""
    dcg = 0.0
    for pos, mid in enumerate(rec[:k], start=1):
        prio = target.get(mid, 0)
        if prio <= 0:
            continue
        w = PRIORITY_WEIGHTS.get(prio, 0.0)
        if w > 0:
            dcg += w / math.log2(pos + 1)
    return dcg


def _as_target(target: dict[str, int] | str) -> dict[str, int]:
    """Приводит target к dict (парсит строку, если передали сырую)."""
    if isinstance(target, str):
        return parse_ground_truth(target)
    return target


def pndcg_at_k(
    rec: list[str],
    target: dict[str, int] | str,
    k: int = 5,
) -> float:
    """pNDCG@k одного студента со clip до 1.0 (как официальный чекер)."""
    tgt = _as_target(target)
    return min(1.0, _dcg_at_k_raw(rec, tgt, k=k))


def hit_at_k(
    rec: list[str],
    target: dict[str, int] | str,
    k: int = 5,
) -> int:
    """1 если хотя бы один из top-k попал в target (любой приоритет), иначе 0.

    Возвращает int (не bool), чтобы np.mean(hits) давал долю студентов.
    """
    tgt = _as_target(target)
    for mid in rec[:k]:
        if mid in tgt:
            return 1
    return 0


def recall_at_k(
    rec: list[str],
    target: dict[str, int] | str,
    k: int = 5,
) -> float:
    """|top-k ∩ target| / |target|. Знаменатель — len(target), не константа 5."""
    tgt = _as_target(target)
    if not tgt:
        return 0.0
    hit = sum(1 for mid in rec[:k] if mid in tgt)
    return hit / len(tgt)


def evaluate_pndcg5(
    rec: dict[str, list[str]],
    gt_raw: dict[str, str],
    k: int = 5,
) -> tuple[float, dict[str, float]]:
    """(mean, per_student). Пропускает студентов без rec или с пустым gt."""
    per_student: dict[str, float] = {}
    for sid, raw in gt_raw.items():
        if sid not in rec:
            continue
        target = parse_ground_truth(raw)
        if not target:
            continue
        per_student[sid] = min(1.0, _dcg_at_k_raw(rec[sid], target, k=k))
    if not per_student:
        return 0.0, {}
    mean = sum(per_student.values()) / len(per_student)
    return float(mean), per_student


if __name__ == "__main__":
    print("=" * 72)
    print("Self-test src/metric.py")
    print("=" * 72)

    # 1. Идеальный ответ: сумма весов ≥ 1.0 → clip до 1.0
    gt = "m_a:1 m_b:2 m_c:3 m_d:4 m_e:5"
    rec = ["m_a", "m_b", "m_c", "m_d", "m_e"]
    score = pndcg_at_k(rec, parse_ground_truth(gt))
    assert abs(score - 1.0) < 1e-9, f"test 1: {score} ≠ 1.0"
    print(f"  [1] perfect match        = {score:.4f}  (ожидали 1.0)")

    # 2. Только priority-1 на позиции 1 — тоже 1.0 (clip)
    rec = ["m_a", "m_x", "m_y", "m_z", "m_w"]
    score = pndcg_at_k(rec, parse_ground_truth(gt))
    assert abs(score - 1.0) < 1e-9, f"test 2: {score} ≠ 1.0"
    print(f"  [2] prio-1 на позиции 1  = {score:.4f}  (ожидали 1.0)")

    # 3. priority-1 на позиции 2: 1.0 / log2(3) ≈ 0.6309
    rec = ["m_x", "m_a", "m_y", "m_z", "m_w"]
    score = pndcg_at_k(rec, parse_ground_truth(gt))
    expected = 1.0 / math.log2(3)
    assert abs(score - expected) < 1e-9, f"test 3: {score} ≠ {expected}"
    print(f"  [3] prio-1 на позиции 2  = {score:.4f}  (ожидали {expected:.4f})")

    # 4. priority-5 на позиции 1: 0.35 / log2(2) = 0.35
    gt4 = "m_a:5"
    rec = ["m_a", "m_x", "m_y", "m_z", "m_w"]
    score = pndcg_at_k(rec, parse_ground_truth(gt4))
    assert abs(score - 0.35) < 1e-9, f"test 4: {score} ≠ 0.35"
    print(f"  [4] prio-5 на позиции 1  = {score:.4f}  (ожидали 0.35)")

    # 5. Совсем без попаданий → 0
    rec = ["m_x", "m_y", "m_z", "m_w", "m_v"]
    score = pndcg_at_k(rec, parse_ground_truth("m_a:1 m_b:2"))
    assert abs(score) < 1e-9, f"test 5: {score} ≠ 0"
    print(f"  [5] без попаданий        = {score:.4f}  (ожидали 0)")

    # 6. parse_ground_truth: кавычки, пустая строка, None, битый токен
    assert parse_ground_truth('"m_a:1 m_b:3"') == {"m_a": 1, "m_b": 3}
    assert parse_ground_truth("") == {}
    assert parse_ground_truth(None) == {}
    assert parse_ground_truth("m_a:1 broken m_b:3") == {"m_a": 1, "m_b": 3}
    print(f"  [6] parse_ground_truth   = OK")

    # 7. evaluate_pndcg5: mean + per_student
    rec_all = {
        "s1": ["m_a", "m_b", "m_x", "m_y", "m_z"],
        "s2": ["m_x", "m_a", "m_y", "m_z", "m_w"],
    }
    gt_all = {"s1": "m_a:1 m_b:2", "s2": "m_a:1"}
    mean, per = evaluate_pndcg5(rec_all, gt_all, k=5)
    s1 = 1.0 / math.log2(2) + 0.8 / math.log2(3)   # = 1.5047 → clip 1.0
    s2 = 1.0 / math.log2(3)                         # = 0.6309
    expected_mean = (min(1.0, s1) + s2) / 2
    assert abs(per["s1"] - min(1.0, s1)) < 1e-9
    assert abs(per["s2"] - s2) < 1e-9
    assert abs(mean - expected_mean) < 1e-9
    print(f"  [7] evaluate_pndcg5      = {mean:.4f}  (ожидали {expected_mean:.4f})")

    # 8. hit_at_k
    target8 = {"m_a": 1, "m_b": 2}
    assert hit_at_k(["m_a", "m_x"], target8) == 1
    assert hit_at_k(["m_x", "m_y"], target8) == 0
    assert hit_at_k(["m_x", "m_a", "m_y", "m_z", "m_w"], target8) == 1
    assert hit_at_k(["m_x", "m_y", "m_z", "m_w", "m_v"], target8) == 0
    assert hit_at_k([], target8) == 0
    assert hit_at_k(["m_a"], {}) == 0
    assert hit_at_k(["m_a"], "m_a:1 m_b:2") == 1   # строковый target
    print(f"  [8] hit_at_k             = OK")

    # 9. recall_at_k
    assert abs(recall_at_k(["m_a", "m_x"], {"m_a": 1, "m_b": 2}) - 0.5) < 1e-9
    assert abs(recall_at_k(["m_a", "m_b"], {"m_a": 1, "m_b": 2}) - 1.0) < 1e-9
    assert abs(recall_at_k([], {"m_a": 1, "m_b": 2})) < 1e-9
    assert abs(recall_at_k(["m_x"], {})) < 1e-9
    print(f"  [9] recall_at_k          = OK")

    # 10. target как сырая строка в pndcg_at_k
    assert abs(pndcg_at_k(["m_a"], "m_a:1") - 1.0) < 1e-9
    assert abs(pndcg_at_k([], "m_a:1")) < 1e-9
    print(f"  [10] pndcg_at_k(str target) = OK")

    print()
    print("Все тесты прошли. src/metric.py полностью согласован с evaluate.py.")