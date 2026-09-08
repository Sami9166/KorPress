"""
compressor.py

학습된 인코더로 각 span의 p(DROP)을 계산한 뒤(여기서는 더미/실제 라벨로 대체
가능), L·threshold를 적용하고 중첩 span 충돌을 처리해 압축문을 생성한다.

핵심: 이 파일의 로직(L 필터링 ~ 압축문 생성)은 p_drop이 "진짜 인코더 출력"이든
"랜덤 더미 값"이든 "실제 라벨(auto_label)"이든 상관없이 동일하게 작동한다.
그래서 encoder_train.py가 학습되는 동안, 여기 로직을 더미 확률로 먼저
구현·검증해두고, checkpoint가 나오면 predict_with_encoder()만 실제
모델 추론으로 갈아끼우면 된다.

충돌 처리 규칙 (팀 합의 사항):
    상위(더 큰) span이 DROP이면, 그 안에 포함된 하위 span은 판정과 무관하게
    강제 DROP. 상위가 KEEP이면 하위는 각자 자기 판정을 따른다.

사용법:
    python compressor.py --demo                 # 더미 확률로 데모
    python compressor.py --demo --real-labels ./span_labels.csv.gz --utterance-uid C00543_chunk0001_u5
"""

import argparse
import random
from collections import defaultdict
from typing import List, Dict, Any


# ============================================================
# 1. p(DROP) 획득 - 더미 / 실제 인코더 두 방식 모두 지원
# ============================================================

def assign_dummy_probs(spans: List[Dict[str, Any]], seed: int = 0) -> List[Dict[str, Any]]:
    """
    인코더 checkpoint가 아직 없을 때, 개발/테스트용으로 랜덤 p_drop을 부여.
    실제 checkpoint가 나오면 predict_with_encoder()로 교체.
    """
    rng = random.Random(seed)
    out = []
    for s in spans:
        s = dict(s)
        s["p_drop"] = rng.random()
        out.append(s)
    return out


def predict_with_encoder(spans: List[Dict[str, Any]], model, tokenizer, batch_size: int = 64):
    """
    (참고용) 표준 HuggingFace 분류 모델 형식을 가정한 추론 함수.
    주의: 다슬님이 실제로 만든 모델은 이 형식이 아니라 커스텀
    ContextualSpanClassifier(문맥 전체를 넣고 span 위치만 pooling)라
    이 함수로는 안 맞는다. 실제 연동은 아래 load_real_predictions()를 사용할 것.
    """
    import torch

    texts = [s["text"] for s in spans]
    all_probs = []
    model.eval()
    with torch.no_grad():
        for i in range(0, len(texts), batch_size):
            batch_texts = texts[i:i + batch_size]
            enc = tokenizer(batch_texts, truncation=True, max_length=64,
                             padding=True, return_tensors="pt")
            enc = {k: v.to(model.device) for k, v in enc.items()}
            logits = model(**enc).logits
            probs = torch.softmax(logits, dim=-1)[:, 1]  # index 1 = DROP class
            all_probs.extend(probs.cpu().tolist())

    out = []
    for s, p in zip(spans, all_probs):
        s = dict(s)
        s["p_drop"] = p
        out.append(s)
    return out


def load_real_predictions(span_labels_path: str, predictions_path: str, utterance_uid: str = None):
    """
    다슬님의 predict.py가 만든 예측 결과(span_id, drop_probability)를
    원본 라벨 데이터(span_labels.csv.gz - span_id, word_ids, size, utterance_uid 등)와
    span_id 기준으로 합쳐서, compress()가 바로 쓸 수 있는 spans 리스트를 만든다.

    utterance_uid를 지정하면 그 발화의 span만 반환 (데모/검증용).
    지정 안 하면 predictions_path에 있는 전체 span을 반환 (대량 처리용,
    다만 test.jsonl에 포함된 발화로 한정됨 - train/val 발화는 예측 결과가 없음).
    """
    import ast
    import pandas as pd

    labels_df = pd.read_csv(span_labels_path, compression="gzip" if span_labels_path.endswith(".gz") else None)
    preds_df = pd.read_csv(predictions_path, compression="gzip" if predictions_path.endswith(".gz") else None)

    if "span_id" not in labels_df or "span_id" not in preds_df:
        raise ValueError("라벨·예측 파일 모두 span_id 열이 필요합니다.")
    id_column = "word_ids" if "word_ids" in labels_df else "eojeol_indices"
    if id_column not in labels_df:
        raise ValueError("라벨 파일에 word_ids 또는 eojeol_indices 열이 없습니다.")
    uid_column = "utterance_uid" if "utterance_uid" in labels_df else "sentence_id"
    text_column = "span" if "span" in labels_df else "text"
    if text_column not in labels_df:
        raise ValueError("라벨 파일에 span 또는 text 열이 없습니다.")

    merged = labels_df.merge(preds_df[["span_id", "drop_probability"]], on="span_id", how="inner")

    if utterance_uid is not None:
        merged = merged[merged[uid_column] == utterance_uid]

    def parse_ids(value):
        parsed = value if isinstance(value, (list, tuple)) else ast.literal_eval(str(value))
        ids = [int(item) for item in parsed]
        # span_labels.csv.gz stores chunk-local eojeol positions as 0-based;
        # dependency spans/compressor use 1-based surface word ids.
        return [item + 1 for item in ids] if id_column == "eojeol_indices" else ids

    spans = []
    for _, row in merged.iterrows():
        spans.append({
            "span_id": row["span_id"],
            "size": int(row["size"]) if "size" in row else len(parse_ids(row[id_column])),
            "word_ids": parse_ids(row[id_column]),
            "text": row[text_column],
            "p_drop": row["drop_probability"],
            "utterance_uid": row[uid_column],
        })
    return spans


# ============================================================
# 2. size <= L 필터링
# ============================================================

def _flatten_spans(spans) -> List[Dict[str, Any]]:
    """Accept both a flat span list and generate_all_spans()'s L-keyed dict."""
    if isinstance(spans, dict):
        flattened = []
        for value in spans.values():
            if not isinstance(value, list):
                raise TypeError("span dictionary values must be lists")
            flattened.extend(value)
        return flattened
    return list(spans)


def _span_size(span: Dict[str, Any]) -> int:
    """Read size while remaining compatible with older span JSON files."""
    if "size" in span:
        return int(span["size"])
    return len(span.get("word_ids", []))


def filter_by_L(spans: List[Dict[str, Any]], L: int) -> List[Dict[str, Any]]:
    spans = _flatten_spans(spans)
    return [s for s in spans if _span_size(s) <= L]


# ============================================================
# 3. threshold 적용
# ============================================================

def apply_threshold(spans: List[Dict[str, Any]], threshold: float) -> List[Dict[str, Any]]:
    out = []
    for s in spans:
        s = dict(s)
        s["predicted_drop"] = s["p_drop"] >= threshold
        out.append(s)
    return out


# ============================================================
# 4. 포함·중첩 span 충돌 처리
#    (상위 span DROP -> 하위 span 강제 DROP)
# ============================================================

def resolve_conflicts(spans: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
    """
    큰 span부터 처리. 어떤 span이 DROP으로 확정되면, 그 span의 word_ids를
    완전히 포함하는(부분집합인) 더 작은 span들도 전부 강제 DROP 처리한다.
    """
    spans = [dict(s) for s in spans]
    spans_sorted = sorted(spans, key=lambda s: -_span_size(s))

    dropped_word_id_sets = []  # 이미 DROP 확정된 span들의 word_ids(set)

    for s in spans_sorted:
        word_id_set = set(s["word_ids"])

        forced = any(word_id_set.issubset(bigger) for bigger in dropped_word_id_sets)
        if forced:
            s["final_decision"] = "DROP"
            s["forced_by_parent"] = True
        else:
            s["final_decision"] = "DROP" if s["predicted_drop"] else "KEEP"
            s["forced_by_parent"] = False

        if s["final_decision"] == "DROP":
            dropped_word_id_sets.append(word_id_set)

    return spans_sorted


# ============================================================
# 5. 최종 DROP 어절 집합 -> 압축문 생성
# ============================================================

def compute_final_drop_word_ids(spans_resolved: List[Dict[str, Any]]) -> set:
    """resolve_conflicts()가 반환한 (final_decision 붙은) span들에서 최종 삭제 어절 집합 계산. (max 규칙 경로)"""
    drop_ids = set()
    for s in spans_resolved:
        if s["final_decision"] == "DROP":
            drop_ids.update(s["word_ids"])
    return drop_ids


def compute_word_scores(spans: List[Dict[str, Any]], mode: str = "max") -> Dict[int, float]:
    """
    어절 id -> 집계된 p_drop 점수. 어절 하나는 여러 span(크기 1~L)에 동시에
    포함될 수 있으므로, 그 span들의 p_drop을 어떻게 합칠지가 mode.

    - max  (기본값, 지금까지 쓰던 규칙): 포함하는 span 중 하나라도 DROP 확신이면 삭제.
           큰 span은 "지우는 방향"으로만 작동 -> L을 키워도 결과가 거의 안 바뀜.
    - mean: 포함하는 모든 span의 평균. 큰 span이 KEEP이면 평균을 끌어내려
            어절을 보호하는 효과가 생김 (양방향 작동).
    - min : 포함하는 모든 span이 전부 DROP이어야 삭제. 가장 강한 보호.
    """
    scores = defaultdict(list)
    for s in spans:
        for wid in s["word_ids"]:
            scores[wid].append(s["p_drop"])

    if mode == "max":
        return {w: max(v) for w, v in scores.items()}
    if mode == "mean":
        return {w: sum(v) / len(v) for w, v in scores.items()}
    if mode == "min":
        return {w: min(v) for w, v in scores.items()}
    raise ValueError(f"알 수 없는 drop_rule: {mode} (max/mean/min 중 하나여야 함)")


def compute_final_drop_word_ids_by_rule(spans: List[Dict[str, Any]], threshold: float,
                                         mode: str = "max") -> set:
    """mode에 따라 어절 단위로 직접 삭제 여부를 결정 (resolve_conflicts를 거치지 않음)."""
    scores = compute_word_scores(spans, mode)
    return {w for w, v in scores.items() if v >= threshold}


def compress_sentence(words: List[Dict[str, Any]], drop_word_ids: set) -> Dict[str, Any]:
    """
    words: [{"id":1,"text":"철수는",...}, ...] (dependency_spans 표준 형식)
    drop_word_ids: 삭제 확정된 word id 집합
    """
    kept = [w for w in words if w["id"] not in drop_word_ids]
    compressed_text = " ".join(w["text"] for w in kept)
    original_text = " ".join(w["text"] for w in words)

    original_len = len(original_text)
    compressed_len = len(compressed_text)
    character_compression_ratio = (
        1 - (compressed_len / original_len) if original_len > 0 else 0.0
    )
    eojeol_compression_ratio = (
        1 - (len(kept) / len(words)) if words else 0.0
    )

    return {
        "original": original_text,
        "compressed": compressed_text,
        "n_words_original": len(words),
        "n_words_compressed": len(kept),
        "n_words_dropped": len(words) - len(kept),
        # Keep the old key for compatibility, but expose its unit explicitly.
        "compression_ratio": character_compression_ratio,
        "character_compression_ratio": character_compression_ratio,
        "eojeol_compression_ratio": eojeol_compression_ratio,
    }


# ============================================================
# 6. 전체 파이프라인 한 번에
# ============================================================

def compress(words, spans, L: int, threshold: float, use_dummy: bool = True, seed: int = 0,
             drop_rule: str = "max"):
    """
    drop_rule="max" (기본값): 기존 span 충돌 처리 파이프라인 그대로
        (filter_by_L -> apply_threshold -> resolve_conflicts -> compute_final_drop_word_ids)
    drop_rule="mean"/"min": 어절 단위로 직접 집계 (compute_word_scores 참고).
        이 경우 개별 span의 final_decision 개념이 없어 n_spans_dropped는
        "그 span 자신의 p_drop이 threshold를 넘었는지" 기준의 참고값으로 계산.
    """
    spans = _flatten_spans(spans)
    if use_dummy:
        spans = assign_dummy_probs(spans, seed=seed)
    spans = filter_by_L(spans, L)

    if drop_rule == "max":
        spans_th = apply_threshold(spans, threshold)
        spans_resolved = resolve_conflicts(spans_th)
        drop_ids = compute_final_drop_word_ids(spans_resolved)
        n_spans_dropped = sum(1 for s in spans_resolved if s["final_decision"] == "DROP")
    else:
        drop_ids = compute_final_drop_word_ids_by_rule(spans, threshold, mode=drop_rule)
        n_spans_dropped = sum(1 for s in spans if s["p_drop"] >= threshold)

    result = compress_sentence(words, drop_ids)
    result["L"] = L
    result["threshold"] = threshold
    result["drop_rule"] = drop_rule
    result["n_spans_considered"] = len(spans)
    result["n_spans_dropped"] = n_spans_dropped
    result["n_forced_by_parent"] = (
        sum(1 for s in spans_resolved if s.get("forced_by_parent")) if drop_rule == "max" else None
    )
    return result


# ============================================================
# 데모 / 셀프테스트
# ============================================================

def _demo_with_mock_data():
    """더미 확률로 전체 흐름 시연 (인코더 없이도 실행 가능)."""
    words = [
        {"id": 1, "text": "철수는"}, {"id": 2, "text": "어제"}, {"id": 3, "text": "새로"},
        {"id": 4, "text": "산"}, {"id": 5, "text": "빨간"}, {"id": 6, "text": "자전거를"},
        {"id": 7, "text": "타고"}, {"id": 8, "text": "학교에"}, {"id": 9, "text": "갔다"},
    ]
    spans = [
        {"size": 1, "word_ids": [2], "text": "어제"},
        {"size": 1, "word_ids": [3], "text": "새로"},
        {"size": 2, "word_ids": [2, 3], "text": "어제 새로"},
        {"size": 4, "word_ids": [3, 4, 5, 6], "text": "새로 산 빨간 자전거를"},
    ]

    print("=== 더미 확률 데모 ===")
    result = compress(words, spans, L=4, threshold=0.5, use_dummy=True, seed=1)
    for k, v in result.items():
        print(f"  {k}: {v}")


def _demo_with_real_labels(csv_path: str, utterance_uid: str):
    """실제 세연님 라벨(auto_label)을 p_drop=1.0/0.0으로 취급해 실제 데이터로 검증."""
    import pandas as pd
    import ast

    df = pd.read_csv(csv_path, compression="gzip" if csv_path.endswith(".gz") else None)
    sub = df[df["utterance_uid"] == utterance_uid].copy()
    if len(sub) == 0:
        print(f"utterance_uid '{utterance_uid}' 를 찾을 수 없음")
        return

    spans = []
    for _, row in sub.iterrows():
        word_ids = ast.literal_eval(row["word_ids"])
        p_drop = 1.0 if row["auto_label"] == "DROP" else 0.0
        spans.append({"size": row["size"], "word_ids": word_ids, "text": row["span"], "p_drop": p_drop})

    # 원문 words 목록은 size=1 span들에서 역으로 복원 (word_id -> text)
    id_to_text = {}
    for s in spans:
        if s["size"] == 1:
            id_to_text[s["word_ids"][0]] = s["text"]
    words = [{"id": i, "text": id_to_text[i]} for i in sorted(id_to_text)]

    print(f"=== 실제 라벨 데모: {utterance_uid} ===")
    print(f"원문: {' '.join(w['text'] for w in words)}")

    result = compress(words, spans, L=8, threshold=0.5, use_dummy=False)
    for k, v in result.items():
        print(f"  {k}: {v}")


def self_test():
    """resolve_conflicts의 핵심 규칙(상위 DROP -> 하위 강제 DROP)을 검증."""
    words = [{"id": i, "text": f"w{i}"} for i in range(1, 5)]
    spans = [
        {"size": 1, "word_ids": [1], "p_drop": 0.0},  # 개별로는 KEEP 확률
        {"size": 1, "word_ids": [2], "p_drop": 0.0},
        {"size": 2, "word_ids": [1, 2], "p_drop": 0.9},  # 상위 span은 DROP
    ]
    spans = apply_threshold(spans, threshold=0.5)
    resolved = resolve_conflicts(spans)

    small_spans = [s for s in resolved if s["size"] == 1]
    assert all(s["final_decision"] == "DROP" for s in small_spans), (
        "[self_test 실패] 상위 span이 DROP인데 하위 span이 강제 DROP되지 않음"
    )
    assert all(s["forced_by_parent"] for s in small_spans), (
        "[self_test 실패] forced_by_parent 플래그가 정상적으로 표시되지 않음"
    )

    drop_ids = compute_final_drop_word_ids(resolved)
    assert drop_ids == {1, 2}, f"[self_test 실패] 최종 DROP id 집합이 예상과 다름: {drop_ids}"

    return True


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--demo", action="store_true")
    ap.add_argument("--real-labels", help="span_labels.csv.gz 경로")
    ap.add_argument("--utterance-uid", help="--real-labels와 함께 사용")
    args = ap.parse_args()

    self_test()
    # Windows PowerShell의 기본 cp949에서도 깨지지 않도록 ASCII만 사용한다.
    print("self-test passed (conflict resolution is working)\n")

    if args.demo:
        _demo_with_mock_data()
        if args.real_labels and args.utterance_uid:
            print()
            _demo_with_real_labels(args.real_labels, args.utterance_uid)
