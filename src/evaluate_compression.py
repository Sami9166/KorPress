"""
evaluate_compression.py

압축 전/후 문장 쌍을 받아서 다슬님 실험 계획서의 측정 항목 일부를 자동 계산한다.
인코더 checkpoint 여부와 무관하게 지금 바로 개발/검증 가능 (compressor.py의
더미 확률 결과로도 테스트 가능, 실제 checkpoint 나오면 그 결과에 그대로 적용).

측정 항목:
- compression_ratio: 이미 compressor.py에 있음 (여기선 재사용만)
- 수치(NUM)/고유명사(PROPN) 보존율: 원문 단어의 upos 태그 기반 (Stanza가 이미 붙여줌)
- 날짜 보존율: 정규식 기반 (숫자+년/월/일 패턴)
- 부정 표현 보존율: 어절 안에 부정 형태소(안/못/않/없/아니) 포함 여부 기반
- 문법적 완전성: 압축문을 다시 dependency parsing해서 root가 정상적으로
  잡히는지 확인 (Stanza 모델이 있으면 정식 파서, 없으면 오프라인 휴리스틱)

사용법:
    python evaluate_compression.py --demo
"""

import argparse
import re
from typing import List, Dict, Any


NEGATION_MARKERS = ["안", "못", "않", "없", "아니"]
DATE_PATTERN = re.compile(r"\d+\s*(년|월|일|시|분|초)")
NUMBER_PATTERN = re.compile(r"\d+")


# ============================================================
# 1. 원문에서 "보존되어야 할 것들" 추출
# ============================================================

def extract_preservation_targets(words: List[Dict[str, Any]]) -> Dict[str, List[str]]:
    """
    원문 words(dependency_parser 표준 형식, upos 있으면 활용)에서
    카테고리별로 "압축 후에도 남아있어야 바람직한" 단어들을 추출.
    """
    targets = {"number": [], "date": [], "negation": [], "proper_noun": []}

    for w in words:
        text = w["text"]
        upos = w.get("upos", "")

        if upos == "NUM" or NUMBER_PATTERN.search(text):
            targets["number"].append(text)
        if DATE_PATTERN.search(text):
            targets["date"].append(text)
        if upos == "PROPN":
            targets["proper_noun"].append(text)
        if any(marker in text for marker in NEGATION_MARKERS):
            targets["negation"].append(text)

    return targets


# ============================================================
# 2. 보존율 계산
# ============================================================

def compute_preservation_rates(targets: Dict[str, List[str]], compressed_text: str) -> Dict[str, Any]:
    """
    카테고리별로 "원문에 있던 항목이 압축문에도 그대로 남아있는 비율"을 계산.
    항목이 원래 0개인 카테고리는 None(해당 없음)으로 표시.
    """
    rates = {}
    for category, items in targets.items():
        if not items:
            rates[f"{category}_preservation_rate"] = None
            continue
        survived = sum(1 for item in items if item in compressed_text)
        rates[f"{category}_preservation_rate"] = survived / len(items)
    return rates


# ============================================================
# 3. 문법적 완전성
# ============================================================

def check_grammatical_completeness_offline(compressed_text: str) -> Dict[str, Any]:
    """
    실제 재파싱 없이 쓸 수 있는 가벼운 휴리스틱 체크.
    (정식 체크는 check_grammatical_completeness_with_parser 사용)
    """
    text = compressed_text.strip()
    if not text:
        return {"is_empty": True, "ends_with_sentence_final": False, "likely_complete": False}

    # 한국어 문장 종결 어미로 흔히 끝나는 패턴 (완벽하진 않지만 1차 필터로 유용)
    sentence_final_pattern = re.compile(r"(다|요|까|네|죠|음|함)\.?$")
    ends_ok = bool(sentence_final_pattern.search(text))

    return {
        "is_empty": False,
        "ends_with_sentence_final": ends_ok,
        "likely_complete": ends_ok,
    }


def check_grammatical_completeness_with_parser(compressed_text: str, parser) -> Dict[str, Any]:
    """
    KoreanDependencyParser로 압축문을 재파싱해서 정식으로 문법적 완전성을 확인.
    """
    result = parser.parse_sentence(compressed_text)
    words = result["words"]

    if not words:
        return {"is_empty": True, "has_root": False, "root_is_predicate": False, "likely_complete": False}

    roots = [w for w in words if w["deprel"] == "root"]
    has_root = len(roots) >= 1
    root_is_predicate = has_root and roots[0].get("upos") in ("VERB", "ADJ") and not roots[0].get("is_punct", False)

    return {
        "is_empty": False,
        "has_root": has_root,
        "root_is_predicate": root_is_predicate,
        "likely_complete": has_root and root_is_predicate,
    }


# ============================================================
# 4. 하나의 압축 결과에 대해 전체 지표 한 번에
# ============================================================

def evaluate_single(original_words: List[Dict[str, Any]], compressed_text: str,
                     parser=None) -> Dict[str, Any]:
    targets = extract_preservation_targets(original_words)
    metrics = compute_preservation_rates(targets, compressed_text)

    if parser is not None:
        metrics["grammar"] = check_grammatical_completeness_with_parser(compressed_text, parser)
    else:
        metrics["grammar"] = check_grammatical_completeness_offline(compressed_text)

    return metrics


def evaluate_batch(compress_results: List[Dict[str, Any]], original_words_list: List[List[Dict]],
                    parser=None) -> Dict[str, Any]:
    """
    여러 압축 결과에 대해 평가하고, 카테고리별 평균 보존율까지 집계.
    compress_results: compressor.compress()가 반환한 딕셔너리들의 리스트
    """
    all_metrics = []
    for result, orig_words in zip(compress_results, original_words_list):
        m = evaluate_single(orig_words, result["compressed"], parser=parser)
        all_metrics.append(m)

    summary = {}
    for key in ["number_preservation_rate", "date_preservation_rate",
                "negation_preservation_rate", "proper_noun_preservation_rate"]:
        values = [m[key] for m in all_metrics if m[key] is not None]
        summary[f"avg_{key}"] = sum(values) / len(values) if values else None

    n_complete = sum(1 for m in all_metrics if m["grammar"]["likely_complete"])
    summary["grammatical_completeness_rate"] = n_complete / len(all_metrics) if all_metrics else None
    summary["n_evaluated"] = len(all_metrics)

    return {"per_sentence": all_metrics, "summary": summary}


# ============================================================
# 데모 / 셀프테스트
# ============================================================

def self_test():
    # 1) 보존율 계산 검증
    words = [
        {"id": 1, "text": "2024년", "upos": "NUM"},
        {"id": 2, "text": "철수는", "upos": "PROPN"},
        {"id": 3, "text": "약속을", "upos": "NOUN"},
        {"id": 4, "text": "안", "upos": "ADV"},
        {"id": 5, "text": "지켰다", "upos": "VERB"},
    ]
    targets = extract_preservation_targets(words)
    assert "2024년" in targets["number"], "숫자 추출 실패"
    assert "2024년" in targets["date"], "날짜 추출 실패"
    assert "철수는" in targets["proper_noun"], "고유명사 추출 실패"
    assert "안" in targets["negation"], "부정 표현 추출 실패"

    # 압축문에서 "안"이 사라진 경우 -> negation_preservation_rate가 0이어야 함
    compressed_dropped_negation = "2024년 철수는 약속을 지켰다"
    rates = compute_preservation_rates(targets, compressed_dropped_negation)
    assert rates["negation_preservation_rate"] == 0.0, (
        f"[self_test 실패] 부정 표현이 빠졌는데도 보존율이 0이 아님: {rates['negation_preservation_rate']}"
    )
    assert rates["proper_noun_preservation_rate"] == 1.0, "고유명사가 남았는데 보존율 계산 오류"

    # 2) 문법 완전성 오프라인 휴리스틱 검증
    complete = check_grammatical_completeness_offline("학교에 갔다.")
    incomplete = check_grammatical_completeness_offline("학교에 그리고")
    empty = check_grammatical_completeness_offline("")
    assert complete["likely_complete"] is True, "정상 문장을 불완전하다고 판정함"
    assert incomplete["likely_complete"] is False, "불완전 문장을 완전하다고 판정함"
    assert empty["is_empty"] is True, "빈 문장 처리 실패"

    return True


def _demo():
    original_words = [
        {"id": 1, "text": "2024년", "upos": "NUM"},
        {"id": 2, "text": "3월에", "upos": "NOUN"},
        {"id": 3, "text": "철수는"},
        {"id": 4, "text": "약속을"},
        {"id": 5, "text": "절대"},
        {"id": 6, "text": "안"},
        {"id": 7, "text": "어겼다."},
    ]
    # 부정 표현("안")이 실수로 삭제된 나쁜 압축 예시
    bad_compressed = "2024년 3월에 철수는 약속을 어겼다."
    # 부정 표현이 살아남은 정상적인 압축 예시
    good_compressed = "3월에 철수는 절대 안 어겼다."

    print("=== 나쁜 압축 (부정어 실수로 삭제 - 의미가 반대로 바뀜) ===")
    print(evaluate_single(original_words, bad_compressed))

    print("\n=== 정상적인 압축 (부정어 보존) ===")
    print(evaluate_single(original_words, good_compressed))


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--demo", action="store_true")
    args = ap.parse_args()

    self_test()
    print("✅ self-test 통과\n")

    if args.demo:
        _demo()
