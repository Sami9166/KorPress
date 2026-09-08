"""
check_integrity.py

결과 JSON(sample_result.json 또는 전체 span_results.json)에 대해
"항상 참이어야 하는 구조적 조건"들을 전수 검사한다.

사람이 눈으로 28,727개를 다 훑는 건 불가능하지만, 이 조건들은 코드로
전수 검사가 가능하다 — "버그가 없다는 증명"은 아니지만, "알려진 종류의
구조적 오류는 전혀 없다"는 훨씬 강한 형태의 보장을 준다.

검사 항목:
1. 문장에 단어가 하나도 없는 경우
2. word id 중복
3. root가 정확히 1개 이상 있는지 (여러 문장 병합 시 여러 개 있는 건 정상)
4. root가 문장부호로 잡힌 경우 (지난번 발견된 버그의 회귀 검사)
5. head가 실제 존재하지 않는 id를 가리키는 경우 (참조 무결성 깨짐)
6. 자기 자신을 head로 가리키는 경우 (즉시 순환 참조)
7. span의 word_ids가 실제 words에 없는 id를 포함하는 경우
8. 같은 L값 안에서 (size, word_ids) 조합이 중복되는 경우

사용법:
    python check_integrity.py --input sample_result.json
    python check_integrity.py --input span_results.json --sample-print 5
"""

import argparse
import json
from collections import defaultdict


def check_one(r):
    """문장 하나(결과 딕셔너리 하나)에 대한 이슈 리스트 반환."""
    issues = []
    words = r.get("words", [])

    if not words:
        issues.append("빈 문장(단어 없음)")
        return issues

    ids = [w["id"] for w in words]
    id_set = set(ids)

    if len(ids) != len(id_set):
        issues.append("중복된 word id 존재")

    roots = [w for w in words if w["deprel"] == "root"]
    if len(roots) == 0:
        issues.append("root가 하나도 없음")

    for root in roots:
        if root.get("is_punct"):
            issues.append(f"root가 문장부호: '{root['text']}'")

    for w in words:
        if w["head"] != 0 and w["head"] not in id_set:
            issues.append(f"'{w['text']}'(id={w['id']})의 head({w['head']})가 존재하지 않는 id를 가리킴")
        if w["head"] == w["id"]:
            issues.append(f"'{w['text']}'(id={w['id']})가 자기 자신을 head로 가리킴(순환)")

    for L, spans in r.get("spans", {}).items():
        seen = set()
        for s in spans:
            key = (s["size"], tuple(s["word_ids"]))
            if key in seen:
                issues.append(f"L={L}에서 중복 span: {key}")
            seen.add(key)
            for wid in s["word_ids"]:
                if wid not in id_set:
                    issues.append(f"L={L} span의 word_id({wid})가 실제 단어 목록에 없음")

    return issues


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--input", required=True)
    ap.add_argument("--sample-print", type=int, default=10, help="이슈 있는 문장 중 몇 개까지 출력할지")
    args = ap.parse_args()

    with open(args.input, "r", encoding="utf-8") as f:
        results = json.load(f)

    issue_counts = defaultdict(int)
    problem_sentences = []

    for i, r in enumerate(results):
        issues = check_one(r)
        if issues:
            problem_sentences.append((i, r.get("sentence", "")[:60], issues))
            for issue in issues:
                # 앞부분 패턴만 뽑아서 카테고리별로 집계 (구체적 텍스트는 다양하므로)
                category = issue.split(":")[0].split("(")[0].strip()
                issue_counts[category] += 1

    print(f"전체 {len(results)}개 문장 검사 완료")
    print(f"이슈 있는 문장: {len(problem_sentences)}개 ({len(problem_sentences)/len(results)*100:.2f}%)")

    if issue_counts:
        print("\n=== 이슈 카테고리별 집계 ===")
        for cat, count in sorted(issue_counts.items(), key=lambda x: -x[1]):
            print(f"  {cat}: {count}건")

        print(f"\n=== 이슈 있는 문장 샘플 (최대 {args.sample_print}개) ===")
        for i, sent, issues in problem_sentences[:args.sample_print]:
            print(f"\n[{i}] {sent}")
            for issue in issues:
                print(f"    - {issue}")
    else:
        print("\n✅ 구조적 이슈 전혀 없음 (전수 검사 기준)")


if __name__ == "__main__":
    main()
