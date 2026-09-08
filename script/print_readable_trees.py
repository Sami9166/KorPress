"""
print_readable_trees.py

sample_result.json(또는 다른 결과 JSON)을 사람이 눈으로 훑어보기 쉬운
들여쓰기 트리 텍스트로 출력한다. raw JSON을 직접 읽는 것보다 훨씬 빠르게
100개 샘플을 검증할 수 있다.

또한 "쉼표 없는 나열형 문장"처럼 구조가 의심스러운 케이스를 자동으로
표시해서, 어느 문장부터 봐야 할지 우선순위를 잡아준다.

사용법:
    python print_readable_trees.py --input sample_result.json
    python print_readable_trees.py --input sample_result.json --only-suspicious
    python print_readable_trees.py --input sample_result.json --limit 20
"""

import argparse
import json


def build_children_map(words):
    children = {w["id"]: [] for w in words}
    children[0] = []  # root의 부모
    for w in words:
        children.setdefault(w["head"], []).append(w["id"])
    return children


def print_tree(words, node_id, children, id_to_word, depth=0):
    if node_id != 0:
        w = id_to_word[node_id]
        indent = "  " * depth
        print(f"{indent}{'└─ ' if depth > 0 else ''}{w['text']} [{w['deprel']}]")
    for child_id in sorted(children.get(node_id, [])):
        print_tree(words, child_id, children, id_to_word, depth + 1)


def looks_suspicious(words):
    """
    쉼표 없는 나열형처럼 구조가 의심스러운 문장을 대략적으로 탐지.
    - "compound" 관계가 비정상적으로 길게 연쇄되는 경우
    - 문장 안에 "삼국의"류 같은 "N의"가 3번 이상 반복되는 경우 (나열 신호)
    """
    compound_count = sum(1 for w in words if w["deprel"] == "compound")
    nmod_count = sum(1 for w in words if w["deprel"] == "nmod")
    if compound_count >= 4:
        return True
    if nmod_count >= 4:
        return True
    return False


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--input", required=True)
    ap.add_argument("--limit", type=int, default=None, help="앞에서부터 몇 개만 볼지")
    ap.add_argument("--only-suspicious", action="store_true", help="의심스러운 문장만 출력")
    args = ap.parse_args()

    with open(args.input, "r", encoding="utf-8") as f:
        results = json.load(f)

    if args.limit:
        results = results[:args.limit]

    n_suspicious = 0
    for i, r in enumerate(results):
        words = r["words"]
        suspicious = looks_suspicious(words)
        if suspicious:
            n_suspicious += 1
        if args.only_suspicious and not suspicious:
            continue

        id_to_word = {w["id"]: w for w in words}
        children = build_children_map(words)
        root_id = next((w["id"] for w in words if w["deprel"] == "root"), None)

        tag = " ⚠️ 의심스러운 구조(나열형 가능성)" if suspicious else ""
        print(f"\n{'='*70}")
        print(f"[{i}] {r['sentence']}{tag}")
        print('='*70)
        if root_id:
            print_tree(words, 0, children, id_to_word)
        else:
            print("(root를 못 찾음 — 별도 확인 필요)")

    print(f"\n\n총 {len(results)}개 중 구조 의심 {n_suspicious}개 ({n_suspicious/len(results)*100:.1f}%)")


if __name__ == "__main__":
    main()
