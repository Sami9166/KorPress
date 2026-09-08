"""Build dependency-span candidates from one-sentence-per-line input.

전체 파이프라인: 문장들 -> dependency parsing -> L값별 span 후보 -> JSON 저장

사용법 (최초 1회 한국어 Stanza 모델 다운로드 필요):
    python script/build_span_candidates.py --input sentences.txt --output result.json

sentences.txt: 한 줄에 문장 하나씩
"""

import argparse
import csv
import json
import sys
from pathlib import Path
from typing import List

SRC_DIR = Path(__file__).resolve().parents[1] / "src"
if str(SRC_DIR) not in sys.path:
    sys.path.insert(0, str(SRC_DIR))

from dependency_spans import KoreanDependencyParser, download_korean_model, generate_all_spans


def run_pipeline(sentences: List[str], L_values=(1, 2, 4, 8), use_gpu: bool = False):
    parser = KoreanDependencyParser(use_gpu=use_gpu)
    results = []
    for sent in sentences:
        parsed = parser.parse_sentence(sent)
        spans = generate_all_spans(parsed["words"], L_values=L_values)
        results.append({
            "sentence": parsed["sentence"],
            "words": parsed["words"],
            "spans": spans,
            "multi_sentence_detected": parsed.get("multi_sentence_detected", False),
            "num_stanza_sentences": parsed.get("num_stanza_sentences", 1),
        })
    return results


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--input", help="문장이 한 줄씩 들어있는 txt 파일")
    ap.add_argument("--output", help="결과 JSON 저장 경로")
    ap.add_argument("--L", nargs="+", type=int, default=[1, 2, 4, 8], help="생성할 L값들")
    ap.add_argument("--gpu", action="store_true", help="GPU 사용 여부")
    ap.add_argument("--download", action="store_true", help="최초 실행 시 모델 다운로드")
    ap.add_argument(
        "--metadata",
        help="prepare_aihub_input.py가 만든 metadata.csv. 입력 문장 순서와 일치해야 합니다.",
    )
    ap.add_argument(
        "--selftest",
        action="store_true",
        help="Stanza를 실행하지 않고 parser/span 유틸리티 회귀 테스트만 수행",
    )
    args = ap.parse_args()

    if args.selftest:
        from dependency_spans import self_test

        self_test()
        print("span candidate self-test passed")
        return

    if not args.input or not args.output:
        ap.error("--input과 --output은 --selftest를 쓰지 않을 때 필수입니다.")

    if args.download:
        download_korean_model()

    with open(args.input, "r", encoding="utf-8") as f:
        sentences = [line.strip() for line in f if line.strip()]

    results = run_pipeline(sentences, L_values=tuple(args.L), use_gpu=args.gpu)

    if args.metadata:
        with open(args.metadata, "r", encoding="utf-8-sig", newline="") as f:
            metadata = list(csv.DictReader(f))
        if len(metadata) != len(results):
            raise ValueError(
                f"metadata 행 수({len(metadata)})와 입력 문장 수({len(results)})가 다릅니다."
            )
        for result, row in zip(results, metadata):
            result["metadata"] = row

    with open(args.output, "w", encoding="utf-8") as f:
        json.dump(results, f, ensure_ascii=False, indent=2)

    print(f"{len(sentences)}개 문장 처리 완료 -> {args.output}")


if __name__ == "__main__":
    main()
