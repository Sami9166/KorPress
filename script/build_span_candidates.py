"""Build dependency-span candidates from one-sentence-per-line input.

전체 파이프라인: 문장들 -> dependency parsing -> L값별 span 후보 -> JSON 저장

사용법 (최초 1회 한국어 Stanza 모델 다운로드 필요):
    python script/build_span_candidates.py --input sentences.txt --output result.json

sentences.txt: 한 줄에 문장 하나씩
"""

import argparse
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
        })
    return results


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--input", required=True, help="문장이 한 줄씩 들어있는 txt 파일")
    ap.add_argument("--output", required=True, help="결과 JSON 저장 경로")
    ap.add_argument("--L", nargs="+", type=int, default=[1, 2, 4, 8], help="생성할 L값들")
    ap.add_argument("--gpu", action="store_true", help="GPU 사용 여부")
    ap.add_argument("--download", action="store_true", help="최초 실행 시 모델 다운로드")
    args = ap.parse_args()

    if args.download:
        download_korean_model()

    with open(args.input, "r", encoding="utf-8") as f:
        sentences = [line.strip() for line in f if line.strip()]

    results = run_pipeline(sentences, L_values=tuple(args.L), use_gpu=args.gpu)

    with open(args.output, "w", encoding="utf-8") as f:
        json.dump(results, f, ensure_ascii=False, indent=2)

    print(f"{len(sentences)}개 문장 처리 완료 -> {args.output}")


if __name__ == "__main__":
    main()
