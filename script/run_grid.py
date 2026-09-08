"""
run_grid.py

L × threshold 그리드 전체를 돌려서 조합별 압축 결과와 지표를 저장한다.

입력:
- span_labels.csv.gz     : 세연님 라벨 데이터 (span_id, word_ids, size, span, utterance_uid ...)
- test_predictions.csv.gz: 다슬님 predict.py 출력 (span_id, drop_probability)

출력 (--out-dir 아래):
- results_L{L}_t{tau}.jsonl : 조합별 발화 단위 압축 결과 + 지표 (이미 있으면 건너뜀 → 재개 가능)
- summary.csv               : 조합별 집계표 (압축률, 빈 문장 비율, 보존율, 문법 완전성, 속도)
- scoring_sample.csv        : 의미 보존 채점용 샘플 (모든 조합에 같은 발화 N개, 비교 가능하도록)

결정사항 반영:
- L ∈ {1,2,4,8}, threshold ∈ {0.5,0.7,0.9} 기본값 (12조합)
- 발화 전체 삭제는 막지 않음(allow_full_drop). 대신 빈 문장 비율을 지표로 출력.
- --drop-rule max(기본값)/mean/min : 어절이 여러 span에 동시에 포함될 때 그
  span들의 p_drop을 어떻게 합칠지. max는 "큰 span은 지우는 방향으로만 작동"해서
  L을 키워도 결과가 거의 안 바뀌는 특성이 있음(검증됨). mean/min은 "상위 span이
  KEEP이면 하위 어절을 보호"하는 효과가 있어 L이 실제로 결과를 바꾸지만, 정답
  라벨 기준 F1은 낮아지는 트레이드오프가 있음.

문법 완전성:
- 기본은 오프라인 휴리스틱(문장 종결 어미 패턴).
- --grammar-parser 를 주면 dependency_spans.KoreanDependencyParser 로 압축문을 재파싱해
  root가 서술어로 잡히는지 정식 확인 (Stanza 모델 필요). 이때 원문도 파싱해서
  upos 를 얻으므로 고유명사(PROPN) 보존율도 계산된다.

사용법:
    python run_grid.py --span-labels span_labels.csv.gz --predictions test_predictions.csv.gz --out-dir grid_results
    python run_grid.py ... --grammar-parser          # 정식 문법 체크 + 고유명사 보존율 (느림)
    python run_grid.py ... --drop-rule mean          # 보호 규칙 버전으로 별도 실행 (max 결과와 안 겹침)
"""

import argparse
import csv
import json
import random
import time
from collections import defaultdict
from pathlib import Path
import sys

SRC_DIR = Path(__file__).resolve().parents[1] / "src"
if str(SRC_DIR) not in sys.path:
    sys.path.insert(0, str(SRC_DIR))

from compressor import (
    load_real_predictions, filter_by_L, apply_threshold,
    resolve_conflicts, compute_final_drop_word_ids,
    compute_final_drop_word_ids_by_rule, compress_sentence,
)
from evaluate_compression import evaluate_single


# ------------------------------------------------------------
# 데이터 준비
# ------------------------------------------------------------

def group_by_utterance(spans):
    by_utt = defaultdict(list)
    for s in spans:
        by_utt[s["utterance_uid"]].append(s)
    return by_utt


def reconstruct_words(spans):
    """size=1 span들로부터 원문 어절 목록 복원 (upos 없음)."""
    id_to_text = {}
    for s in spans:
        if s["size"] == 1:
            id_to_text[s["word_ids"][0]] = s["text"]
    return [{"id": i, "text": id_to_text[i]} for i in sorted(id_to_text)]


def words_with_parser(parser, original_text, fallback_words):
    """
    원문을 파서로 다시 파싱해 upos 가 붙은 words 를 만든다.
    토큰 수가 span 데이터와 다르면(드물게) fallback 사용 — word_ids 정합성이 우선.
    """
    try:
        parsed = parser.parse_sentence(original_text)["words"]
    except Exception:
        return fallback_words
    if len(parsed) != len(fallback_words):
        return fallback_words
    merged = []
    for fw, pw in zip(fallback_words, parsed):
        merged.append({"id": fw["id"], "text": fw["text"], "upos": pw.get("upos", "")})
    return merged


# ------------------------------------------------------------
# 조합 하나 실행
# ------------------------------------------------------------

def run_combo(by_utt, words_cache, L, tau, parser=None, drop_rule="max"):
    records = []
    t0 = time.time()

    for uid, spans in by_utt.items():
        words = words_cache[uid]

        filtered = filter_by_L(spans, L)
        if drop_rule == "max":
            th = apply_threshold(filtered, tau)
            resolved = resolve_conflicts(th)
            drop_ids = compute_final_drop_word_ids(resolved)
            n_spans_dropped = sum(1 for s in resolved if s["final_decision"] == "DROP")
        else:
            drop_ids = compute_final_drop_word_ids_by_rule(filtered, tau, mode=drop_rule)
            n_spans_dropped = sum(1 for s in filtered if s["p_drop"] >= tau)
        result = compress_sentence(words, drop_ids)

        metrics = evaluate_single(words, result["compressed"], parser=parser)
        is_empty = result["compressed"].strip() == ""

        records.append({
            "utterance_uid": uid,
            "L": L,
            "threshold": tau,
            "original": result["original"],
            "compressed": result["compressed"],
            "n_words_original": result["n_words_original"],
            "n_words_dropped": result["n_words_dropped"],
            "word_drop_ratio": result["n_words_dropped"] / max(result["n_words_original"], 1),
            "char_compression_ratio": result["compression_ratio"],
            "is_empty": is_empty,
            "n_spans_considered": len(filtered),
            "n_spans_dropped": n_spans_dropped,
            "number_preservation_rate": metrics["number_preservation_rate"],
            "date_preservation_rate": metrics["date_preservation_rate"],
            "negation_preservation_rate": metrics["negation_preservation_rate"],
            "proper_noun_preservation_rate": metrics["proper_noun_preservation_rate"],
            "grammar_likely_complete": bool(metrics["grammar"]["likely_complete"]),
        })

    elapsed = time.time() - t0
    return records, elapsed


# ------------------------------------------------------------
# 집계
# ------------------------------------------------------------

def _mean(values):
    values = [v for v in values if v is not None]
    return sum(values) / len(values) if values else None


def summarize(records, elapsed, L, tau):
    n = len(records)
    non_empty = [r for r in records if not r["is_empty"]]
    return {
        "L": L,
        "threshold": tau,
        "n_utterances": n,
        "avg_char_compression_ratio": _mean([r["char_compression_ratio"] for r in records]),
        "avg_word_drop_ratio": _mean([r["word_drop_ratio"] for r in records]),
        "empty_rate": sum(1 for r in records if r["is_empty"]) / n if n else None,
        "avg_number_preservation": _mean([r["number_preservation_rate"] for r in records]),
        "avg_date_preservation": _mean([r["date_preservation_rate"] for r in records]),
        "avg_negation_preservation": _mean([r["negation_preservation_rate"] for r in records]),
        "avg_proper_noun_preservation": _mean([r["proper_noun_preservation_rate"] for r in records]),
        "grammar_complete_rate_all": _mean([r["grammar_likely_complete"] for r in records]),
        "grammar_complete_rate_nonempty": _mean([r["grammar_likely_complete"] for r in non_empty]),
        "compress_seconds_total": elapsed,
        "compress_ms_per_utterance": elapsed / n * 1000 if n else None,
    }


# ------------------------------------------------------------
# 메인
# ------------------------------------------------------------

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--span-labels", required=True)
    ap.add_argument("--predictions", required=True)
    ap.add_argument("--out-dir", default="grid_results")
    ap.add_argument("--L", nargs="+", type=int, default=[1, 2, 4, 8])
    ap.add_argument("--thresholds", nargs="+", type=float, default=[0.5, 0.7, 0.9])
    ap.add_argument("--grammar-parser", action="store_true",
                     help="dependency_spans 로 압축문 재파싱 (정식 문법 체크 + 고유명사 보존율)")
    ap.add_argument("--drop-rule", choices=["max", "mean", "min"], default="max",
                     help="max(기본값)=지금까지 쓰던 규칙(L에 둔감). "
                          "mean/min=상위 span KEEP이 하위 어절을 보호하는 규칙(L이 실제로 영향을 줌).")
    ap.add_argument("--gpu", action="store_true", help="--grammar-parser 사용 시 Stanza를 GPU로 실행")
    ap.add_argument("--sample-size", type=int, default=200, help="의미 보존 채점용 샘플 발화 수")
    ap.add_argument("--seed", type=int, default=42)
    args = ap.parse_args()

    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    print("예측 결과 로드 중...")
    spans = load_real_predictions(args.span_labels, args.predictions)
    by_utt = group_by_utterance(spans)
    print(f"발화 {len(by_utt)}개, span {len(spans)}개")

    parser = None
    if args.grammar_parser:
        from dependency_spans import KoreanDependencyParser
        parser = KoreanDependencyParser(use_gpu=args.gpu)
        print("정식 문법 체크 모드 (파서 로드 완료)")

    # 원문 words 캐시 (조합마다 다시 만들 필요 없음)
    words_cache = {}
    for uid, s in by_utt.items():
        base = reconstruct_words(s)
        if parser is not None:
            base = words_with_parser(parser, " ".join(w["text"] for w in base), base)
        words_cache[uid] = base

    summaries = []
    for L in args.L:
        for tau in args.thresholds:
            path = out_dir / f"results_L{L}_t{tau}_r{args.drop_rule}.jsonl"
            if path.exists():
                print(f"[건너뜀] {path.name} (이미 존재)")
                with open(path, encoding="utf-8") as f:
                    records = [json.loads(line) for line in f]
                elapsed = float("nan")
            else:
                print(f"[실행] L={L}, threshold={tau}, rule={args.drop_rule} ...", end=" ", flush=True)
                records, elapsed = run_combo(by_utt, words_cache, L, tau, parser=parser, drop_rule=args.drop_rule)
                with open(path, "w", encoding="utf-8") as f:
                    for r in records:
                        f.write(json.dumps(r, ensure_ascii=False) + "\n")
                print(f"{elapsed:.1f}초")
            s = summarize(records, elapsed, L, tau)
            s["drop_rule"] = args.drop_rule
            summaries.append(s)

    # summary.csv (규칙별로 파일 분리 - max 결과 덮어쓰지 않음)
    summary_path = out_dir / f"summary_r{args.drop_rule}.csv"
    with open(summary_path, "w", encoding="utf-8", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=list(summaries[0].keys()))
        writer.writeheader()
        writer.writerows(summaries)

    # 채점용 샘플: 모든 조합에 같은 발화 집합
    rng = random.Random(args.seed)
    sample_uids = set(rng.sample(sorted(by_utt.keys()), min(args.sample_size, len(by_utt))))
    sample_path = out_dir / f"scoring_sample_r{args.drop_rule}.csv"
    with open(sample_path, "w", encoding="utf-8", newline="") as f:
        writer = csv.writer(f)
        writer.writerow(["L", "threshold", "drop_rule", "utterance_uid", "original", "compressed", "is_empty"])
        for L in args.L:
            for tau in args.thresholds:
                with open(out_dir / f"results_L{L}_t{tau}_r{args.drop_rule}.jsonl", encoding="utf-8") as rf:
                    for line in rf:
                        r = json.loads(line)
                        if r["utterance_uid"] in sample_uids:
                            writer.writerow([L, tau, args.drop_rule, r["utterance_uid"], r["original"], r["compressed"], r["is_empty"]])

    print(f"\n완료: {summary_path}, {sample_path}")
    print("\n=== 요약 ===")
    for s in summaries:
        print(f"L={s['L']} τ={s['threshold']}: 압축률 {s['avg_char_compression_ratio']*100:5.1f}% | "
              f"빈문장 {s['empty_rate']*100:5.1f}% | 부정어보존 {(_fmt(s['avg_negation_preservation']))} | "
              f"문법완전(비어있지않은것) {(_fmt(s['grammar_complete_rate_nonempty']))}")


def _fmt(v):
    return f"{v*100:5.1f}%" if v is not None else "  n/a"


if __name__ == "__main__":
    main()
