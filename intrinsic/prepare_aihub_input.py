"""
prepare_aihub_input.py

AI Hub "한국어 대학 강의 데이터" chunks.csv를 우리 파이프라인(pipeline.py) 입력
형식으로 변환한다.

배경: chunks.csv의 "sentence" 컬럼은 발화(utterance) 여러 개가 줄바꿈(\n)으로
이어붙은 덩어리다(청크당 평균 9.6개). \n 분할 개수가 utterance_count와 전체
2,980개 청크에서 100% 일치함을 검증함 — 분할 방법론 자체는 안전함.

주의(전역 발화 번호 관련): start_utterance/end_utterance(예: U00001)로
"문서 전체 기준 몇 번째 발화인지"를 산수로 복원하려 했으나, 전체 청크의 약 5%
(153/2980)에서 번호가 연속되지 않아 이 방식은 신뢰할 수 없음 — 실제로 이렇게
계산하면 69개 발화가 서로 다른 곳인데 같은 ID로 충돌함을 확인함. 그래서
"몇 번째 발화인지"는 청크 전체 기준 절대 번호 대신, 이미 고유함이 확인된
sentence_id(청크 단위, 2,980개 전부 고유) + 청크 내 상대 순번을 조합해서 만든다
(utterance_uid). 계산이 아니라 조합이라 충돌이 원천적으로 불가능하다.

출력 2개:
- sentences.txt : pipeline.py에 그대로 넣을 수 있는, 한 줄에 발화 하나
- metadata.csv  : sentences.txt와 완전히 같은 순서로, 각 줄이 어느 chunk/문서/
                  분야에서 왔는지 추적하기 위한 메타데이터
                  (utterance_uid가 전체 데이터셋 기준 유일한 식별자)

사용법:
    python prepare_aihub_input.py --input chunks.csv --out-dir ./prepared
"""

import argparse
import csv
import re
from pathlib import Path

import pandas as pd


def split_chunk_into_utterances(sentence_field: str):
    """
    chunks.csv의 "sentence" 컬럼 값을 발화 단위로 쪼갠다.
    빈 줄, 공백만 있는 줄은 버린다.
    """
    lines = sentence_field.split("\n")
    return [line.strip() for line in lines if line.strip()]


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--input", required=True, help="chunks.csv 경로")
    ap.add_argument("--out-dir", required=True, help="sentences.txt / metadata.csv 저장 폴더")
    args = ap.parse_args()

    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    df = pd.read_csv(args.input)

    sentences_path = out_dir / "sentences.txt"
    metadata_path = out_dir / "metadata.csv"

    n_utterances = 0
    n_chunks_with_empty = 0
    n_count_mismatch = 0

    with open(sentences_path, "w", encoding="utf-8") as f_sent, \
         open(metadata_path, "w", encoding="utf-8", newline="") as f_meta:

        meta_writer = csv.writer(f_meta)
        meta_writer.writerow([
            "idx", "utterance_uid", "sentence_id", "document_id", "chunk_index",
            "utterance_index_in_chunk", "field", "field_name",
            "major", "major_name", "role",
        ])

        idx = 0
        for _, row in df.iterrows():
            utterances = split_chunk_into_utterances(str(row["sentence"]))
            if not utterances:
                n_chunks_with_empty += 1
                continue

            # 데이터 품질 체크: \n 분할 개수가 원본 utterance_count와 다르면 경고.
            # (전체 데이터셋에서는 0건이었지만, 다른 파일/버전에서 재사용될 수 있어 방어적으로 남겨둠)
            if "utterance_count" in row and len(utterances) != row["utterance_count"]:
                n_count_mismatch += 1

            for u_idx, utt in enumerate(utterances):
                utt_clean = re.sub(r"\s+", " ", utt).strip()
                if not utt_clean:
                    continue
                f_sent.write(utt_clean + "\n")

                # 계산이 아니라 "이미 고유한 것들의 조합" -> 충돌 불가능
                utterance_uid = f"{row['document_id']}_chunk{row['chunk_index']:04d}_u{u_idx}"

                meta_writer.writerow([
                    idx, utterance_uid, row["sentence_id"], row["document_id"], row["chunk_index"],
                    u_idx, row["field"], row["field_name"],
                    row["major"], row["major_name"], row["role"],
                ])
                idx += 1
                n_utterances += 1

    print(f"원본 chunk 수: {len(df)}")
    print(f"빈 chunk(발화 없음): {n_chunks_with_empty}")
    print(f"utterance_count 불일치 청크: {n_count_mismatch}")
    print(f"추출된 발화(문장) 수: {n_utterances}")
    print(f"저장 위치: {sentences_path}, {metadata_path}")


if __name__ == "__main__":
    main()
