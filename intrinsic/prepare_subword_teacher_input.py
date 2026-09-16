"""Create tokenized chunk records for the ChatGPT subword teacher."""

from __future__ import annotations

import argparse
import csv
import json
import unicodedata
from pathlib import Path
from typing import Any, Dict, List


class _LocalWordPieceEncoding(dict):
    def __init__(self, input_ids: List[int], word_ids: List[int | None]):
        super().__init__(input_ids=input_ids)
        self._word_ids = word_ids

    def word_ids(self) -> List[int | None]:
        return self._word_ids


class _LocalWordPieceTokenizer:
    """Small stdlib fallback for the saved KLUE WordPiece tokenizer."""

    def __init__(self, tokenizer_json: Path):
        payload = json.loads(tokenizer_json.read_text(encoding="utf-8"))
        model = payload.get("model", {})
        if model.get("type") != "WordPiece":
            raise ValueError(f"WordPiece tokenizer가 아닙니다: {tokenizer_json}")
        self.vocab: Dict[str, int] = {
            str(token): int(index) for token, index in model["vocab"].items()
        }
        self.unk_token = str(model.get("unk_token", "[UNK]"))
        self.max_chars_per_word = int(model.get("max_input_chars_per_word", 100))

    @staticmethod
    def _is_punctuation(char: str) -> bool:
        category = unicodedata.category(char)
        return category.startswith("P") or category.startswith("S")

    def _basic_pieces(self, word: str) -> List[str]:
        pieces: List[str] = []
        current: List[str] = []
        for char in word:
            if self._is_punctuation(char):
                if current:
                    pieces.append("".join(current))
                    current = []
                pieces.append(char)
            else:
                current.append(char)
        if current:
            pieces.append("".join(current))
        return pieces

    def _wordpiece(self, text: str) -> List[str]:
        if len(text) > self.max_chars_per_word:
            return [self.unk_token]
        result: List[str] = []
        for piece in self._basic_pieces(text):
            if piece in self.vocab:
                result.append(piece)
                continue
            start = 0
            found: List[str] = []
            failed = False
            while start < len(piece):
                end = len(piece)
                match = None
                while start < end:
                    candidate = piece[start:end]
                    if start > 0:
                        candidate = "##" + candidate
                    if candidate in self.vocab:
                        match = candidate
                        break
                    end -= 1
                if match is None:
                    failed = True
                    break
                found.append(match)
                start = end
            result.extend([self.unk_token] if failed else found)
        return result

    def __call__(self, words: List[str], **_: Any) -> _LocalWordPieceEncoding:
        tokens: List[str] = []
        word_ids: List[int | None] = []
        for word_index, word in enumerate(words):
            for token in self._wordpiece(word):
                tokens.append(token)
                word_ids.append(word_index)
        return _LocalWordPieceEncoding(
            [self.vocab.get(token, self.vocab[self.unk_token]) for token in tokens],
            word_ids,
        )

    def convert_ids_to_tokens(self, token_id: int) -> str:
        for token, index in self.vocab.items():
            if index == token_id:
                return token
        return self.unk_token


def _load_tokenizer(model_name: str, local_path: Path | None):
    try:
        from transformers import AutoTokenizer

        return AutoTokenizer.from_pretrained(str(local_path or model_name), use_fast=True)
    except (ImportError, OSError, RuntimeError, ValueError):
        if local_path is None:
            raise RuntimeError(
                "transformers를 사용할 수 없습니다. --tokenizer-path에 encoder의 "
                "tokenizer 디렉터리를 지정하세요."
            )
        return _LocalWordPieceTokenizer(local_path / "tokenizer.json")


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--chunks", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--tokenizer", default="klue/roberta-base")
    parser.add_argument(
        "--tokenizer-path",
        type=Path,
        help="저장된 fast tokenizer 디렉터리. 없으면 Transformers Hub에서 로드",
    )
    args = parser.parse_args()

    tokenizer = _load_tokenizer(args.tokenizer, args.tokenizer_path)
    if not hasattr(tokenizer, "word_ids") and not hasattr(tokenizer, "__call__"):
        raise ValueError("subword teacher input에 사용할 tokenizer를 읽지 못했습니다.")

    args.output.parent.mkdir(parents=True, exist_ok=True)
    count = 0
    subword_count = 0
    with args.chunks.open("r", encoding="utf-8-sig", newline="") as source, args.output.open(
        "w", encoding="utf-8", newline=""
    ) as target:
        reader = csv.DictReader(source)
        if not reader.fieldnames or "sentence_id" not in reader.fieldnames:
            raise ValueError("chunks.csv에 sentence_id 열이 필요합니다.")
        text_column = "sentence" if "sentence" in reader.fieldnames else "context"
        if text_column not in reader.fieldnames:
            raise ValueError("chunks.csv에 sentence 또는 context 열이 필요합니다.")

        for row_number, row in enumerate(reader, start=2):
            sentence_id = str(row.get("sentence_id") or "").strip()
            text = str(row.get(text_column) or "")
            if not sentence_id:
                raise ValueError(f"sentence_id가 비어 있습니다: {args.chunks}:{row_number}")

            words = text.split()
            encoding = tokenizer(
                words,
                is_split_into_words=True,
                add_special_tokens=False,
                truncation=False,
            )
            input_ids = list(encoding["input_ids"])
            word_ids = encoding.word_ids()
            tokens = []
            for token_index, (token_id, word_id) in enumerate(zip(input_ids, word_ids)):
                if word_id is None:
                    continue
                tokens.append(
                    {
                        "index": len(tokens),
                        "word_index": int(word_id),
                        "token": tokenizer.convert_ids_to_tokens(int(token_id)),
                        "token_id": int(token_id),
                    }
                )

            record = {
                "sentence_id": sentence_id,
                "text": text,
                "tokenizer": args.tokenizer,
                "n_words": len(words),
                "tokens": tokens,
            }
            target.write(json.dumps(record, ensure_ascii=False) + "\n")
            count += 1
            subword_count += len(tokens)

    print(f"생성 완료: {args.output}")
    print(f"chunks={count}, subwords={subword_count}, tokenizer={args.tokenizer}")


if __name__ == "__main__":
    main()
