"""
dependency_spans.py

한국어 문장을 Stanza로 파싱해서 표준 dependency tree 형식으로 변환한다.
세연님/다슬님이 바로 쓸 수 있는 JSON 구조로 출력한다.

주의: 이 모듈을 처음 실행하면 Stanza가 한국어 UD 모델을 다운로드한다.
인터넷 연결이 되는 환경(본인 로컬 PC)에서 최초 1회 실행 시 자동으로 받아진다.
"""

from typing import List, Dict, Any


def download_korean_model():
    """최초 1회만 실행하면 됨. 로컬 캐시(~/stanza_resources)에 저장됨."""
    import stanza

    stanza.download("ko", verbose=False)


def fix_go_root(words: List[Dict[str, Any]], max_iter: int = 5) -> List[Dict[str, Any]]:
    """
    후처리 패치 (B안, v2): "-고" 연결어미가 root로 잘못 잡히는 문제 교정.

    문제: "먹고 나갔다"에서 실제 핵심(root)은 "나갔다"인데,
    Stanza가 "먹고"를 root로, "나갔다"를 그 밑에 딸린 자식으로 잘못 판정하는
    패턴이 검증됨.

    v1 → v2 변경 사항:
    - v1은 root의 자식 중 deprel=="conj"인 것만 찾아서 승격시켰는데,
      실제로는 deprel이 conj가 아닌 경우("먹고 나갔다"에서 재현됨)도 있어서
      그 케이스를 놓치는 문제가 있었음
    - v2는 deprel 종류를 따지지 않고, "root가 '고'로 끝나면서 문장의 마지막
      어절이 아닐 때, 문장의 마지막 어절을 그냥 새 root로 승격"하는 방식으로
      단순화. "-고"로 끝나는 문장은 마지막 어절이 실제 핵심 동사인 경우가
      절대다수이므로 더 넓고 안전하게 커버됨

    교정 방법:
    - root로 잡힌 어절이 "고"로 끝나고, 문장의 마지막 어절이 아니면
      → 문장의 마지막 어절을 새 root로 승격
    - 원래 root는 새 root에 딸린 advcl(부사절)로 강등
    - "-고" 연쇄(예: "A하고 B하고 C했다")도 한 번의 승격으로 바로 해결됨
      (마지막 어절로 직행하기 때문에 중간 단계를 거칠 필요가 없음)

    주의: "-고 싶다/있다" 같은 보조용언 구문은 root가 애초에 "싶다/있다"로
    잡혀서 "고"로 끝나지 않으므로 이 패치가 개입하지 않음 (안전, 검증됨).
    """
    for _ in range(max_iter):
        root = next((w for w in words if w["deprel"] == "root"), None)
        if root is None or not root["text"].endswith("고"):
            break

        # 문장 끝에 마침표·닫는 괄호가 있으면 그것을 새 root로 올리면 안 된다.
        last_word = next(
            (
                word
                for word in reversed(sorted(words, key=lambda w: w["id"]))
                if not word.get("is_punct", False)
                and word.get("upos") not in {"PUNCT", "SYM"}
            ),
            None,
        )
        if last_word is None or last_word["id"] == root["id"]:
            break  # 이미 root가 마지막 어절이면 더 손댈 것 없음

        root["head"] = last_word["id"]
        root["deprel"] = "advcl"
        last_word["head"] = 0
        last_word["deprel"] = "root"

    return words


def _extract_sentence_words(stanza_sentence) -> List[Dict[str, Any]]:
    """Convert one Stanza sentence to the project word schema."""
    return [
        {
            "id": word.id,
            "text": word.text,
            "lemma": word.lemma,
            "upos": word.upos,
            "head": word.head,
            "deprel": word.deprel,
            "is_punct": word.upos == "PUNCT",
        }
        for word in stanza_sentence.words
    ]


def merge_multi_sentence_words(
    sentence_word_lists: List[List[Dict[str, Any]]],
    apply_go_fix: bool = True,
) -> List[Dict[str, Any]]:
    """Merge Stanza's sentence-level trees without dropping later sentences."""
    merged: List[Dict[str, Any]] = []
    offset = 0
    for sentence_words in sentence_word_lists:
        if not sentence_words:
            continue
        words = [dict(word) for word in sentence_words]
        if apply_go_fix:
            # Apply the correction before changing ids so it cannot cross a
            # sentence boundary while looking for the final predicate.
            words = fix_go_root(words)
        local_max = max(word["id"] for word in words)
        for word in words:
            word["id"] += offset
            if word["head"] != 0:
                word["head"] += offset
        merged.extend(words)
        offset += local_max
    return merged


class KoreanDependencyParser:
    def __init__(self, use_gpu: bool = False):
        import stanza

        # tokenize, mwt(다어절 토큰 분리), pos, lemma, depparse 파이프라인
        self.nlp = stanza.Pipeline(
            lang="ko",
            processors="tokenize,pos,lemma,depparse",
            use_gpu=use_gpu,
            verbose=False,
        )

    def parse_sentence(self, sentence: str, apply_go_fix: bool = True) -> Dict[str, Any]:
        """
        한 문장 또는 여러 문장이 섞인 텍스트를 파싱해서 표준 형식으로 반환.

        반환 형식:
        {
            "sentence": "철수는 학교에 갔다",
            "words": [
                {"id": 1, "text": "철수는", "lemma": "철수", "upos": "PROPN",
                 "head": 3, "deprel": "nsubj", "is_punct": False},
                ...
            ],
            "multi_sentence_detected": False,
            "num_stanza_sentences": 1,
        }

        여러 문장으로 분리되더라도 각 문장에 먼저 후처리를 적용한 뒤 id를
        이어 붙인다. 따라서 뒤 문장을 조용히 버리지 않고, ``head=0`` root가
        문장별로 유지된다.
        """
        doc = self.nlp(sentence)
        if not doc.sentences:
            raise ValueError("Stanza가 문장을 생성하지 못했습니다: 입력이 비어 있는지 확인하세요.")
        sentence_word_lists = [_extract_sentence_words(sent) for sent in doc.sentences]
        words = merge_multi_sentence_words(sentence_word_lists, apply_go_fix=apply_go_fix)
        return {
            "sentence": sentence,
            "words": words,
            "multi_sentence_detected": len(doc.sentences) > 1,
            "num_stanza_sentences": len(doc.sentences),
        }

    def parse_document(self, sentences: List[str], apply_go_fix: bool = True) -> List[Dict[str, Any]]:
        """여러 문장을 배치로 파싱."""
        return [self.parse_sentence(s, apply_go_fix=apply_go_fix) for s in sentences]


def _find_root(parent: Dict[int, int], x: int) -> int:
    while parent[x] != x:
        parent[x] = parent[parent[x]]
        x = parent[x]
    return x


def _is_connected(word_ids: List[int], head_of: Dict[int, int]) -> bool:
    """윈도우 안의 dependency edge가 하나의 연결 요소인지 확인한다."""
    if len(word_ids) <= 1:
        return True

    parent = {word_id: word_id for word_id in word_ids}
    id_set = set(word_ids)
    for word_id in word_ids:
        head = head_of.get(word_id, 0)
        if head in id_set:
            left = _find_root(parent, word_id)
            right = _find_root(parent, head)
            if left != right:
                parent[left] = right
    return len({_find_root(parent, word_id) for word_id in word_ids}) == 1


def generate_spans_for_L(words: List[Dict[str, Any]], L: int) -> List[Dict[str, Any]]:
    """표면상 연속이고 dependency tree에서 연결된 정확히 L개 span을 만든다."""
    if not isinstance(L, int) or L < 1:
        raise ValueError(f"L은 1 이상의 정수여야 합니다: {L!r}")

    head_of = {word["id"]: word["head"] for word in words}
    text_of = {word["id"]: word["text"] for word in words}
    id_list = [word["id"] for word in words]
    spans: List[Dict[str, Any]] = []
    for start in range(len(id_list) - L + 1):
        word_ids = id_list[start : start + L]
        if _is_connected(word_ids, head_of):
            spans.append(
                {
                    "start": word_ids[0],
                    "end": word_ids[-1],
                    "size": L,
                    "text": " ".join(text_of[word_id] for word_id in word_ids),
                    "word_ids": word_ids,
                    "contains_punct": any(
                        word.get("is_punct", False)
                        for word in words
                        if word["id"] in word_ids
                    ),
                }
            )
    return spans


def generate_spans_up_to_L(words: List[Dict[str, Any]], max_L: int) -> List[Dict[str, Any]]:
    """Return every valid span whose size is in ``1..max_L``."""
    if not isinstance(max_L, int) or max_L < 1:
        raise ValueError(f"L은 1 이상의 정수여야 합니다: {max_L!r}")
    return [
        span
        for size in range(1, min(max_L, len(words)) + 1)
        for span in generate_spans_for_L(words, size)
    ]


def generate_all_spans(
    words: List[Dict[str, Any]], L_values: List[int] = (1, 2, 4, 8)
) -> Dict[str, Any]:
    """여러 L값의 span 후보를 누적 풀(크기 1부터 L까지)로 반환한다."""
    return {
        str(L): generate_spans_up_to_L(words, L)
        for L in L_values
    }


def self_test() -> bool:
    """Run parser-independent regression checks for the span utilities."""
    go_case = [
        {"id": 1, "text": "먹고", "head": 0, "deprel": "root"},
        {"id": 2, "text": "나갔다", "head": 1, "deprel": "parataxis"},
    ]
    fixed = fix_go_root([dict(word) for word in go_case])
    assert next(word for word in fixed if word["deprel"] == "root")["text"] == "나갔다"

    punct_case = go_case + [
        {"id": 3, "text": ".", "head": 1, "deprel": "punct", "is_punct": True}
    ]
    fixed_punct = fix_go_root([dict(word) for word in punct_case])
    assert next(word for word in fixed_punct if word["deprel"] == "root")["text"] == "나갔다"

    words = [
        {"id": 1, "text": "영희는", "head": 3, "deprel": "nsubj"},
        {"id": 2, "text": "학교에", "head": 3, "deprel": "obl"},
        {"id": 3, "text": "갔다", "head": 0, "deprel": "root"},
    ]
    sizes = sorted({span["size"] for span in generate_all_spans(words, [1, 2])["2"]})
    assert sizes == [1, 2], f"L 누적 span 생성 실패: {sizes}"
    return True
