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
                if word.get("upos") not in {"PUNCT", "SYM"}
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
        문장 하나를 파싱해서 표준 형식으로 반환.

        반환 형식:
        {
            "sentence": "철수는 학교에 갔다",
            "words": [
                {"id": 1, "text": "철수는", "lemma": "철수", "upos": "PROPN",
                 "head": 3, "deprel": "nsubj"},
                ...
            ]
        }

        head=0은 root를 의미 (관례상 ROOT 노드의 id를 0으로 취급).
        id는 문장 내에서 1부터 시작하는 표면 순서(surface order)와 동일하다.
        따라서 "표면상 연속"인지 여부는 id의 연속성으로 바로 판단 가능하다.
        """
        doc = self.nlp(sentence)
        if not doc.sentences:
            raise ValueError("Stanza가 문장을 생성하지 못했습니다: 입력이 비어 있는지 확인하세요.")
        if len(doc.sentences) != 1:
            raise ValueError(
                "parse_sentence는 한 문장만 받습니다. "
                "여러 문장은 먼저 분리해 parse_document에 넘기세요."
            )
        sent = doc.sentences[0]
        words = []
        for w in sent.words:
            words.append({
                "id": w.id,
                "text": w.text,
                "lemma": w.lemma,
                "upos": w.upos,
                "head": w.head,      # 0이면 root
                "deprel": w.deprel,
            })

        if apply_go_fix:
            words = fix_go_root(words)

        return {"sentence": sentence, "words": words}

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
                }
            )
    return spans


def generate_all_spans(
    words: List[Dict[str, Any]], L_values: List[int] = (1, 2, 4, 8)
) -> Dict[str, Any]:
    """여러 L값의 span 후보를 크기별 딕셔너리로 반환한다."""
    return {
        str(L): generate_spans_for_L(words, L) if L <= len(words) else []
        for L in L_values
    }
