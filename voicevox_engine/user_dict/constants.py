"""ユーザー辞書関連の定数とユーティリティ"""

from dataclasses import dataclass
from enum import StrEnum

import numpy as np


class WordTypes(StrEnum):
    """品詞を表す列挙型。"""

    PROPER_NOUN = "PROPER_NOUN"  # 固有名詞
    LOCATION_NAME = "LOCATION_NAME"  # 地名
    ORGANIZATION_NAME = "ORGANIZATION_NAME"  # 組織・施設名
    PERSON_NAME = "PERSON_NAME"  # 人名
    PERSON_FAMILY_NAME = "PERSON_FAMILY_NAME"  # 姓
    PERSON_GIVEN_NAME = "PERSON_GIVEN_NAME"  # 名
    COMMON_NOUN = "COMMON_NOUN"  # 一般名詞
    VERB = "VERB"  # 動詞
    ADJECTIVE = "ADJECTIVE"  # 形容詞
    SUFFIX = "SUFFIX"  # 接尾辞


@dataclass
class WordProperty:
    """UserDictWord の生成に必要な、ユーザー辞書に登録する単語情報。"""

    surface: list[str]  # 表層形 (アクセント句ごとに1つずつ)
    pronunciation: list[str]  # 発音 (アクセント句ごとに1つずつ)
    accent_type: list[int]  # アクセント型 (アクセント句ごとに1つずつ)
    word_type: WordTypes  # 品詞
    priority: int  # 優先度


@dataclass(frozen=True)
class _PartOfSpeechDetail:
    """品詞ごとの情報。"""

    part_of_speech: str  # 品詞名
    part_of_speech_detail_1: str  # 品詞細分類1
    part_of_speech_detail_2: str  # 品詞細分類2
    part_of_speech_detail_3: str  # 品詞細分類3
    context_id: int  # 辞書の左・右文脈 ID (ref: https://github.com/VOICEVOX/open_jtalk/blob/1.11/src/mecab-naist-jdic/_left-id.def) # noqa
    cost_candidates: list[int]  # コストのパーセンタイル
    accent_associative_rules: list[str]  # アクセント結合規則の一覧


# tools/get_cost_candidates.py で生成された、優先度に対応する生起コストのパーセンタイル
_COSTS_PROPER_NOUN = [-988, 3488, 4768, 6048, 7328, 8609, 8734, 8859, 8984, 9110, 14176]
_COSTS_LOCATION_NAME = [147, 4350, 5403, 6456, 7509, 8562, 8778, 8994, 9210, 9427, 13960]  # fmt: skip # noqa
_COSTS_ORGANIZATION_NAME = [1691, 4087, 4808, 5530, 6251, 6973, 7420, 7867, 8314, 8762, 13887]  # fmt: skip # noqa
_COSTS_PERSON_NAME = [515, 516, 2253, 3990, 5727, 7464, 8098, 8732, 9366, 10000, 10001]
_COSTS_PERSON_FAMILY_NAME = [1991, 4126, 4974, 5823, 6672, 7521, 8001, 8481, 8961, 9442, 12808]  # fmt: skip # noqa
_COSTS_PERSON_GIVEN_NAME = [2209, 3772, 4905, 6038, 7171, 8304, 8728, 9152, 9576, 10000, 13842]  # fmt: skip # noqa
_COSTS_COMMON_NOUN = [-4445, 49, 1473, 2897, 4321, 5746, 6554, 7362, 8170, 8979, 15001]
_COSTS_VERB = [3100, 6160, 6360, 6561, 6761, 6962, 7414, 7866, 8318, 8771, 13433]
_COSTS_ADJECTIVE = [1527, 3266, 3561, 3857, 4153, 4449, 5149, 5849, 6549, 7250, 10001]
_COSTS_SUFFIX = [4399, 5373, 6041, 6710, 7378, 8047, 9440, 10834, 12228, 13622, 15847]

# 品詞ごとにどのような値をユーザー辞書 CSV に記述すべきかを表す定数
PART_OF_SPEECH_DATA: dict[WordTypes, _PartOfSpeechDetail] = {
    WordTypes.PROPER_NOUN: _PartOfSpeechDetail(
        part_of_speech="名詞",
        part_of_speech_detail_1="固有名詞",
        part_of_speech_detail_2="一般",
        part_of_speech_detail_3="*",
        context_id=1348,
        cost_candidates=_COSTS_PROPER_NOUN,
        accent_associative_rules=["*", "C1", "C2", "C3", "C4", "C5"],
    ),
    WordTypes.LOCATION_NAME: _PartOfSpeechDetail(
        part_of_speech="名詞",
        part_of_speech_detail_1="固有名詞",
        part_of_speech_detail_2="地域",
        part_of_speech_detail_3="一般",
        context_id=1353,
        cost_candidates=_COSTS_LOCATION_NAME,
        accent_associative_rules=["*", "C1", "C2", "C3", "C4", "C5"],
    ),
    WordTypes.ORGANIZATION_NAME: _PartOfSpeechDetail(
        part_of_speech="名詞",
        part_of_speech_detail_1="固有名詞",
        part_of_speech_detail_2="組織",
        part_of_speech_detail_3="*",
        context_id=1352,
        cost_candidates=_COSTS_ORGANIZATION_NAME,
        accent_associative_rules=["*", "C1", "C2", "C3", "C4", "C5"],
    ),
    WordTypes.PERSON_NAME: _PartOfSpeechDetail(
        part_of_speech="名詞",
        part_of_speech_detail_1="固有名詞",
        part_of_speech_detail_2="人名",
        part_of_speech_detail_3="一般",
        context_id=1349,
        cost_candidates=_COSTS_PERSON_NAME,
        accent_associative_rules=["*", "C1", "C2", "C3", "C4", "C5"],
    ),
    WordTypes.PERSON_FAMILY_NAME: _PartOfSpeechDetail(
        part_of_speech="名詞",
        part_of_speech_detail_1="固有名詞",
        part_of_speech_detail_2="人名",
        part_of_speech_detail_3="姓",
        context_id=1350,
        cost_candidates=_COSTS_PERSON_FAMILY_NAME,
        accent_associative_rules=["*", "C1", "C2", "C3", "C4", "C5"],
    ),
    WordTypes.PERSON_GIVEN_NAME: _PartOfSpeechDetail(
        part_of_speech="名詞",
        part_of_speech_detail_1="固有名詞",
        part_of_speech_detail_2="人名",
        part_of_speech_detail_3="名",
        context_id=1351,
        cost_candidates=_COSTS_PERSON_GIVEN_NAME,
        accent_associative_rules=["*", "C1", "C2", "C3", "C4", "C5"],
    ),
    WordTypes.COMMON_NOUN: _PartOfSpeechDetail(
        part_of_speech="名詞",
        part_of_speech_detail_1="一般",
        part_of_speech_detail_2="*",
        part_of_speech_detail_3="*",
        context_id=1345,
        cost_candidates=_COSTS_COMMON_NOUN,
        accent_associative_rules=["*", "C1", "C2", "C3", "C4", "C5"],
    ),
    WordTypes.VERB: _PartOfSpeechDetail(
        part_of_speech="動詞",
        part_of_speech_detail_1="自立",
        part_of_speech_detail_2="*",
        part_of_speech_detail_3="*",
        context_id=642,
        cost_candidates=_COSTS_VERB,
        accent_associative_rules=["*"],
    ),
    WordTypes.ADJECTIVE: _PartOfSpeechDetail(
        part_of_speech="形容詞",
        part_of_speech_detail_1="自立",
        part_of_speech_detail_2="*",
        part_of_speech_detail_3="*",
        context_id=20,
        cost_candidates=_COSTS_ADJECTIVE,
        accent_associative_rules=["*"],
    ),
    WordTypes.SUFFIX: _PartOfSpeechDetail(
        part_of_speech="名詞",
        part_of_speech_detail_1="接尾",
        part_of_speech_detail_2="一般",
        part_of_speech_detail_3="*",
        context_id=1358,
        cost_candidates=_COSTS_SUFFIX,
        accent_associative_rules=["*", "C1", "C2", "C3", "C4", "C5"],
    ),
}

# ユーザー辞書の優先度の最小値と最大値
USER_DICT_MIN_PRIORITY = 0
USER_DICT_MAX_PRIORITY = 10


def _search_cost_candidates(context_id: int) -> list[int]:
    # 循環インポート回避のためここでインポート
    from voicevox_engine.user_dict.model import UserDictInputError

    for value in PART_OF_SPEECH_DATA.values():
        if value.context_id == context_id:
            return value.cost_candidates
    raise UserDictInputError("品詞 ID が不正です。")


def cost2priority(context_id: int, cost: int) -> int:
    """生起コストから辞書の優先度を計算する。"""
    assert -32768 <= cost <= 32767
    cost_candidates = _search_cost_candidates(context_id)
    # cost_candidatesの中にある値で最も近い値を元にpriorityを返す
    # 参考: https://qiita.com/Krypf/items/2eada91c37161d17621d
    # この関数とpriority2cost関数によって、辞書ファイルのcostを操作しても最も近いpriorityのcostに上書きされる
    return (
        USER_DICT_MAX_PRIORITY
        - np.argmin(np.abs(np.array(cost_candidates) - cost)).item()
    )


def priority2cost(context_id: int, priority: int) -> int:
    """辞書の優先度から生起コストを計算する。"""
    assert USER_DICT_MIN_PRIORITY <= priority <= USER_DICT_MAX_PRIORITY
    cost_candidates = _search_cost_candidates(context_id)
    return cost_candidates[USER_DICT_MAX_PRIORITY - priority]
