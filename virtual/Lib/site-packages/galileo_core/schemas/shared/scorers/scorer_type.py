from enum import Enum
from typing import Literal


class ScorerType(str, Enum):
    luna = "luna"
    plus = "plus"


PlusScorerType = Literal[ScorerType.plus]
LunaOrPlusScorerType = Literal[ScorerType.luna, ScorerType.plus]
