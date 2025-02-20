from types import SimpleNamespace
from typing import Optional

from pydantic import BaseModel

PaginationDefaults = SimpleNamespace(starting_token=0, limit=25)


class PaginationRequestMixin(BaseModel):
    starting_token: int = PaginationDefaults.starting_token
    limit: int = PaginationDefaults.limit


class PaginationResponseMixin(PaginationRequestMixin, BaseModel):
    # Did we paginate?
    paginated: bool = False
    # If we paginated, what is the next starting token?
    # If we didn't paginate, this is None.
    next_starting_token: Optional[int] = None
