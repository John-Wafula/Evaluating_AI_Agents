from typing import Any, Dict, List, Optional

from pydantic import BaseModel, ConfigDict


class Settings(BaseModel):
    """Settings for a prompt run that a user can configure."""

    model_alias: Optional[str] = None
    temperature: Optional[float] = None
    max_tokens: Optional[int] = None
    stop_sequences: Optional[List[str]] = None

    top_p: Optional[float] = None
    frequency_penalty: Optional[float] = None
    presence_penalty: Optional[float] = None

    # Internal settings.
    n: Optional[int] = None
    tools: Optional[List[Dict[str, Any]]] = None
    response_format: Optional[Dict[str, str]] = None

    model_config = ConfigDict(
        # Avoid Pydantic's protected namespace warning since we want to use
        # `model_alias` as a field name.
        protected_namespaces=(),
        # Disallow extra fields.
        extra="allow",
    )
