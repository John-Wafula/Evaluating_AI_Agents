from typing import Generator, List

from pydantic import RootModel, field_validator

from galileo_core.schemas.shared.message import Message
from galileo_core.schemas.shared.message_role import MessageRole


class Messages(RootModel[List[Message]]):
    def __len__(self) -> int:
        return len(self.root)

    def __iter__(self) -> Generator[Message, None, None]:  # type: ignore[override]
        yield from self.root

    def __getitem__(self, item: int) -> Message:
        return self.root[item]

    @field_validator("root", mode="after")
    def system_message_first(cls, messages: List[Message]) -> List[Message]:
        for message in messages[1:]:
            if message.role == MessageRole.system:
                raise ValueError("There can only be 1 system message and it must be the first message.")
        return messages
