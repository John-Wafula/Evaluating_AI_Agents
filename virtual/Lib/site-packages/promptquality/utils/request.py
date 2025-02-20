from enum import Enum
from http.client import HTTPException
from json import loads
from typing import Any, Callable, Dict, Optional, Union
from urllib.parse import urljoin

from requests import Response


def _validate_response(response: Response) -> None:
    if response.ok:
        return

    error_message = response.text
    try:
        if detail := response.json().get("detail"):
            error_message = detail
    finally:
        error_message = f"❌ Something didn't go quite right. Error message: {error_message}"
        raise HTTPException(error_message) from None


def make_request(
    request_method: Callable,
    base_url: str,
    endpoint: str,
    body: Optional[Dict] = None,
    data: Optional[Dict] = None,
    files: Optional[Dict] = None,
    params: Optional[Dict] = None,
    headers: Optional[Dict[str, str]] = None,
    timeout: Union[int, None] = None,
    stream: bool = False,
) -> Any:
    """
    Make a request to a Galileo API endpoint.

    For streaming responses, set stream=True. We assume that streaming responses are json lines for now.
    """
    if not stream:
        response = request_method(
            urljoin(base_url, endpoint),
            json=body,
            data=data,
            files=files,
            params=params,
            headers=headers or {},
            timeout=timeout,
        )
        _validate_response(response)
        return response.json()
    else:
        # Stream response as json lines.
        lines = list()
        with request_method(
            urljoin(base_url, endpoint),
            json=body,
            data=data,
            files=files,
            params=params,
            headers=headers or {},
            timeout=timeout,
            stream=stream,
        ) as response:
            _validate_response(response)
            for line in response.iter_lines():
                if line:
                    lines.append(loads(line.decode("utf-8")))
        return lines


class HttpHeaders(str, Enum):
    accept = "accept"
    content_type = "Content-Type"
    application_json = "application/json"

    @staticmethod
    def accept_json() -> Dict[str, str]:
        return {HttpHeaders.accept: HttpHeaders.application_json}

    @staticmethod
    def content_type_json() -> Dict[str, str]:
        return {HttpHeaders.content_type: HttpHeaders.application_json}

    @staticmethod
    def json() -> Dict[str, str]:
        return {**HttpHeaders.accept_json(), **HttpHeaders.content_type_json()}
