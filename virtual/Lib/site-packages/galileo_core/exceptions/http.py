from http.client import HTTPException

from galileo_core.helpers.logger import logger


class GalileoHTTPException(HTTPException):
    """Galileo HTTP exception to wrap all http exceptions."""

    def __init__(self, message: str, status_code: int, response_text: str) -> None:
        self.message = message
        self.status_code = status_code
        self.response_text = response_text
        logger.error(f"Galileo API returned HTTP status code {status_code}. Error was: {response_text}.")
