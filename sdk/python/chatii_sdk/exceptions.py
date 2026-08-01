class ChatiiError(Exception):
    """Base class for all Chatii SDK exceptions."""

    pass


class ChatiiAPIError(ChatiiError):
    """Raised when an API endpoint returns a non-200 response."""

    pass
