from fastapi import APIRouter


class VersionRegistry:
    """Manages API routers grouped by target semantic version tags."""

    def __init__(self) -> None:
        self.versions = {}

    def register_version(self, version_tag: str, router: APIRouter) -> None:
        self.versions[version_tag] = router
