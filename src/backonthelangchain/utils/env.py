"""Environment loading helpers for notebooks and scripts."""

from dotenv import load_dotenv


def load_project_env() -> None:
    """Load local .env settings when present."""
    load_dotenv()
