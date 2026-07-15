"""
Backward-compatible configuration shim.

Existing imports of `config` continue to work; new code should import from `core.config`.
"""

from core.config import *  # noqa: F401,F403
