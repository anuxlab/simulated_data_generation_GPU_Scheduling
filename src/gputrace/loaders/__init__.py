"""
Loader registry. Importing this package registers all built-in loaders
(alibaba2020, google2011, synthetic) via their ``@register(...)`` decorator.
"""

from .base import BaseLoader, get_loader, list_loaders, register  # noqa: F401
from . import alibaba2020  # noqa: F401
from . import google_cluster  # noqa: F401
from . import synthetic  # noqa: F401

__all__ = ["BaseLoader", "get_loader", "list_loaders", "register"]
