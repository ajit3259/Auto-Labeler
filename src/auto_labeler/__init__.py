import warnings

# Suppress Pydantic serializer warnings from LiteLLM/Pydantic v2 interaction
# These are harmless noise in this context.
warnings.filterwarnings("ignore", message=".*Pydantic serializer warnings.*", category=UserWarning)

__version__ = "0.1.3"

from .core import AutoLabeler

__all__ = ["AutoLabeler"]
