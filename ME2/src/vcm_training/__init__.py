"""Training and inference components for voice-command models."""

from .models import ARCHITECTURES, build_model

__all__ = ["ARCHITECTURES", "build_model"]
