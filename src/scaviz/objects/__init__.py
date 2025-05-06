"""
This module contains the pygfx world objects developped for scaviz.
Unlike pygfx, we develop jointly the world object, its material (if necessary) and its shader.
"""

__all__ = ["Seq", "SeqMaterial"]

from .pooled_seq import Seq, SeqMaterial
