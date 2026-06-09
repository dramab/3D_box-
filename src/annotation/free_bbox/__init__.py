"""Free-space 3D bounding box placement annotation pipeline."""

from src.annotation.free_bbox.datatypes import FreeBBoxConfig, FreeBBoxResult
from src.annotation.free_bbox.pipeline import FreeBBoxPipeline

__all__ = ["FreeBBoxConfig", "FreeBBoxPipeline", "FreeBBoxResult"]
