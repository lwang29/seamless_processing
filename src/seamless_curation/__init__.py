"""Fully automated curation pipeline for a co-speech upper-body subset of the
Seamless Interaction dataset.

The source tree is read-only. Everything this package produces lands under
``outputs/`` (tables, manifests) or ``artifacts/`` (private participant-media
derivatives); both are git-ignored, and the accepted subset is defined by a
manifest of frame ranges rather than by copied media.
"""

__version__ = "1.0.0"
