"""Plan 009: automatic flood-trace / damage change map (V1) and event validation (V2-V6, H6).

``coregister``, ``cloudmask``, ``normalise``, ``features`` and ``change`` build the change map
from imagery and footprints only; they never import HAND, DEM or river code. ``validate``
scores HAND against the change map and the audit, behind the plan 005 guard.
"""
