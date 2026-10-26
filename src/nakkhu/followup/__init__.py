"""Plan 014 follow-up tests on the HAND result (prereg addendum ``prereg_v9_a3``).

- :mod:`nakkhu.followup.surfaces`: label-free derived predictors of the addendum's
  ``definitions`` (main-river HAND, stream-threshold HAND, drainage matching, terrain features,
  the team recipe).
- :mod:`nakkhu.followup.prep`: ``nakkhu followup prep``, every label-free table the tests read.
- :mod:`nakkhu.followup.stats` / :mod:`nakkhu.followup.analyses` / :mod:`nakkhu.followup.ml`:
  the Phase-1 statistics (reusing :mod:`nakkhu.flood.validate`).
- :mod:`nakkhu.followup.run`: ``nakkhu followup run --test <id>``, which journals the access
  through :func:`nakkhu.evaluate.addendum.record_addendum_access` before any label is read.
"""
