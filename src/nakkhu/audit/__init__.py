"""Author yes/no audit (plan 006, R-AUD): frames, pooled stratified samples, local chips, a
keyboard-only click page served on 127.0.0.1, and design-weighted estimators.

Every audited item lies in the test region. Sampling, chips and the page need the frozen
pre-registration (:func:`nakkhu.evaluate.guard.require_frozen`); turning answers into estimates
about a source or a model is a test access and is journalled (``nakkhu audit estimate``).
"""
