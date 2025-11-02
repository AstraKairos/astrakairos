"""
    Testing script to determine the precision and accuracy of the physicality
    probability calculation model.
    Core idea:
    - Systems with a ≥0.99 physicality probability are "true" physical binaries.
    - Systems with a ≤0.01 physicality probability are "optical doubles" ("chance
    alignments").
    - Anything in between is uncertain.
"""

physical_limit = 0.99
optical_limit = 0.01