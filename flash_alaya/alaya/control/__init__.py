"""Control-condition extension points.

Camera control is intentionally disabled in this first clean rewrite. The
trainer already keeps sink/memory/visual-condition separate from control, so
camera can be added here without touching the core rollout split.
"""

