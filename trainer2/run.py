"""Entry point: python -m trainer2.run (the code lives in cli/pipeline.py)."""
from __future__ import annotations

# The real implementation is in cli/pipeline.py
from .cli.pipeline import main, run, self_test

# Names this module offers
__all__ = ["main", "run", "self_test"]

# Run the pipeline when used as a script
if __name__ == "__main__":
    main()
