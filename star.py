"""Compatibility wrapper for running the STAR search from a source checkout."""

from star.pipeline import *  # noqa: F401,F403
from star.pipeline import main


if __name__ == "__main__":
    main()
