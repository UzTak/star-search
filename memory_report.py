"""Compatibility wrapper for running the STAR memory report from a source checkout."""

from star import memory_report as _impl

globals().update(
    {
        name: getattr(_impl, name)
        for name in dir(_impl)
        if not (name.startswith("__") and name.endswith("__"))
    }
)


if __name__ == "__main__":
    main()
