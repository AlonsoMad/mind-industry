"""Legacy CLI entry point — redirects to mind.cli.main.

.. deprecated::
    Running ``python3 src/mind/cli.py`` is deprecated.
    Use ``mind detect run --config ...`` instead.
"""
import sys
import warnings


def main():
    warnings.warn(
        "Running 'python3 src/mind/cli.py' is deprecated. "
        "Use 'mind detect run --config ...' instead.",
        DeprecationWarning, stacklevel=2,
    )
    # Fall through to the old argparse logic for backward compat
    from mind.cli._legacy import legacy_main
    legacy_main()


if __name__ == "__main__":
    main()
