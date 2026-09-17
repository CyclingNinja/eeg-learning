"""Enable ``python -m eeg_learning`` — delegates to the CLI's ``main``."""

import sys

from eeg_learning.cli import main

if __name__ == "__main__":
    sys.exit(main())
