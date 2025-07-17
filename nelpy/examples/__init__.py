"""
nelpy.examples
==============

This module contains helper functions to load example data.
"""

from ._utils import (
    download_example_dataset,
    get_example_data_home,
    load_example_dataset,
)

__all__ = ["load_example_dataset", "download_example_dataset", "get_example_data_home"]

__version__ = "0.0.1"
