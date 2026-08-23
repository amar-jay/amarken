"""Local command-line inference for every registered Amarken model.

The package deliberately avoids importing ``cli`` eagerly so
``python -m src.inference.cli`` executes without runpy's double-import warning.
"""
