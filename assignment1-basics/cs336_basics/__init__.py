# import importlib.metadata

# __version__ = importlib.metadata.version("cs336_basics")

try:
    from importlib.metadata import version, PackageNotFoundError
except Exception:  # Py<3.8 fallback if you ever need it
    from importlib_metadata import version, PackageNotFoundError  # type: ignore

try:
    __version__ = version("cs336_basics")
except PackageNotFoundError:
    # Not installed as a distribution; use a safe fallback
    __version__ = "0.0.0"