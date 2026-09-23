"""graphrag — top-level package."""
from importlib.metadata import version, PackageNotFoundError

try:
    __version__ = version("graphrag-neo4j-laya")
except PackageNotFoundError:
    __version__ = "0.1.0-dev"

__all__ = ["__version__"]
