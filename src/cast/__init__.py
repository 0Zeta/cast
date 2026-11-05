"""Detect and localize audio-visual forgeries."""


from .config import (CastConfig, EvalConfig, ExecutionConfig,
                     RankingConfig, ScanConfig, load_config_document)

__version__ = "0.1.0"

__all__ = ["CastConfig", "ScanConfig", "RankingConfig", "EvalConfig",
           "ExecutionConfig", "load_config_document", "__version__"]
