"""Provides loss metrics for coregistration."""


from .mi import mi
from .rmse import rmse

__all__ = [
    "mi",
    "rmse"
]
