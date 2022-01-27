"""
Some utilities for operations
"""

import functools  # for deprecated_alias decorator
import warnings

cacheSteps = [
    "plot",
    "clip",
    "flag",
    "norm",
    "smooth",
]  # steps to use chaced data


# fancy backwards compatibility of keywords: allow aliases
# https://stackoverflow.com/questions/49802412/how-to-implement-deprecation-in-python-with-argument-alias#
def deprecated_alias(**aliases):
    """
    Function to allow aliases
    """

    def deco(func):
        @functools.wraps(func)
        def wrapper(*args, **kwargs):
            rename_kwargs(func.__name__, kwargs, aliases)
            return func(*args, **kwargs)

        return wrapper

    return deco


def rename_kwargs(func_name, kwargs, aliases):
    """
    Rename aliases
    """
    for alias, new in aliases.items():
        if alias in kwargs:
            if new in kwargs:
                raise TypeError(
                    "{} received both {} and {}".format(func_name, alias, new)
                )
            warnings.warn(
                "{} is deprecated; use {}".format(alias, new),
                DeprecationWarning,
            )
            kwargs[new] = kwargs.pop(alias)
