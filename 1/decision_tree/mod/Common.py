from __future__ import annotations
from typing import Any, NoReturn


def attrgetter(obj: object, name: str, value: Any = None) -> Any:
    """
    Returns value from nested objects/chained attributes (basically, getattr() on steroids)
    :param obj: Primary object
    :param name: Path to an attribute (dot separated)
    :param value: Default value returned if a function fails to find the requested attribute value
    :return:
    """
    for attribute in name.split('.'):
        obj = getattr(obj, attribute, value)
    return obj


def attrsetter(obj: object, name: str, value: Any) -> NoReturn:
    """
    Sets the value of an attribute of a (nested) object (basically, setattr() on steroids)
    :param obj: Primary object
    :param name: Path to an attribute (dot separated)
    :param value: Value to be set
    """
    pre, _, post = name.rpartition('.')
    setattr(attrgetter(obj, pre) if pre else obj, post, value)
