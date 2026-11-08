"""The check a registered base class makes of its subclasses when they are defined."""

__all__ = ["check_abstract_members"]


def check_abstract_members(cls: type) -> None:
    """Refuse a class that leaves an abstract method or property undefined.

    Python checks these only when a class is instantiated, which a class that a
    campaign names is not until the campaign first uses it.

    Raises:
        TypeError: Naming what the class must define.
    """
    abstract = sorted(
        name
        for name in dir(cls)
        if getattr(getattr(cls, name, None), "__isabstractmethod__", False)
    )
    if abstract:
        raise TypeError(f"{cls.__name__} must define {', '.join(abstract)}")
