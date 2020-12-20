# Create a Bunch class that acts like a struct (mybunch.key = val)

class Bunch(dict):
    """Dictionary-like object that exposes its keys as attributes."""

    def __init__(self, **kwargs):  # noqa: D102
        dict.__init__(self, kwargs)
        self.__dict__ = self
