def ReturnsNamedTuple(func):
    """
    Returns the results of a method as 
    a named tuple where possible.
    """
    def wrapper(*args, **kwargs):
        results = func(*args, **kwargs)
        try:
            pass
        except TypeError:
            return results