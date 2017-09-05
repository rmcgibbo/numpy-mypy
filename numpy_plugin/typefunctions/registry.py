__all__ = ['register', 'registry']

registry = {}


def register(name: str):
    def inner(f):
        registry[name] = f
        return f

    return inner
