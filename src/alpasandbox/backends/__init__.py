from .base import Backend


def make_backend(name: str, **kw) -> Backend:
    if name == "builtin":
        from .builtin import BuiltinBackend
        return BuiltinBackend(**kw)
    if name == "carla":
        from .carla_backend import CarlaBackend
        return CarlaBackend(**kw)
    raise ValueError(f"unknown backend '{name}' (builtin | carla)")
