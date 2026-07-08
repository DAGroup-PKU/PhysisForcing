from typing import Any

from fvcore.common.registry import Registry as _Registry


class Registry(_Registry):
    """
    1. Skip duplicate registrations.
    2. Support .register(name, obj)
    """
    def _do_register(self, name: str, obj: Any) -> None:
        if name in self._obj_map:
            return
        self._obj_map[name] = obj

    def register(self, *args) -> Any:
        if len(args) == 0:
            return super().register()
        elif len(args) == 1:
            obj = args[0]
            return super().register(obj)
        elif len(args) == 2:
            name, obj = args
            self._do_register(name, obj)
        else:
            raise ValueError("Cannot provide more than two args")
