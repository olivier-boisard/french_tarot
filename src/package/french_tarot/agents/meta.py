from functools import update_wrapper

from french_tarot.exceptions import FrenchTarotException

_singledispatch_registry = {}


def _get_first_arg_type(func_overload):
    return list(func_overload.__annotations__.values())[0]


def singledispatchmethod(func):
    """Provides method overloading.

    Manual implementation as it will be available with Python 3.8.
    """

    if func in _singledispatch_registry:
        raise FrenchTarotException("Function already declared as polymorphic")
    _singledispatch_registry[func] = {"default": func}

    def wrapper(*args, **kwargs):
        first_arg_type = type(args[1])
        default_func = _singledispatch_registry[func]["default"]
        func_to_call = _singledispatch_registry[func].get(first_arg_type, default_func)
        # noinspection PyArgumentList
        return func_to_call(*args, **kwargs)

    def register(func_overload):
        first_arg_type = _get_first_arg_type(func_overload)
        _singledispatch_registry[func][first_arg_type] = func_overload

    wrapper.register = register
    update_wrapper(wrapper, func)
    return wrapper
