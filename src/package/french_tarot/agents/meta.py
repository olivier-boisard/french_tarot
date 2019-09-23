from functools import update_wrapper

from french_tarot.exceptions import FrenchTarotException

_singledispatch_registry = {}

DEFAULT = "default"


def singledispatchmethod(func):
    """Provides method overloading.

    Manual implementation as it will be available with Python 3.8.
    """

    if func in _singledispatch_registry:
        raise FrenchTarotException("Function already declared as polymorphic")
    _singledispatch_registry[func] = {DEFAULT: func}

    def wrapper(*args, **kwargs):
        arg_type = type(args[1])
        registry = _singledispatch_registry[func]

        func_to_call = _determine_func_to_call(arg_type, registry)
        _add_type_to_registry_if_not_there(arg_type, func_to_call, registry)

        # noinspection PyArgumentList
        return func_to_call(*args, **kwargs)

    def register(func_overload):
        first_arg_type = _get_first_arg_type(func_overload)
        _singledispatch_registry[func][first_arg_type] = func_overload

    wrapper.register = register
    update_wrapper(wrapper, func)
    return wrapper


def _get_first_arg_type(func_overload):
    return list(func_overload.__annotations__.values())[0]


def _determine_func_to_call(arg_type, registry):
    func_to_call = _recursively_determine_func_to_call([arg_type], registry)
    if func_to_call is None:
        func_to_call = registry[DEFAULT]
    return func_to_call


def _add_type_to_registry_if_not_there(arg_type, func_to_call, registry):
    if arg_type not in registry:
        registry[arg_type] = func_to_call


def _recursively_determine_func_to_call(arg_types: type, registry: dict):
    if len(arg_types) == 0:
        func_to_call = None
    else:
        func_to_call = None
        new_arg_types = []
        for arg_type in arg_types:
            func_to_call = registry.get(arg_type)
            if func_to_call is not None:
                break
            new_arg_types.extend(arg_type.__bases__)

        if func_to_call is None:
            func_to_call = _recursively_determine_func_to_call(new_arg_types, registry)
    return func_to_call
