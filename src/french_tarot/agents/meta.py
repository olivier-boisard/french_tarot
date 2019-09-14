class singledispatchmethod:
    """Provides method overloading.

    Manual implementation as it will be available with Python 3.8.
    """

    def __init__(self, func):
        self._default_function = func
        self._singledispatch_registry = {}

    def register(self, func_overload):
        first_arg_type = list(func_overload.__annotations__.value())[1]
        self._singledispatch_registry[first_arg_type] = func_overload

    def __call__(self, *args, **kwargs):
        first_arg_type = args[1]
        func_to_call = self._singledispatch_registry.get(first_arg_type, self._default_function)
        return func_to_call(*args, **kwargs)
