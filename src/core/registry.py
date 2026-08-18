from src.core.errors import RegistryError


class Registry:
    def __init__(self, namespace):
        self.namespace = namespace
        self.entries = {}

    def register(self, name):
        def decorator(cls_or_fn):
            if name in self.entries:
                raise RegistryError(
                    f"'{name}' is already registered in namespace '{self.namespace}'."
                )
            self.entries[name] = cls_or_fn
            return cls_or_fn

        return decorator

    def get(self, name):
        if name not in self.entries:
            available = sorted(self.entries.keys())
            raise RegistryError(
                f"'{name}' is not registered in namespace '{self.namespace}'. "
                f"Available: {available}"
            )
        return self.entries[name]

    def build(self, name, *args, **params):
        target = self.get(name)
        return target(*args, **params)

    def keys(self):
        return sorted(self.entries.keys())


DATASETS = Registry("dataset")
TRANSFORMS = Registry("transform")
MODELS = Registry("model")
LOSSES = Registry("loss")
METRICS = Registry("metric")
ADAPTERS = Registry("adapter")
BUILDERS = Registry("builder")
