import importlib
import inspect


def load_model_class(identifier: str, prefix: str = "src.models."):
    module_path, class_name = identifier.split('@')

    # Import the module
    module = importlib.import_module(prefix + module_path)
    cls = getattr(module, class_name)
    
    return cls