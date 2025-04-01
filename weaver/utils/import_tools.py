import os
import sys
import types
from importlib.util import spec_from_file_location, module_from_spec

def import_module(path, name='_mod'):
    path = os.path.abspath(path)
    parent_dir = os.path.dirname(path)
    package_name = os.path.basename(parent_dir)

    package_parent = os.path.dirname(parent_dir)
    if package_parent not in sys.path:
        sys.path.insert(0, package_parent)

    if package_name not in sys.modules:
        pkg = types.ModuleType(package_name)
        pkg.__path__ = [parent_dir]
        sys.modules[package_name] = pkg

    spec = spec_from_file_location(f"{package_name}.{name}", path)
    mod = module_from_spec(spec)
    mod.__package__ = package_name
    spec.loader.exec_module(mod)
    return mod
