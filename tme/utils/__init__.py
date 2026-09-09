import importlib

_module_map = {
    ".subdivide": [
        "solve_subdivide",
    ],
    ".normal_field": [
        "compute_normal_field",
    ],
    ".serialization": [
        "serialize",
        "deserialize",
        "is_gzipped",
    ],
    ".logging": [
        "get_logger",
        "setup_logging",
        "debug_enabled",
    ],
    ".cli": [
        "match_template",
        "sanitize_name",
        "print_entry",
        "get_func_fullname",
        "print_block",
        "check_positive",
        "check_bounded_dtype",
        "existing_file",
        "load_and_validate_mask" "DeprecatedAction",
        "ArgumentMetadata",
    ],
}

_lazy_imports = {}
for module_path, functions in _module_map.items():
    for func_name in functions:
        _lazy_imports[func_name] = (module_path, func_name)


def __getattr__(name):
    module_path, attr_name = _lazy_imports.get(name, ("", ""))

    if not module_path:
        raise AttributeError(f"module {__name__!r} has no attribute {name!r}")

    mod = importlib.import_module(module_path, __name__)
    if attr_name:
        mod = getattr(mod, attr_name)

    globals()[name] = mod
    return mod
