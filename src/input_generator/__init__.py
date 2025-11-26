import warnings
from mlcg_tk import input_generator as _new

warnings.simplefilter("once", DeprecationWarning)

warnings.warn(
    "You are importing 'input_generator' directly. "
    "This is deprecated and will be removed in a future release. "
    "Please update your imports to use 'mlcg_tk.input_generator'.",
    DeprecationWarning,
    stacklevel=2,
)

# re-export everything
__all__ = getattr(_new, "__all__", None)
for name in dir(_new):
    if not name.startswith("_"):
        globals()[name] = getattr(_new, name)
