from .incertidumbres import MedicionIndirecta, ProcesadorDatos

__all__ = [
    'MedicionIndirecta',
    'ProcesadorDatos',
]

def __incertidumbres_debug():
    # helper function so we don't import os globally
    import os
    debug_str = os.getenv('INCERTIDUMBRES_DEBUG', 'False')
    if debug_str in ('True', 'False'):
        return eval(debug_str)
    else:
        raise RuntimeError("unrecognized value for INCERTIDUMBRES_DEBUG: %s" %
                           debug_str)

INCERTIDUMBRES_DEBUG = __incertidumbres_debug()  # type: bool

# Clean up imports
import sys
del sys