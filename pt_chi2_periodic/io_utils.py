"""io_utils.py -- dump results to compact JSON (complex -> [re, im], floats rounded)."""
import json
import numpy as np


def _clean(o, nd):
    if isinstance(o, dict):
        return {str(k): _clean(v, nd) for k, v in o.items()}
    if isinstance(o, (list, tuple)):
        return [_clean(v, nd) for v in o]
    if isinstance(o, np.ndarray):
        return _clean(o.tolist(), nd)
    if isinstance(o, (complex, np.complexfloating)):
        return [_clean(float(o.real), nd), _clean(float(o.imag), nd)]
    if isinstance(o, (float, np.floating)):
        v = float(o)
        if not np.isfinite(v):
            return None
        return float(f"{v:.{nd}g}")
    if isinstance(o, (np.integer,)):
        return int(o)
    if isinstance(o, (np.bool_,)):
        return bool(o)
    return o


def dump(path, obj, nd=6):
    with open(path, 'w') as f:
        json.dump(_clean(obj, nd), f, separators=(',', ':'))


def load(path):
    with open(path) as f:
        return json.load(f)
