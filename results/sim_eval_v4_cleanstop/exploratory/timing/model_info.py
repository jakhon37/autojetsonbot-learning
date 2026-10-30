"""Architecture and size of the BC ONNX models: op count and types, parameters, I/O shapes.

    python model_info.py MODEL.onnx [MODEL.onnx ...]

Two models have the same architecture here when their node op-type sequences and their
initializer shapes (in graph order) are identical; the weights may differ.
"""

import hashlib
import json
import sys
from collections import Counter

import numpy as np
import onnx


def dims(value_info):
    return [d.dim_value if d.HasField("dim_value") else (d.dim_param or "?") for d in value_info.type.tensor_type.shape.dim]


def info(path):
    model = onnx.load(path)
    g = model.graph
    init_names = {i.name for i in g.initializer}
    params = sum(int(np.prod(i.dims)) for i in g.initializer)
    ops = [n.op_type for n in g.node]
    sig = hashlib.sha256(
        json.dumps([ops, [(list(i.dims), i.data_type) for i in g.initializer]]).encode()
    ).hexdigest()[:16]
    with open(path, "rb") as f:
        sha = hashlib.sha256(f.read()).hexdigest()
    return {
        "path": path,
        "sha256": sha,
        "bytes": len(open(path, "rb").read()),
        "opset": [(o.domain or "ai.onnx", o.version) for o in model.opset_import],
        "producer": "%s %s" % (model.producer_name, model.producer_version),
        "inputs": [(i.name, dims(i)) for i in g.input if i.name not in init_names],
        "outputs": [(o.name, dims(o)) for o in g.output],
        "nodes": len(ops),
        "op_types": dict(sorted(Counter(ops).items())),
        "initializers": len(g.initializer),
        "parameters": params,
        "arch_signature": sig,
    }


def main(paths):
    rows = [info(p) for p in paths]
    for r in rows:
        print(json.dumps(r))
    return 0


if __name__ == "__main__":
    sys.exit(main(sys.argv[1:]))
