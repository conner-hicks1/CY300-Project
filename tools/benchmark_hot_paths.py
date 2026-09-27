"""
Micro-benchmarks for per-frame hot paths found by the
profiler. Needs no OpenGL context.

    py -3.14 tools/benchmark_hot_paths.py

Prints microseconds per call (best of several runs).
"""

import sys
import timeit

from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import numpy as np  # noqa: E402

from ecs.components import (  # noqa: E402
    MeshRendererComponent,
    TransformComponent
)
from ecs.registry import Registry  # noqa: E402
from math3d.matrices import look_at, normal_matrix  # noqa: E402
from math3d.transform import Transform  # noqa: E402


def best_microseconds(statement, number=20000, repeat=5) -> float:

    return min(timeit.repeat(statement, number=number, repeat=repeat)) / number * 1e6


def main() -> int:

    model = Transform(position=(1, 2, 3), rotation=(10, 20, 30), scale=(2, 1, 3)).matrix

    registry = Registry()

    entities = []

    for _ in range(20):
        entity = registry.create()
        registry.add(entity, TransformComponent())
        registry.add(entity, MeshRendererComponent(mesh=None, material=None))
        entities.append(entity)

    def get_loop():
        for entity in registry.view(TransformComponent, MeshRendererComponent):
            registry.get(entity, TransformComponent)
            registry.get(entity, MeshRendererComponent)

    results = {
        "normal_matrix": best_microseconds(lambda: normal_matrix(model)),
        "look_at": best_microseconds(lambda: look_at((0.0, 1.2, 4.5), (0.0, 0.0, 0.0), (0.0, 1.0, 0.0))),
        "registry.get": best_microseconds(lambda: registry.get(entities[7], TransformComponent)),
        "view + 2x get (20 entities)": best_microseconds(get_loop, number=2000),
    }

    if hasattr(registry, "view_with"):

        def view_with_loop():
            for _entity, _transform, _renderer in registry.view_with(TransformComponent, MeshRendererComponent):
                pass

        results["view_with (20 entities)"] = best_microseconds(view_with_loop, number=2000)

    for name, microseconds in results.items():
        print(f"{name:<30}{microseconds:10.2f} us")

    return 0


if __name__ == "__main__":
    sys.exit(main())
