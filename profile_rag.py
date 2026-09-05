"""Profile one production RAG request with cProfile."""

import argparse
import cProfile
import pstats

from source.production_config import ProductionSettings
from source.production_runtime import ProductionRuntime


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("query", nargs="?", default="Khoản 2 Điều 5 quy định gì?")
    parser.add_argument("--output", default="rag-profile.prof")
    args = parser.parse_args()
    runtime = ProductionRuntime(ProductionSettings.from_env())
    runtime.start()
    try:
        profiler = cProfile.Profile()
        profiler.enable()
        assert runtime.service is not None
        runtime.service.answer(args.query)
        profiler.disable()
        profiler.dump_stats(args.output)
        pstats.Stats(profiler).sort_stats("cumulative").print_stats(30)
    finally:
        runtime.stop()


if __name__ == "__main__":
    main()