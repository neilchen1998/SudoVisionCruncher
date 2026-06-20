from collections.abc import Callable
from time import perf_counter
from typing import Any


class PipelineProfiler:
    """
    Records execution times for named stages in a processing pipeline

    Example:
        profiler = PipelineProfiler()

        model = profiler.profile("Load model", load_model, model_path)

        profiler.report()
    """

    def __init__(self) -> None:
        self.timings: dict[str, float] = {}

    def profile(self, name: str, fn: Callable[..., Any], *args, **kwargs) -> Any:
        """
        Executes a callable function and records its execution time

        Args:
            name: The human-readable name of the pipeline
            fn: The callable function
            *args: Positional arguments passed to the callable function
            **kwargs: Keyword arguments passed to the callable function

        Return:
            The return values from the callable function
        """

        start = perf_counter()
        result = fn(*args, **kwargs)
        self.timings[name] = perf_counter() - start
        return result

    def report(self):
        """
        Prints the summary report of recorded execution times
        """

        # Calculate the total time
        total = sum(self.timings.values())

        print("\n======= Pipeline Profile =======")

        for name, t in self.timings.items():
            pct = t / total * 100
            print(f"{name:<20}{t:>10.3f}s ({pct:>2.1f}%)")

        print(f"\tTotal time elapsed: {total:.3f}s")

    def get_report(self) -> dict[str, float]:
        """
        Gets a copy of the report

        Return:
            dict[str, flooat]: A dictionary with the function name as the key and its execution time as the value
        """

        return self.timings.copy()
