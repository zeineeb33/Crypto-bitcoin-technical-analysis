"""Process-wide runtime setup shared by the CLI and the API."""

import sklearn.ensemble  # noqa: F401 - load the OpenMP/BLAS runtimes so they can be capped
from threadpoolctl import threadpool_limits

from cryptosignal.config import Settings
from cryptosignal.logging_utils import setup_logging


def configure_runtime(settings: Settings) -> None:
    """Configure logging and cap native thread pools.

    Datasets here are a few thousand rows: OpenMP/BLAS thread start-up costs more than
    it saves (8 threads made a logistic-regression fit ~8x slower than 1), and the
    target container only has a fraction of one CPU anyway.
    """
    setup_logging(settings.log_level)
    threadpool_limits(limits=settings.threads)
