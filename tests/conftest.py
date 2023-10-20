import pytest
import glob
import os

from tests.utils import HYPERPARAMETERS_RESULTS_PATH

import logging

logger = logging.getLogger(__name__)


@pytest.fixture(scope="function", autouse=True)
def cleanup_output_files():
    # Setup: Anything before the yield is the setup. You can leave it empty if there's no setup.

    yield  # This will allow the test to run.

    # Teardown: Anything after the yield is the teardown.
    for file in glob.glob("./resources/output_*"):
        if file == HYPERPARAMETERS_RESULTS_PATH:
            continue
        try:
            os.remove(file)
        except Exception as e:
            logger.error(f"Error deleting {file}. Error: {e}")
