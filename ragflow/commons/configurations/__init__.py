from ragflow.commons.configurations.BaseConfigurations import (
    BaseConfigurations,
    CVGradeAnswerPrompt,
    CVGradeRetrieverPrompt,
    CVRetrieverSearchType,
    CVSimilarityMethod,
    LLM_MODELS,
    EMB_MODELS,
)
from ragflow.commons.configurations.Hyperparameters import Hyperparameters
from ragflow.commons.configurations.QAConfigurations import QAConfigurations

import os
import dotenv
import logging

logger = logging.getLogger(__name__)

# Here we load .env file which contains the API keys to access LLMs, etc.
if os.environ.get("ENV") == "DEV":
    try:
        dotenv.load_dotenv(dotenv.find_dotenv(), override=True)
    except Exception as ex:
        logger.error(
            "Failed to load .env file in 'DEV' environment containing API keys."
        )
        raise ex
