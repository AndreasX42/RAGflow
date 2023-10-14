from backend.commons.configurations.BaseConfigurations import (
    BaseConfigurations,
    CVGradeAnswerPrompt,
    CVGradeDocumentsPrompt,
    CVRetrieverSearchType,
    LLM_MODELS,
    EMB_MODELS,
)
from backend.commons.configurations.Hyperparameters import Hyperparameters
from backend.commons.configurations.QAConfigurations import QAConfigurations

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
