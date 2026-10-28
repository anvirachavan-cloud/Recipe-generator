"""
IBM Granite LLM Integration Module
Wraps IBM watsonx.ai API for text generation using the Granite model.
"""

import os
import logging
from typing import Optional, Dict, Any

from ibm_watsonx_ai import APIClient, Credentials
from ibm_watsonx_ai.foundation_models import ModelInference
from ibm_watsonx_ai.metanames import GenTextParamsMetaNames as GenParams
from langchain_ibm import WatsonxLLM

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# ── IBM watsonx.ai configuration ────────────────────────────────────────────
IBM_API_KEY = os.getenv("IBM_API_KEY", "z7zQ2T0jAwU2vuKNrLTNqxQCbiYujxGGNC2UWezhql7O")
IBM_PROJECT_ID = os.getenv("IBM_PROJECT_ID", "f333d587-8212-4756-b9fc-657dd1c42612")
IBM_ENDPOINT_URL = os.getenv("IBM_ENDPOINT_URL", "https://au-syd.ml.cloud.ibm.com")

# IBM Granite & IBM-hosted models for the au-syd region.
# granite-3-1-8b-base: IBM Granite available in au-syd.
# llama-3-3-70b-instruct: Best instruction-following model available in au-syd for RAG generation.
GRANITE_MODEL_ID = "ibm/granite-3-1-8b-base"
GRANITE_CHAT_MODEL_ID = "meta-llama/llama-3-3-70b-instruct"

# Default generation parameters
DEFAULT_PARAMS: Dict[str, Any] = {
    GenParams.MAX_NEW_TOKENS: 1024,
    GenParams.MIN_NEW_TOKENS: 50,
    GenParams.TEMPERATURE: 0.7,
    GenParams.TOP_P: 0.9,
    GenParams.TOP_K: 50,
    GenParams.REPETITION_PENALTY: 1.1,
    GenParams.STOP_SEQUENCES: ["Human:", "User:", "\n\nHuman", "---END---"],
}

RECIPE_PARAMS: Dict[str, Any] = {
    GenParams.MAX_NEW_TOKENS: 1500,
    GenParams.MIN_NEW_TOKENS: 100,
    GenParams.TEMPERATURE: 0.75,
    GenParams.TOP_P: 0.92,
    GenParams.TOP_K: 60,
    GenParams.REPETITION_PENALTY: 1.15,
    GenParams.STOP_SEQUENCES: ["Human:", "User:", "---END---"],
}

QA_PARAMS: Dict[str, Any] = {
    GenParams.MAX_NEW_TOKENS: 800,
    GenParams.MIN_NEW_TOKENS: 30,
    GenParams.TEMPERATURE: 0.3,
    GenParams.TOP_P: 0.85,
    GenParams.TOP_K: 40,
    GenParams.REPETITION_PENALTY: 1.1,
    GenParams.STOP_SEQUENCES: ["Human:", "User:", "---END---"],
}


def get_watsonx_credentials() -> Credentials:
    """Build IBM watsonx.ai credentials object."""
    return Credentials(
        url=IBM_ENDPOINT_URL,
        api_key=IBM_API_KEY,
    )


def get_granite_llm(
    model_id: str = GRANITE_CHAT_MODEL_ID,
    params: Optional[Dict[str, Any]] = None,
) -> WatsonxLLM:
    """
    Return a LangChain-compatible WatsonxLLM using IBM Granite.
    
    Args:
        model_id: The IBM Granite model ID to use.
        params: Override generation parameters.
    
    Returns:
        WatsonxLLM instance ready for LangChain chains.
    """
    generation_params = params or DEFAULT_PARAMS
    logger.info("Initializing IBM Granite LLM: %s", model_id)

    llm = WatsonxLLM(
        model_id=model_id,
        url=IBM_ENDPOINT_URL,
        apikey=IBM_API_KEY,
        project_id=IBM_PROJECT_ID,
        params=generation_params,
    )
    logger.info("IBM Granite LLM initialized successfully.")
    return llm


def get_recipe_llm() -> WatsonxLLM:
    """Return an LLM tuned for creative recipe generation."""
    return get_granite_llm(model_id=GRANITE_CHAT_MODEL_ID, params=RECIPE_PARAMS)


def get_qa_llm() -> WatsonxLLM:
    """Return an LLM tuned for precise Q&A answering."""
    return get_granite_llm(model_id=GRANITE_CHAT_MODEL_ID, params=QA_PARAMS)


def direct_generate(
    prompt: str,
    model_id: str = GRANITE_CHAT_MODEL_ID,
    params: Optional[Dict[str, Any]] = None,
) -> str:
    """
    Directly call the IBM watsonx.ai model without LangChain.
    Useful for health checks or simple one-shot generations.
    """
    creds = get_watsonx_credentials()
    model = ModelInference(
        model_id=model_id,
        credentials=creds,
        project_id=IBM_PROJECT_ID,
        params=params or DEFAULT_PARAMS,
    )
    response = model.generate_text(prompt=prompt)
    return response


def health_check() -> Dict[str, Any]:
    """Verify IBM watsonx.ai connectivity."""
    try:
        result = direct_generate(
            "Say 'OK' if you are working.",
            params={
                GenParams.MAX_NEW_TOKENS: 10,
                GenParams.TEMPERATURE: 0.1,
            },
        )
        return {"status": "healthy", "model": GRANITE_CHAT_MODEL_ID, "response": result}
    except Exception as exc:
        logger.error("IBM Granite health check failed: %s", exc)
        return {"status": "unhealthy", "error": str(exc)}
