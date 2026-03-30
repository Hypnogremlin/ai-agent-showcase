# Multi-provider LangChain LLM factory with primary/backup fallback.
# Mirrors the CrewAI LLM config (llm_config.py) but produces LangChain ChatModel instances.
# Architectural choice: separate configs for CrewAI (batch processing) vs LangChain (interactive chat)
# because the two frameworks have different model interfaces and parameter conventions.
"""
LLM Configuration for Chronicler Chat

This module manages LLM settings for the Chronicler chat agent with
LangChain's chat model interface.
"""

import json
import logging
import os
from pathlib import Path

from langchain_anthropic import ChatAnthropic
from langchain_openai import ChatOpenAI
from langchain_core.language_models import BaseChatModel

try:
    from langchain_google_genai import ChatGoogleGenerativeAI
except ImportError:  # pragma: no cover - optional dependency
    ChatGoogleGenerativeAI = None

try:
    from langchain_google_vertexai import ChatVertexAI
except ImportError:  # pragma: no cover - optional dependency
    ChatVertexAI = None

try:
    from langchain_xai import ChatXAI
except ImportError:  # pragma: no cover - optional dependency
    ChatXAI = None

logger = logging.getLogger("ChroniclerLLM")

# Configuration file path
LLM_CONFIG_PATH = Path(__file__).parent / "chat_llm_config.json"


def load_llm_settings() -> dict:
    """
    Load LLM settings from the chat configuration file.

    Returns:
    --------
    dict: Configuration dictionary with primary, backup, and current_mode
    """
    if not LLM_CONFIG_PATH.exists():
        logger.error(f"LLM config file not found: {LLM_CONFIG_PATH}")
        raise FileNotFoundError(f"LLM config file not found: {LLM_CONFIG_PATH}")

    with open(LLM_CONFIG_PATH, 'r', encoding='utf-8') as f:
        settings = json.load(f)

    return settings


def get_chronicler_llm(temperature: float = 0.7, max_tokens: int = 1500) -> BaseChatModel:
    """
    Get a LangChain-compatible chat model based on chat LLM settings.

    This function reads from chat_llm_config.json for Chronicler chat-specific
    LLM configuration, separate from the notes crew configuration.

    Parameters:
    -----------
    temperature: float
        Temperature setting for response creativity (default: 0.7)
    max_tokens: int
        Maximum tokens for response length (default: 1500)

    Returns:
    --------
    BaseChatModel: LangChain chat model configured for Chronicler chat
    """
    # Load settings from chat config
    settings = load_llm_settings()
    mode = settings.get("current_mode", "primary")

    # Select configuration based on mode
    if mode == "backup":
        config = settings["backup"]
        logger.info(f"Using backup LLM for Chronicler: {config['provider']} - {config['model']}")
    else:
        config = settings["primary"]
        logger.info(f"Using primary LLM for Chronicler: {config['provider']} - {config['model']}")

    config_temperature = config.get("temperature", temperature)
    config_max_tokens = config.get("max_output_tokens", config.get("max_tokens", max_tokens))

    # Create appropriate LangChain model based on provider
    try:
        if config["provider"] == "anthropic":
            llm = ChatAnthropic(
                model=config["model"],
                temperature=config_temperature,
                max_tokens=config_max_tokens,
            )
            logger.debug(f"Created ChatAnthropic with model: {config['model']}")

        elif config["provider"] == "openai":
            llm = ChatOpenAI(
                model=config["model"],
                temperature=config_temperature,
                max_tokens=config_max_tokens,
            )
            logger.debug(f"Created ChatOpenAI with model: {config['model']}")

        elif config["provider"] == "google":
            if ChatGoogleGenerativeAI is None:
                raise ImportError(
                    "langchain-google-genai is required for Google Gemini models. "
                    "Install via `pip install langchain-google-genai`."
                )

            model_name = config["model"]
            if "/" in model_name:
                _, model_name = model_name.split("/", 1)

            llm = ChatGoogleGenerativeAI(
                model=model_name,
                temperature=config_temperature,
                max_output_tokens=config_max_tokens,
            )
            logger.debug(f"Created ChatGoogleGenerativeAI with model: {model_name}")

        elif config["provider"] == "vertex_ai":
            if ChatVertexAI is None:
                raise ImportError(
                    "langchain-google-vertexai is required for Vertex AI models. "
                    "Install via `pip install langchain-google-vertexai`."
                )

            llm = ChatVertexAI(
                model=config["model"],
                temperature=config_temperature,
                max_output_tokens=config_max_tokens,
            )
            logger.debug(f"Created ChatVertexAI with model: {config['model']}")

        elif config["provider"] == "xai":
            if ChatXAI is None:
                raise ImportError(
                    "langchain-xai is required for XAI Grok models. "
                    "Install via `pip install langchain-xai`."
                )

            llm = ChatXAI(
                model=config["model"],
                temperature=config_temperature,
                max_tokens=config_max_tokens,
                streaming=config.get("streaming", True),
                stream_usage=True,
                timeout=None,
                max_retries=2,
            )
            logger.debug(f"Created ChatXAI with model: {config['model']}, streaming enabled")

        else:
            # Fallback to primary default if unknown provider
            logger.warning(f"Unknown provider: {config['provider']}, using primary default")
            llm = ChatAnthropic(
                model="claude-3-5-haiku-20241022",
                temperature=config_temperature,
                max_tokens=config_max_tokens,
            )

        return llm

    except Exception as e:
        logger.error(f"Error creating LLM: {e}, attempting backup provider")

        backup_config = settings.get("backup")
        if not backup_config:
            logger.error("Backup LLM configuration missing, cannot recover")
            raise

        backup_provider = backup_config.get("provider")
        backup_temperature = backup_config.get("temperature", temperature)
        backup_max_tokens = backup_config.get("max_output_tokens", backup_config.get("max_tokens", max_tokens))

        try:
            if backup_provider == "openai":
                logger.info(f"Falling back to OpenAI backup model: {backup_config['model']}")
                return ChatOpenAI(
                    model=backup_config["model"],
                    temperature=backup_temperature,
                    max_tokens=backup_max_tokens,
                    )
            elif backup_provider == "anthropic":
                logger.info(f"Falling back to Anthropic backup model: {backup_config['model']}")
                return ChatAnthropic(
                    model=backup_config["model"],
                    temperature=backup_temperature,
                    max_tokens=backup_max_tokens,
                    )
            elif backup_provider == "google":
                if ChatGoogleGenerativeAI is None:
                    raise ImportError(
                        "langchain-google-genai is required for Google Gemini models. "
                        "Install via `pip install langchain-google-genai`."
                    )
                logger.info(f"Falling back to Google backup model: {backup_config['model']}")
                backup_model_name = backup_config["model"]
                if "/" in backup_model_name:
                    _, backup_model_name = backup_model_name.split("/", 1)
                return ChatGoogleGenerativeAI(
                    model=backup_model_name,
                    temperature=backup_temperature,
                    max_output_tokens=backup_max_tokens,
                    )
            elif backup_provider == "vertex_ai":
                if ChatVertexAI is None:
                    raise ImportError(
                        "langchain-google-vertexai is required for Vertex AI models. "
                        "Install via `pip install langchain-google-vertexai`."
                    )
                logger.info(f"Falling back to Vertex AI backup model: {backup_config['model']}")
                return ChatVertexAI(
                    model=backup_config["model"],
                    temperature=backup_temperature,
                    max_output_tokens=backup_max_tokens,
                    )
            elif backup_provider == "xai":
                if ChatXAI is None:
                    raise ImportError(
                        "langchain-xai is required for XAI Grok models. "
                        "Install via `pip install langchain-xai`."
                    )
                logger.info(f"Falling back to XAI backup model: {backup_config['model']}")
                return ChatXAI(
                    model=backup_config["model"],
                    temperature=backup_temperature,
                    max_tokens=backup_max_tokens,
                    streaming=backup_config.get("streaming", True),
                    stream_usage=True,
                    )
            else:
                raise ValueError(f"Unsupported backup provider: {backup_provider}")
        except Exception as backup_error:
            logger.error(f"Failed to initialize backup LLM: {backup_error}")
            raise


def get_current_llm_info() -> dict:
    """
    Get information about the currently configured LLM.

    Returns:
    --------
    dict: Dictionary with mode, provider, and model information
    """
    settings = load_llm_settings()
    mode = settings.get("current_mode", "primary")

    if mode == "backup":
        config = settings["backup"]
    else:
        config = settings["primary"]

    return {
        "mode": mode,
        "provider": config["provider"],
        "model": config["model"]
    }
