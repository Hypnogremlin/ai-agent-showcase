# Multi-provider LLM configuration with per-role overrides and primary/backup fallback.
# Supports Anthropic, OpenAI, Google (Gemini), XAI (Grok), and Vertex AI.
# Separate from the LangChain chat config (chat_llm_config.py) — this produces
# CrewAI LLM instances for batch session processing pipelines.
"""
LLM Settings Utilities for Notes Crew

This module provides utilities for managing LLM settings across CrewAI crews,
including loading settings, switching between providers, and error detection.
"""

import json
import logging
import os
import time
from pathlib import Path
from typing import Dict, Any, Optional, Callable, Any as ReturnType

from crewai import LLM

# Configure logging
logger = logging.getLogger("NotesLLMConfig")

# Path to the LLM settings file
SETTINGS_FILE = Path(__file__).parent / "config" / "notes_llm_config.json"
DEFAULT_GEMINI_BASE_URL = "https://generativelanguage.googleapis.com/v1beta"
DEFAULT_GEMINI_MODEL = "gemini-2.5-flash"


def _resolve_google_api_key(config: Dict[str, Any]) -> Optional[str]:
    """Resolve the API key to use for Google Gemini requests."""
    # Check for service account JSON file path first
    credentials_path = os.getenv("GOOGLE_APPLICATION_CREDENTIALS")
    if credentials_path:
        return credentials_path

    # Fall back to direct API keys
    for key in (
        config.get("api_key"),
        os.getenv("GOOGLE_API_KEY"),
        os.getenv("GEMINI_API_KEY"),
        os.getenv("GOOGLE_GENAI_API_KEY"),
    ):
        if key:
            return key
    return None

def load_llm_settings() -> Dict[str, Any]:
    """
    Load current LLM settings from config/notes_llm_config.json

    Returns:
    --------
    Dict containing LLM settings, with fallback defaults if file doesn't exist
    """
    try:
        if SETTINGS_FILE.exists():
            with open(SETTINGS_FILE, 'r', encoding='utf-8') as f:
                settings = json.load(f)
            logger.debug(f"Loaded LLM settings from {SETTINGS_FILE}")
            return settings
        else:
            logger.warning(f"LLM settings file not found at {SETTINGS_FILE}, using defaults")
            return _get_default_settings()
    except Exception as e:
        logger.error(f"Error loading LLM settings: {e}, using defaults")
        return _get_default_settings()

def _get_default_settings() -> Dict[str, Any]:
    """Return default LLM settings if file doesn't exist or can't be loaded"""
    return {
        "primary": {
            "provider": "xai",
            "model": "grok-4-1-fast-non-reasoning",
            "temperature": 0.8,
            "max_output_tokens": 4096,
            "streaming": True
        },
        "backup": {
            "provider": "google",
            "model": DEFAULT_GEMINI_MODEL,
            "temperature": 0.8,
            "max_output_tokens": 4096,
            "streaming": True
        },
        "current_mode": "primary",
        "role_overrides": {}
    }

def save_llm_settings(settings: Dict[str, Any]) -> None:
    """
    Save LLM settings to the config file

    Parameters:
    -----------
    settings: Dict[str, Any]
        Settings dictionary to save
    """
    try:
        # Ensure directory exists
        SETTINGS_FILE.parent.mkdir(parents=True, exist_ok=True)

        with open(SETTINGS_FILE, 'w', encoding='utf-8') as f:
            json.dump(settings, f, indent=2)
        logger.debug(f"Saved LLM settings to {SETTINGS_FILE}")
    except Exception as e:
        logger.error(f"Error saving LLM settings: {e}")

def get_current_llm(profile: Optional[str] = None) -> LLM:
    """
    Return an LLM instance based on current settings.

    Parameters:
    -----------
    profile : str, optional
        LLM profile to use: "primary", "backup", or None (uses current_mode from config).

    Returns:
    --------
    LLM instance configured with current provider and model
    """
    settings = load_llm_settings()

    if profile is not None:
        resolved_profile = profile
    else:
        resolved_profile = settings.get("current_mode", "primary")

    if resolved_profile == "backup":
        config = settings["backup"]
        logger.info(f"Using backup LLM: {config['provider']} - {config['model']}")
    else:
        config = settings["primary"]
        logger.info(f"Using primary LLM: {config['provider']} - {config['model']}")

    logger.info(f"get_current_llm() called - Model: '{config['model']}', Provider: '{config['provider']}'")

    # Create LLM instance based on provider
    if config["provider"] == "anthropic":
        llm = LLM(
            model=config["model"],
            base_url=config.get("base_url", "https://api.anthropic.com")
        )
    elif config["provider"] == "openai":
        llm = LLM(
            model=config["model"]
            # OpenAI doesn't need base_url specified
        )
    elif config["provider"] == "vertex_ai":
        # Vertex AI configuration
        llm_kwargs: Dict[str, Any] = {
            "model": f"vertex_ai/{config['model']}",  # Add vertex_ai/ prefix
        }

        # Optional parameters
        for optional_param in (
            "temperature",
            "top_p",
            "top_k",
            "max_output_tokens",
            "presence_penalty",
            "frequency_penalty",
            "seed",
        ):
            if optional_param in config:
                llm_kwargs[optional_param] = config[optional_param]

        llm = LLM(**llm_kwargs)
        logger.info(
            "Initialized Vertex AI LLM with model '%s'",
            config["model"],
        )
    elif config["provider"] == "google":
        api_key = _resolve_google_api_key(config)
        if not api_key:
            logger.warning(
                "Google Gemini selected but no API key found. "
                "Set GOOGLE_API_KEY (preferred) or define api_key in notes_llm_config.json."
            )

        # For Google AI Studio, we need to use the correct endpoint format
        # LiteLLM requires GEMINI_API_KEY environment variable for AI Studio authentication
        model_name = config["model"]
        base_url = config.get("base_url", DEFAULT_GEMINI_BASE_URL)

        # If base_url doesn't include the model path, add it
        if "/models/" not in base_url:
            base_url = f"{base_url}/models/{model_name}"

        # Ensure GEMINI_API_KEY is set in environment (LiteLLM uses this for AI Studio)
        if api_key and not os.getenv("GEMINI_API_KEY"):
            os.environ["GEMINI_API_KEY"] = api_key
            logger.debug("Set GEMINI_API_KEY environment variable for LiteLLM")

        llm_kwargs: Dict[str, Any] = {
            "model": model_name,
            "base_url": base_url,
            # Don't pass api_key directly - LiteLLM uses GEMINI_API_KEY env var
        }

        api_base = config.get("api_base")
        if api_base:
            llm_kwargs["api_base"] = api_base

        for optional_param in (
            "temperature",
            "top_p",
            "top_k",
            "max_output_tokens",
            "presence_penalty",
            "frequency_penalty",
            "seed",
        ):
            if optional_param in config:
                llm_kwargs[optional_param] = config[optional_param]

        llm = LLM(**llm_kwargs)
        logger.info(
            "Initialized Google Gemini LLM with model '%s' via base_url '%s'",
            config["model"],
            llm_kwargs["base_url"],
        )
    elif config["provider"] == "xai":
        # XAI (Grok) configuration
        # CrewAI uses LiteLLM which supports xai/ prefix
        llm_kwargs: Dict[str, Any] = {
            "model": f"xai/{config['model']}",  # LiteLLM format
            "api_key": os.getenv("XAI_API_KEY"),
        }

        # Optional parameters
        for optional_param in (
            "temperature",
            "top_p",
            "max_output_tokens",
            "max_tokens",
        ):
            if optional_param in config:
                llm_kwargs[optional_param] = config[optional_param]

        # Streaming support (if CrewAI supports it)
        if "streaming" in config:
            llm_kwargs["stream"] = config["streaming"]

        llm = LLM(**llm_kwargs)
        logger.info(
            "Initialized XAI LLM with model '%s', streaming: %s",
            config["model"],
            config.get("streaming", False)
        )
    else:
        logger.warning(f"Unknown provider: {config['provider']}, defaulting to anthropic")
        llm = LLM(
            model="claude-3-5-haiku-20241022",
            base_url="https://api.anthropic.com"
        )

    # Production workaround: GPT-5 mini doesn't support the 'stop' parameter that CrewAI
    # sends by default. This monkey-patch intercepts LLM calls to strip incompatible params.
    logger.info(f"Checking if model '{config['model']}' needs stop parameter patch...")
    if config["model"] == "gpt-5-mini":
        logger.info("MATCH! Applying GPT-5 mini stop parameter patch")

        # Store original methods
        original_supports_stop_words = llm.supports_stop_words
        original_call = llm.call

        def patched_supports_stop_words():
            logger.info("patched_supports_stop_words() called - returning False for GPT-5 mini")
            return False

        def patched_call(messages, **kwargs):
            logger.info("patched_call() - intercepting LLM call to remove 'stop' parameter")
            # Remove stop parameter before calling original method
            if 'stop' in kwargs:
                logger.info("Removing 'stop' parameter from kwargs")
                kwargs.pop('stop')
            # Also set self.stop to None to prevent it being included in params
            original_stop = llm.stop
            llm.stop = None
            try:
                result = original_call(messages, **kwargs)
                return result
            finally:
                # Restore original stop value
                llm.stop = original_stop

        llm.supports_stop_words = patched_supports_stop_words
        llm.call = patched_call
        logger.info(f"✅ Successfully applied comprehensive stop parameter patch for {config['model']}")
    else:
        logger.info(f"No patch needed for model: {config['model']}")

    return llm


# Per-role LLM routing enables cost optimization: less critical roles (e.g., context_specialist)
# can use cheaper models while synthesis roles use premium models for quality.
def get_llm_for_role(role_name: str) -> LLM:
    """
    Return an LLM instance for a specific agent role.

    Checks role_overrides in config first. If the role has an override,
    uses that LLM profile. Otherwise falls back to get_current_llm()
    (which uses current_mode).

    Parameters:
    -----------
    role_name : str
        The agent role name (e.g., "context_specialist", "chronicler")

    Returns:
    --------
    LLM instance configured for this role
    """
    settings = load_llm_settings()
    role_overrides = settings.get("role_overrides", {})

    if role_name in role_overrides:
        profile = role_overrides[role_name]
        logger.info(f"Role override for '{role_name}': using '{profile}' profile")
        return get_current_llm(profile=profile)

    logger.debug(f"No override for role '{role_name}': using current_mode fallback")
    return get_current_llm()


def switch_to_backup_mode() -> None:
    """
    Temporarily switch current_mode to backup for this run
    This change will persist until reset_to_primary_mode() is called
    """
    settings = load_llm_settings()
    settings["current_mode"] = "backup"
    save_llm_settings(settings)
    logger.info("Switched to backup LLM mode")

def reset_to_primary_mode() -> None:
    """
    Reset back to primary mode (typically called at start of new sessions)
    """
    settings = load_llm_settings()
    settings["current_mode"] = "primary"
    save_llm_settings(settings)
    logger.debug("Reset to primary LLM mode")

def is_anthropic_overloaded_error(exception: Exception) -> bool:
    """
    Check if this is the specific Anthropic overloaded error we want to handle

    Parameters:
    -----------
    exception: Exception
        The exception to check

    Returns:
    --------
    bool: True if this is an Anthropic overloaded error
    """
    exception_str = str(exception).lower()
    exception_type = str(type(exception)).lower()

    return (
        ("anthropicerror" in exception_type or "anthropic" in exception_type) and
        ("overloaded" in exception_str)
    )

def is_anthropic_rate_limit_error(exception: Exception) -> bool:
    """
    Check if this is an Anthropic rate limit error that we should also handle

    Parameters:
    -----------
    exception: Exception
        The exception to check

    Returns:
    --------
    bool: True if this is an Anthropic rate limit error
    """
    exception_str = str(exception).lower()
    exception_type = str(type(exception)).lower()

    return (
        ("anthropicerror" in exception_type or "anthropic" in exception_type) and
        ("rate" in exception_str and "limit" in exception_str)
    )

def should_switch_to_backup(exception: Exception) -> bool:
    """
    Determine if we should switch to backup LLM based on the exception

    Catches errors from Anthropic, Gemini, Vertex, Google, and XAI providers

    Parameters:
    -----------
    exception: Exception
        The exception to evaluate

    Returns:
    --------
    bool: True if we should switch to backup LLM
    """
    exception_str = str(exception).lower()
    exception_type = str(type(exception)).lower()

    # Check for XAI errors
    xai_errors = ["rate limit", "quota exceeded", "xai error", "authentication"]
    if "xai" in exception_str or any(err in exception_str for err in xai_errors):
        logger.warning(f"XAI error detected: {exception}")
        return True

    # Check for other provider errors
    if any(
        provider_keyword in exception_type or provider_keyword in exception_str
        for provider_keyword in ("anthropic", "gemini", "vertex", "google")
    ):
        return True

    return False

def get_current_provider_info() -> Dict[str, str]:
    """
    Get information about the currently active LLM provider

    Returns:
    --------
    Dict with provider, model, and mode information
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

def get_current_notes_model() -> str:
    """
    Return the current primary model name string for notes processing.

    Used by cost_logger to identify which model was used during session processing.

    Returns:
    --------
    str: Model name (e.g. "grok-4-1-fast-non-reasoning", "gemini-2.5-flash")
    """
    settings = load_llm_settings()
    mode = settings.get("current_mode", "primary")
    config = settings["backup"] if mode == "backup" else settings["primary"]
    return config.get("model", "")


# Three-tier retry strategy: try primary -> wait 60s and retry primary -> switch to backup provider.
# Each retry creates a fresh crew instance so @before_kickoff re-assigns LLMs from current config.
def with_llm_fallback(crew_factory: Callable, operation_func: Callable, operation_name: str) -> ReturnType:
    """
    Execute a CrewAI operation with LLM fallback logic.

    This wrapper implements a three-tier retry strategy:
    1. Primary attempt with current LLM
    2. Wait 60 seconds, retry with primary LLM
    3. Switch to backup LLM mode and retry with fresh crew instance

    Parameters:
    -----------
    crew_factory: Callable
        Function that returns a fresh crew instance (e.g., lambda: DnDSessionCrew())
    operation_func: Callable
        Function that takes a crew and executes the operation (e.g., lambda crew: crew.kickoff(data))
    operation_name: str
        Human-readable operation name for logging

    Returns:
    --------
    Result from the operation function

    Raises:
    -------
    Exception: If all retry attempts fail
    """
    try:
        # Primary attempt
        crew = crew_factory()
        result = operation_func(crew)
        logger.debug(f"{operation_name} completed successfully with primary LLM")
        return result

    except Exception as e:
        # Check if this is an error we should handle
        if should_switch_to_backup(e):
            logger.warning(f"LLM API error during {operation_name}: {str(e)}")
            logger.info("Waiting 1 minute before retrying with primary LLM...")

            # Wait 1 minute to give provider time to recover
            time.sleep(60)

            try:
                # Retry with primary LLM first
                logger.info(f"Retrying {operation_name} with primary LLM after wait...")
                retry_crew = crew_factory()
                result = operation_func(retry_crew)
                logger.info(f"{operation_name} completed successfully on retry with primary LLM")
                return result

            except Exception as retry_error:
                # Primary LLM failed again, now switch to backup
                logger.warning(f"Primary LLM failed again after retry: {str(retry_error)}")
                logger.info(f"Now switching to backup LLM for {operation_name}...")

                try:
                    # Switch to backup mode and try immediately
                    switch_to_backup_mode()

                    # Create new crew instance (will load backup LLM via @before_kickoff)
                    backup_crew = crew_factory()
                    provider_info = get_current_provider_info()
                    logger.info(f"Trying {operation_name} with backup LLM: {provider_info['provider']} - {provider_info['model']}")

                    result = operation_func(backup_crew)
                    logger.info(f"{operation_name} completed successfully with backup LLM")
                    return result

                except Exception as backup_error:
                    logger.error(f"Backup LLM also failed: {str(backup_error)}")
                    raise Exception(f"All LLM attempts failed for {operation_name}. Primary: {str(e)}, Retry: {str(retry_error)}, Backup: {str(backup_error)}")
        else:
            logger.error(f"Non-recoverable error during {operation_name}: {str(e)}")
            raise
