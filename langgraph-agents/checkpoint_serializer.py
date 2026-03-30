# Custom checkpoint serializer that strips base64 image data from conversation state
# before persistence. This prevents checkpoint DB bloat from inline images that Discord
# users upload in chat. The LLM still sees full images during processing — stripping
# happens only at the storage layer after the response is generated.
"""
Custom Checkpoint Serializer for Base64 Image Stripping

This module provides a custom serializer that strips base64 image data from
checkpoint state before persisting to SQLite, preventing database bloat.

The serializer intercepts at the STORAGE layer, meaning:
- LLM always receives full base64 images for vision processing
- Only what gets written to disk is stripped
- Placeholder metadata preserves context that an image was present
"""

import logging
from typing import Any, Tuple

from langgraph.checkpoint.serde.jsonplus import JsonPlusSerializer

logger = logging.getLogger("CheckpointSerializer")

# Placeholder URL to replace base64 data
STRIPPED_IMAGE_PLACEHOLDER = "[image stripped for storage]"


def strip_base64_from_object(obj: Any) -> Any:
    """
    Recursively strip base64 image data from checkpoint state.

    This function handles the nested structure of LangGraph checkpoint state,
    which contains messages with content blocks that may include base64 images.

    Parameters:
    -----------
    obj : Any
        The object to process (dict, list, or primitive)

    Returns:
    --------
    Any: Deep copy with base64 data replaced by placeholders
    """
    if obj is None:
        return None

    if isinstance(obj, dict):
        result = {}
        for key, value in obj.items():
            # Check for image_url blocks with base64 data
            if key == "image_url" and isinstance(value, dict):
                url = value.get("url", "")
                if isinstance(url, str) and url.startswith("data:image/"):
                    # Calculate size for logging
                    size_kb = len(url) / 1024
                    logger.info(f"CheckpointSerializer: Stripping base64 image ({size_kb:.1f} KB) from checkpoint")

                    # Replace with placeholder, preserve other fields
                    result[key] = {
                        **strip_base64_from_object(value),
                        "url": STRIPPED_IMAGE_PLACEHOLDER,
                        "was_base64": True,
                    }
                else:
                    result[key] = strip_base64_from_object(value)
            else:
                result[key] = strip_base64_from_object(value)
        return result

    elif isinstance(obj, list):
        return [strip_base64_from_object(item) for item in obj]

    elif isinstance(obj, tuple):
        return tuple(strip_base64_from_object(item) for item in obj)

    # Handle LangChain message objects (they have a content attribute)
    elif hasattr(obj, "content") and hasattr(obj, "copy"):
        # This is likely a LangChain BaseMessage subclass
        content = getattr(obj, "content", None)
        if content is not None:
            stripped_content = strip_base64_from_object(content)
            if stripped_content != content:
                # Content was modified, create new message with stripped content
                try:
                    new_obj = obj.copy(deep=False)
                    new_obj.content = stripped_content
                    return new_obj
                except Exception as e:
                    logger.debug(f"Could not copy message object: {e}")
                    return obj
        return obj

    # Primitives (str, int, float, bool, bytes) pass through unchanged
    return obj


class Base64StrippingSerializer(JsonPlusSerializer):
    """
    Custom serializer that strips base64 image data before checkpoint storage.

    Extends JsonPlusSerializer to intercept serialization and replace base64
    image data with placeholders. This prevents database bloat while:
    - Preserving full images for LLM processing (serialization happens AFTER processing)
    - Maintaining conversation context (placeholders show an image was sent)
    - Being transparent to the rest of the system

    IMPORTANT: We override dumps_typed() because that's what AsyncSqliteSaver calls,
    not dumps(). The checkpointer uses: self.serde.dumps_typed(checkpoint)

    Usage:
    ------
    ```python
    from .checkpoint_serializer import Base64StrippingSerializer
    checkpointer = AsyncSqliteSaver(conn, serde=Base64StrippingSerializer())
    ```
    """

    def dumps_typed(self, obj: Any) -> Tuple[str, bytes]:
        """
        Serialize object to (type, bytes) after stripping base64 image data.

        This is the primary method called by AsyncSqliteSaver for checkpoint storage.
        We intercept here to strip base64 before any serialization format is applied.

        Parameters:
        -----------
        obj : Any
            The checkpoint state to serialize

        Returns:
        --------
        Tuple[str, bytes]: (type_string, serialized_data) with base64 stripped
        """
        # Strip base64 from all nested structures BEFORE serialization
        stripped = strip_base64_from_object(obj)
        return super().dumps_typed(stripped)

    def dumps(self, obj: Any) -> bytes:
        """
        Serialize object to bytes after stripping base64 image data.

        This is a fallback method that may be called in some code paths.

        Parameters:
        -----------
        obj : Any
            The checkpoint state to serialize

        Returns:
        --------
        bytes: Serialized state with base64 images replaced by placeholders
        """
        stripped = strip_base64_from_object(obj)
        return super().dumps(stripped)

    def loads(self, data: bytes) -> Any:
        """
        Deserialize bytes to object.

        No special processing needed for loading - placeholders are valid JSON.

        Parameters:
        -----------
        data : bytes
            Serialized checkpoint data

        Returns:
        --------
        Any: Deserialized checkpoint state
        """
        return super().loads(data)
