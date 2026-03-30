# LangGraph state definitions for the Chronicler agent conversation loop.
# This TypedDict schema drives LangGraph's automatic state persistence via SQLite checkpointer,
# enabling conversation memory across messages and cross-agent communication.
"""
LangGraph State Schema for Chronicler Chat

This module defines the typed state for the Chronicler conversation graph.
LangGraph uses this schema to manage state persistence via SQLite checkpointer.
"""

from typing import Annotated, TypedDict, List, Dict, Optional

from langchain_core.messages import BaseMessage
from langgraph.graph.message import add_messages


class ChroniclerState(TypedDict):
    """
    State schema for the Chronicler conversation graph.

    This state is automatically persisted by LangGraph's checkpointer at every node,
    enabling conversation memory across messages and preserving agent-to-agent
    communication (e.g., Research Familiar results).
    """

    # Conversation history - managed automatically by LangGraph
    # The add_messages reducer appends new messages while handling duplicates
    messages: Annotated[List[BaseMessage], add_messages]

    # Campaign context - pre-loaded campaign data from XML databases
    # Includes: NPCs, PCs, locations, quests, factions, world info
    campaign_context: Dict

    # Campaign context loading timestamp - ISO 8601 format string
    # Used for smart context loading: only reload XML data if stale (>1 minute)
    # Example: "2025-10-13T14:30:00.123456"
    campaign_context_loaded_at: str

    # Campaign ID that the context was loaded for
    # Used to detect campaign switches and trigger context reload
    # Prevents stale context from wrong campaign bleeding into current session
    campaign_context_for_campaign_id: Optional[str]

    # Identity tracking
    guild_id: int
    campaign_id: str

    # Optional file attachment path for Discord messages
    # Set by chronicler_node when generate_art tool creates an image
    # Read by message_listener to attach image to Discord message
    attachment_path: Optional[str]

    # Optional session ID for reposting notes to Discord
    # Set by chronicler_node when edit_session_notes tool is called with operation='repost'
    # Read by message_listener to trigger posting notes to the notes channel
    repost_session_id: Optional[str]


# Type hints for convenience
StateType = ChroniclerState
