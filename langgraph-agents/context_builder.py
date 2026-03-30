# Campaign context builder. Loads game state, characters, NPCs, and session history
# for agent context injection. This data is pre-loaded into the Chronicler's system prompt
# so the agent has immediate access without tool calls for common queries.
# Deeper historical data is fetched on-demand via the Research Familiar's RAG tools.
"""
Context Builder for Chronicler Chat

This module provides campaign context loading for the Chronicler chat feature.
It loads campaign data from XML databases (NPCs, PCs, locations, quests, factions).

Note: Vector database querying is now done on-demand by the Research Familiar,
not upfront. This eliminates expensive auto-querying on every message.
"""

import logging
from typing import Dict

logger = logging.getLogger("ContextBuilder")


def filter_empty_fields(data):
    if isinstance(data, dict):
        return {k: filter_empty_fields(v) for k, v in data.items() if v}
    if isinstance(data, list):
        return [filter_empty_fields(item) for item in data if item]
    return data


class ContextBuilder:
    """
    Builds campaign context for the Chronicler agent.

    This class loads campaign data from XML databases, providing the Chronicler
    with basic knowledge about NPCs, PCs, locations, quests, and factions.

    For deeper historical context (session notes, transcripts), the Chronicler
    uses the Research Familiar tool on-demand.
    """

    def __init__(self):
        """Initialize context builder."""
        pass

    # Production: loads campaign XML files (game_system.xml, PC.xml, npcs.xml)
    # and parses them into structured dicts for the Chronicler's system prompt
    async def load_campaign_context(self, guild_id: int, campaign_id: str) -> Dict:
        """
        Load campaign context from XML databases.

        This loads:
        - Campaign info (name, game system, description)
        - Player characters (detailed data)
        - NPCs (detailed data)
        - Locations (detailed data)
        - Quests (all quests)
        - Factions (all factions)
        - World info

        Parameters:
        -----------
        guild_id: int
            Discord guild ID
        campaign_id: str
            Campaign ID

        Returns:
        --------
        Dict: Campaign context data
        """
        return {}

    # Alias for backward compatibility
    async def _load_campaign_context(self, guild_id: int, campaign_id: str) -> Dict:
        """Backward compatibility alias for load_campaign_context."""
        return await self.load_campaign_context(guild_id, campaign_id)

    # Production: reads session_notes.md, narrative_notes.txt, and detail_notes.txt
    # from the most recent N session directories for conversation context
    async def load_previous_sessions(
        self,
        guild_id: int,
        campaign_id: str,
        max_sessions: int = 4
    ) -> list:
        """
        Load previous session notes, narrative notes, and detail notes for context.

        Parameters:
        -----------
        guild_id: int
            Discord guild ID
        campaign_id: str
            Campaign ID
        max_sessions: int
            Maximum number of previous sessions to load (default: 4)

        Returns:
        --------
        list: List of dicts containing:
            {
                'session_id': str,
                'session_notes': str,
                'narrative_notes': str,
                'detail_notes': str
            }
        """
        return []
