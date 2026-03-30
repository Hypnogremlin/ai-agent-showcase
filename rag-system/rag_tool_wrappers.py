# Bridge layer wrapping CrewAI RagTools as LangChain @tool functions.
# Enables the same RAG backend to serve both CrewAI crews (batch processing)
# and LangGraph agents (interactive chat). This is necessary because CrewAI and
# LangChain have different tool interfaces — CrewAI uses BaseTool._run() while
# LangChain expects decorated functions or StructuredTools.
"""
RAG Tool Wrappers for LangChain Compatibility

This module wraps CrewAI RagTools as LangChain tools, enabling them to work
with LangChain's create_openai_functions_agent. The wrappers call the CrewAI
tools' internal _run() methods while presenting a LangChain-compatible interface.
"""

import logging
from typing import List

from langchain_core.tools import tool

from .rag_utils import create_campaign_rag_tools

logger = logging.getLogger("RAGToolWrappers")


def create_langchain_rag_tools(guild_id: int, campaign_id: str) -> List:
    """
    Create LangChain-compatible RAG tools by wrapping CrewAI RagTools.

    This function creates the three campaign-specific CrewAI RagTools
    (narratives, details, transcripts) and wraps them as LangChain tools
    using the @tool decorator pattern.

    Parameters:
    -----------
    guild_id: int
        Discord guild ID
    campaign_id: str
        Campaign ID

    Returns:
    --------
    List: Three LangChain tools [narratives_tool, details_tool, transcript_tool]
    """
    try:
        # Get CrewAI RagTools
        narratives_rag, details_rag, transcript_rag = create_campaign_rag_tools(
            guild_id,
            campaign_id
        )

        logger.info(f"Created CrewAI RAG tools for guild {guild_id}, campaign {campaign_id}")

        # Wrapping pattern: each @tool function captures the CrewAI RagTool instance
        # via closure and delegates to its _run() method with LangChain-compatible signatures
        @tool
        def campaign_narrative_search(query: str, similarity_threshold: float = 0.5, limit: int = 5) -> str:
            """
            Search narrative session notes that follow a structured analysis format with sections on
            story progression, character developments, NPC interactions, setting, dialogue, plot threads,
            and table talk. Content includes time citations and detailed story elements from TTRPG sessions.

            EFFECTIVE QUERY EXAMPLES:
            • Character information: 'What is Theron's backstory?', 'What did Gerritt do with the mayor?'
            • NPC interactions: 'What is Captain Aldric's personality like?', 'How did Lord Vex betray the party?'
            • Location scenes: 'What happened during the fight at the Thornwick tavern?'
            • Plot developments: 'What is the Crown of Stars prophecy?', 'What happened during the ritual?'
            • Relationships: 'What argument did Theron and Elena have?', 'Why did Marcus trust the stranger?'
            • Story moments: 'What was the revelation about the false king?', 'Who sacrificed at the altar?'

            Args:
                query: The search query to find relevant narrative content
                similarity_threshold: Semantic matching strictness (0.0-1.0). Default 0.5.
                    - 0.7-0.9: Very precise matches only (use for highly specific queries)
                    - 0.5-0.6: Balanced relevance (RECOMMENDED - good starting point)
                    - 0.3-0.4: Broad matches (use when initial search returns nothing)
                    Strategy: Start at 0.5, lower to 0.4 if no results, try 0.3 as last resort.
                limit: Maximum number of results to return. Default 5. Increase to 10-15 for comprehensive research.

            Returns:
                Formatted text with relevant narrative excerpts from session notes
            """
            try:
                logger.info(f"Narratives search called: query='{query[:80]}...', threshold={similarity_threshold}, limit={limit}")
                result = narratives_rag._run(
                    query=query,
                    similarity_threshold=similarity_threshold,
                    limit=limit
                )
                logger.info(f"Narratives search returned {len(result)} chars")
                return result
            except Exception as e:
                logger.error(f"Error in campaign_narrative_search: {e}", exc_info=True)
                return f"Error searching narratives: {str(e)}"

        # Wrap details tool
        @tool
        def campaign_details_search(query: str, similarity_threshold: float = 0.5, limit: int = 5) -> str:
            """
            Search detailed factual notes with sections on game mechanics, combat tactics, treasure/loot,
            items/equipment, location descriptions, clues/discoveries, and world-building references.
            Content includes specific numbers, mechanics, and concrete details with time citations.

            EFFECTIVE QUERY EXAMPLES:
            • Specific NPCs: 'What are Captain Aldric's combat statistics?', 'What items does Merchant Gareth have?'
            • Named locations: 'What is the layout of the Thornwick Inn?', 'What traps are in the Crystal Caverns?'
            • Specific items: 'What are the properties of the Moonblade sword?', 'What does the Cloak of Shadows do?'
            • Combat encounters: 'What tactics were used during the goblin ambush?', 'How much damage does dragon breath do?'
            • Quest items: 'Where are the Crown of Stars fragments?', 'What ritual components are needed?'
            • Mechanics used: 'What was Theron's stealth check result?', 'How much damage did Elena's Fireball deal?'

            Args:
                query: The search query to find relevant detailed content
                similarity_threshold: Semantic matching strictness (0.0-1.0). Default 0.5.
                    - 0.7-0.9: Very precise matches only (use for highly specific queries)
                    - 0.5-0.6: Balanced relevance (RECOMMENDED - good starting point)
                    - 0.3-0.4: Broad matches (use when initial search returns nothing)
                    Strategy: Start at 0.5, lower to 0.4 if no results, try 0.3 as last resort.
                limit: Maximum number of results to return. Default 5. Increase to 10-15 for comprehensive research.

            Returns:
                Formatted text with specific factual details from session notes
            """
            try:
                logger.info(f"Details search called: query='{query[:80]}...', threshold={similarity_threshold}, limit={limit}")
                result = details_rag._run(
                    query=query,
                    similarity_threshold=similarity_threshold,
                    limit=limit
                )
                logger.info(f"Details search returned {len(result)} chars")
                return result
            except Exception as e:
                logger.error(f"Error in campaign_details_search: {e}", exc_info=True)
                return f"Error searching details: {str(e)}"

        # Wrap transcript tool
        @tool
        def campaign_transcript_search(query: str, similarity_threshold: float = 0.4, limit: int = 5) -> str:
            """
            Search raw session transcripts containing verbatim dialogue and conversations from actual play.
            Content includes exact player and DM speech with timestamps, providing the most direct record
            of what was said during sessions.

            EFFECTIVE QUERY EXAMPLES:
            • Exact quotes: 'What did the mayor say about the missing children?', 'What were Elena's exact words to the king?'
            • Dialogue exchanges: 'What conversation did Theron have with Captain Aldric?'
            • Player reactions: 'How did the party react when they found the secret room?'
            • DM narration: 'How did the DM describe the dragon's appearance?'
            • Discussions: 'What did the players debate about the quest choice?'

            Args:
                query: The search query to find relevant transcript content
                similarity_threshold: Semantic matching strictness (0.0-1.0). Default 0.4 (lower for raw text).
                    - 0.6-0.8: Very precise matches only (use for exact quote searches)
                    - 0.4-0.5: Balanced relevance (RECOMMENDED - accounts for transcription variations)
                    - 0.3: Broad matches (use when initial search returns nothing)
                    Strategy: Start at 0.4, lower to 0.3 if no results. Transcripts benefit from lower thresholds due to speech variations.
                limit: Maximum number of results to return. Default 5. Increase to 10-15 for conversation context.

            Returns:
                Formatted text with relevant transcript excerpts including timestamps
            """
            try:
                logger.info(f"Transcript search called: query='{query[:80]}...', threshold={similarity_threshold}, limit={limit}")
                result = transcript_rag._run(
                    query=query,
                    similarity_threshold=similarity_threshold,
                    limit=limit
                )
                logger.info(f"Transcript search returned {len(result)} chars")
                return result
            except Exception as e:
                logger.error(f"Error in campaign_transcript_search: {e}", exc_info=True)
                return f"Error searching transcripts: {str(e)}"

        logger.info("Successfully wrapped all RAG tools as LangChain tools")

        return [campaign_narrative_search, campaign_details_search, campaign_transcript_search]

    except Exception as e:
        logger.error(f"Error creating LangChain RAG tool wrappers: {e}")
        # Return empty list if tools can't be created
        return []
