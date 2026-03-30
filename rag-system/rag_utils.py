# Campaign-specific RAG system using ChromaDB with three content-aware collections
# (narratives, details, transcripts). Each collection uses different chunk sizes
# optimized for its content type — larger chunks for narrative flow, smaller for
# precise dialogue matching.
"""
Enhanced RAG utilities using CrewAI's RagTool with campaign-specific configurations
Provides content-aware chunking and vector database management
"""

import logging
import os
import re
import unicodedata
from pathlib import Path
from typing import Tuple
from crewai_tools import RagTool
from chromadb.config import Settings

logger = logging.getLogger("RAGUtils")


# ChromaDB naming constraints require sanitization — campaign IDs may contain
# unicode characters (e.g., accented names from non-English campaigns)
def _sanitize_collection_name(campaign_id: str) -> str:
    """
    Sanitize campaign ID to create valid Chroma collection names.

    Chroma collection names must:
    1. Contain 3-63 characters
    2. Start and end with alphanumeric character
    3. Contain only alphanumeric characters, underscores or hyphens
    4. Contain no two consecutive periods
    5. Not be a valid IPv4 address
    
    Parameters:
    -----------
    campaign_id: str
        Original campaign ID that may contain foreign characters
        
    Returns:
    --------
    str: Sanitized campaign ID safe for Chroma collection names
    """
    # Normalize unicode characters to ASCII equivalents
    # This converts accented characters like 'á' to 'a', 'ñ' to 'n', etc.
    normalized = unicodedata.normalize('NFD', campaign_id)
    ascii_text = ''.join(char for char in normalized if unicodedata.category(char) != 'Mn')
    
    # Replace any remaining non-alphanumeric characters (except hyphens) with underscores
    sanitized = re.sub(r'[^a-zA-Z0-9\-]', '_', ascii_text)
    
    # Remove consecutive underscores/hyphens and replace with single underscore
    sanitized = re.sub(r'[_\-]+', '_', sanitized)
    
    # Ensure it starts and ends with alphanumeric character
    sanitized = re.sub(r'^[^a-zA-Z0-9]+', '', sanitized)
    sanitized = re.sub(r'[^a-zA-Z0-9]+$', '', sanitized)
    
    # Ensure minimum length of 3 characters
    if len(sanitized) < 3:
        sanitized = sanitized + "_id"
    
    # Ensure maximum length of 63 characters
    if len(sanitized) > 63:
        sanitized = sanitized[:60] + "_id"
    
    # Final validation - if somehow we still have an invalid name, use a fallback
    if not sanitized or not re.match(r'^[a-zA-Z0-9][a-zA-Z0-9_\-]*[a-zA-Z0-9]$', sanitized):
        # Use a hash of the original as fallback
        import hashlib
        hash_suffix = hashlib.md5(campaign_id.encode('utf-8')).hexdigest()[:8]
        sanitized = f"campaign_{hash_suffix}"
    
    return sanitized


def create_campaign_rag_tools(guild_id: int, campaign_id: str) -> Tuple[RagTool, RagTool, RagTool]:
    """
    Create three RagTools for a campaign: narratives, details, and transcripts.
    Each tool has campaign-specific vector database and content-aware chunking.

    Returns:
    --------
    Tuple[RagTool, RagTool, RagTool]: (narratives_tool, details_tool, transcript_tool)
    """
    # Sanitize campaign ID for collection names to handle foreign characters
    sanitized_campaign_id = _sanitize_collection_name(campaign_id)

    # Campaign-specific vector database directory (updated to match new embedding system)
    vector_db_dir = f"chronicler_data/guilds/{guild_id}/campaigns/{campaign_id}/chroma"

    # Ensure vector DB directory exists
    os.makedirs(vector_db_dir, exist_ok=True)

    # Narratives: 600-token chunks with 100-token overlap — larger chunks preserve story flow
    # and character arc continuity across paragraph boundaries
    narratives_config = {
        "vectordb": {
            "provider": "chromadb",
            "config": {}
        },
        "embedding_model": {
            "provider": "openai",
            "config": {
                "model": "text-embedding-3-large"
            }
        },
        "chunker": {
            "chunk_size": 600,
            "chunk_overlap": 100,
            "length_function": "len",
            "min_chunk_size": 50
        }
    }

    # Create tool with minimal config (avoids Pydantic v1/v2 conflict)
    narratives_tool = RagTool(config=narratives_config)

    # Inject custom ChromaDB client with campaign-specific path
    from chromadb import PersistentClient
    from chromadb.utils.embedding_functions import OpenAIEmbeddingFunction
    from crewai.rag.chromadb.client import ChromaDBClient

    # Create ChromaDB Settings (match NEW embedding system to avoid client conflicts)
    chroma_settings = Settings(
        persist_directory=vector_db_dir,
        allow_reset=False,
        anonymized_telemetry=False,
        is_persistent=True
    )

    # Create ChromaDB PersistentClient with custom path
    chroma_client = PersistentClient(path=vector_db_dir, settings=chroma_settings)

    # Get embedding function (use ChromaDB's OpenAIEmbeddingFunction directly to match ingest pipeline)
    api_key = os.getenv("OPENAI_API_KEY")
    if not api_key:
        raise ValueError("OPENAI_API_KEY environment variable required for RAG tools")

    embedding_func = OpenAIEmbeddingFunction(
        api_key=api_key,
        model_name="text-embedding-3-large"
    )

    # Wrap in CrewAI's ChromaDBClient
    wrapped_client = ChromaDBClient(
        client=chroma_client,
        embedding_function=embedding_func,
        default_limit=5,
        default_score_threshold=0.6,
        default_batch_size=100
    )

    # Inject client into adapter (bypasses Pydantic validation)
    narratives_tool.adapter._client = wrapped_client
    narratives_collection_name = f"campaign_{sanitized_campaign_id}_narratives"
    narratives_tool.adapter.collection_name = narratives_collection_name

    # Ensure collection exists and log count
    narratives_collection = wrapped_client.get_or_create_collection(collection_name=narratives_collection_name)
    narratives_count = narratives_collection.count()
    logger.info(f"Narratives collection '{narratives_collection_name}' has {narratives_count} documents")
    narratives_tool.name = "campaign_narrative_search"
    narratives_tool.description = (
        "Search narrative session notes that follow a structured analysis format with sections on "
        "story progression, character developments, NPC interactions, setting, dialogue, plot threads, and table talk. "
        "Content includes time citations and detailed story elements from TTRPG sessions."
        
        "\n\nEFFECTIVE QUERY EXAMPLES:\n"
        "• Character information: 'What is Theron's backstory?', 'What did Gerritt do with the mayor?', 'When did Elena cast Fireball and why?'\n"
        "• NPC interactions: 'What is Captain Aldric's personality like?', 'What was discussed with Innkeeper Brenna?', 'How did Lord Vex betray the party?'\n"
        "• Location scenes: 'What happened during the fight at the Thornwick tavern?', 'What occurred during the Crystal Caverns exploration?', 'What happened at the Shadowmere docks encounter?'\n"
        "• Plot developments: 'What is the Crown of Stars prophecy?', 'What happened during the Cult of the Void ritual?', 'What is the ancient pact with the dragons?'\n"
        "• Relationships: 'What argument did Theron and Elena have?', 'Why did Marcus trust the stranger?', 'When and why did the party decide to split?'\n"
        "• Story moments: 'What was the revelation about the false king?', 'How was the secret passage discovered?', 'Who sacrificed at the altar and why?'\n"
        
        "\n\nEXPECTED OUTPUT FORMAT:\n"
        "Returns structured excerpts from session notes like:\n"
        "\n"
        "Relevant Content:\n"
        "# Quests\n"
        "- **Find the Lost Crown**: The party continues their dangerous journey through the Feywild, seeking clues about the missing artifact.\n"
        "\n"
        "# Decisions/Next Steps\n"
        "- Investigate the purpose of the crystal shard\n"
        "- Determine the threat level of the shadow sprites\n"
        "- Plan the next route through the enchanted forest\n"
        "\n"
        "Each search returns relevant sections with session context and story elements.\n"

        "\n\nINPUT FORMAT: REQUIRED - You MUST include 'query', 'similarity_threshold', and 'limit' fields:\n"
        "{\"query\": \"Where are the drops of the Moonblade?\", \"similarity_threshold\": 0.5, \"limit\": 10}\n\n"
        "SEARCH PARAMETERS:\n"
        "• similarity_threshold: 0.5 (recommended starting point) - Start here for balanced results. "
        "Lower to 0.4 if too few results, increase to 0.6-0.7 only if results are too broad\n"
        "• limit: 10 (recommended) - Number of results to return. Use 5-10 for focused queries, 15-20 for comprehensive research\n\n"
        "SEARCH STRATEGY: For thorough research, try multiple queries with different phrasings and adjust similarity_threshold "
        "to explore the database comprehensively. Lower thresholds help discover tangentially related content."
    )
    
    # Details: 400-token chunks — medium size balances precision for mechanics/stats
    # while retaining enough context for multi-line item or location descriptions
    details_config = {
        "vectordb": {
            "provider": "chromadb",
            "config": {}
        },
        "embedding_model": {
            "provider": "openai",
            "config": {
                "model": "text-embedding-3-large"
            }
        },
        "chunker": {
            "chunk_size": 400,
            "chunk_overlap": 80,
            "length_function": "len",
            "min_chunk_size": 30
        }
    }

    # Create tool with minimal config (avoids Pydantic v1/v2 conflict)
    details_tool = RagTool(config=details_config)

    # Inject custom ChromaDB client with campaign-specific path (reuse same client)
    details_tool.adapter._client = wrapped_client
    details_collection_name = f"campaign_{sanitized_campaign_id}_details"
    details_tool.adapter.collection_name = details_collection_name

    # Ensure collection exists and log count
    details_collection = wrapped_client.get_or_create_collection(collection_name=details_collection_name)
    details_count = details_collection.count()
    logger.info(f"Details collection '{details_collection_name}' has {details_count} documents")
    details_tool.name = "campaign_details_search"
    details_tool.description = (
        "Search detailed factual notes with sections on game mechanics, combat tactics, treasure/loot, "
        "items/equipment, location descriptions, clues/discoveries, and world-building references. "
        "Content includes specific numbers, mechanics, and concrete details with time citations."
        
        "\n\nEFFECTIVE QUERY EXAMPLES:\n"
        "• Specific NPCs: 'What are Captain Aldric's combat statistics?', 'What items are in Merchant Gareth's inventory?', 'Which spells does Wizard Kael know or cast?'\n"
        "• Named locations: 'What is the layout of the Thornwick Inn?', 'What traps are in the Crystal Caverns?', 'What defenses protect Shadowmere Harbor?'\n"
        "• Specific items: 'What are the properties of the Moonblade sword?', 'What abilities does the Cloak of Shadows grant?', 'What are the effects of the healing potion?'\n"
        "• Combat encounters: 'What tactics were used during the goblin ambush?', 'How much damage does the dragon's breath do?', 'How many undead were in the horde?'\n"
        "• Quest items: 'Where are the Crown of Stars fragments located?', 'What ritual components are needed?', 'Where can I find the map to the vault?'\n"
        "• Mechanics used: 'What was Theron's stealth check result?', 'How much damage did Elena's Fireball deal?', 'What was Marcus's saving throw result?'\n"
        
        "\n\nEXPECTED OUTPUT FORMAT:\n"
        "Returns specific factual details like:\n"
        "\n"
        "Relevant Content:\n"
        "## Location Details & Descriptions\n"
        "- Ancient wizard's tower in Thornwick Forest [45-67s]\n"
        "- Room contains:\n"
        "  - Dusty spellbooks [102-125s]\n"
        "  - Alchemical equipment\n"
        "  - Stone pedestal with glowing orb\n"
        "\n"
        "## NPC Details\n"
        "**Mordai the Spellwright**\n"
        "- Race: Half-elf wizard\n"
        "- Occupation: Court mage\n"
        "- Status: Active\n"
        "- Notes: Suspicious of party's motives, guards ancient knowledge\n"
        "\n"
        "Returns precise details with time citations and factual information.\n"

        "\n\nINPUT FORMAT: REQUIRED - You MUST include 'query', 'similarity_threshold', and 'limit' fields:\n"
        "{\"query\": \"Where are the drops of the Moonblade?\", \"similarity_threshold\": 0.5, \"limit\": 10}\n\n"
        "SEARCH PARAMETERS:\n"
        "• similarity_threshold: 0.5 (recommended starting point) - Start here for balanced results. "
        "Lower to 0.4 if too few results, increase to 0.6-0.7 only if results are too broad\n"
        "• limit: 10 (recommended) - Number of results to return. Use 5-10 for specific details, 15-20 for exhaustive fact-finding\n\n"
        "SEARCH STRATEGY: For thorough fact-checking and research, use multiple targeted queries with varied keywords. "
        "Adjust similarity_threshold lower (0.4-0.5) when searching for items, NPCs, or locations that might be mentioned "
        "with different terminology across sessions."
    )
    
    # Transcripts: 300-token chunks — smaller chunks enable precise dialogue matching
    # since individual speaker turns are typically short
    transcript_config = {
        "vectordb": {
            "provider": "chromadb",
            "config": {}
        },
        "embedding_model": {
            "provider": "openai",
            "config": {
                "model": "text-embedding-3-large"
            }
        },
        "chunker": {
            "chunk_size": 300,
            "chunk_overlap": 60,
            "length_function": "len",
            "min_chunk_size": 20
        }
    }

    # Create tool with minimal config (avoids Pydantic v1/v2 conflict)
    transcript_tool = RagTool(config=transcript_config)

    # Inject custom ChromaDB client with campaign-specific path (reuse same client)
    transcript_tool.adapter._client = wrapped_client
    transcript_collection_name = f"campaign_{sanitized_campaign_id}_transcripts"
    transcript_tool.adapter.collection_name = transcript_collection_name

    # Ensure collection exists and log count
    transcript_collection = wrapped_client.get_or_create_collection(collection_name=transcript_collection_name)
    transcript_count = transcript_collection.count()
    logger.info(f"Transcript collection '{transcript_collection_name}' has {transcript_count} documents")
    transcript_tool.name = "campaign_transcript_search"
    transcript_tool.description = (
        "Search raw session transcripts containing exact dialogue, audio artifacts, time markers [XXXs], "
        "and unprocessed conversation. Use when you need verbatim quotes, precise wording, "
        "or to verify specific details not captured in the structured analyses."
        
        "\n\nEFFECTIVE QUERY EXAMPLES:\n"
        "• Exact character quotes: 'What were Theron's exact words to Elena?', 'What did Marcus say about leaving?'\n"
        "• NPC dialogue: 'What warning did Captain Aldric give?', 'What information did Innkeeper Brenna provide?'\n"
        "• Specific conversations: 'What was said during the argument between Theron and Marcus?', 'What was discussed during the negotiation with Lord Vex?'\n"
        "• Precise details: 'What exact damage number was called?', 'How was the spell name pronounced?', 'What was the exact wording of the riddle?'\n"
        "• Player reactions: 'How did the players react to Elena's death?', 'What was the reaction to the plot twist?', 'When did the table laugh at a joke and what was said?'\n"
        "• GM explanations: 'What rule clarification did the GM give about stealth?', 'How did the GM describe the ancient runes?'\n"
        
        "\n\nEXPECTED OUTPUT FORMAT:\n"
        "Returns raw transcript snippets like:\n"
        "\n"
        "Relevant Content:\n"
        "**[245s] player_mage**: That's a Shadowhound, a creature from the Underdark that is dangerous in many ways\n"
        "\n"
        "**[247s] dungeon_master**: The creature from the shadow realm that is\n"
        "\n"
        "**[250s] player_rogue**: creature from the shadow fell that is unusual in many respects, uh\n"
        "\n"
        "WARNING: Contains transcription errors, audio artifacts, speaker misattribution, and unclear audio.\n"
        "Use for exact quotes and verification when structured notes lack precision.\n"

        "\n\nINPUT FORMAT: REQUIRED - You MUST include 'query', 'similarity_threshold', and 'limit' fields:\n"
        "{\"query\": \"What did Marcus say about the ritual?\", \"similarity_threshold\": 0.4, \"limit\": 15}\n\n"
        "SEARCH PARAMETERS:\n"
        "• similarity_threshold: 0.4 (recommended starting point for transcripts) - Transcripts need lower thresholds due to "
        "transcription variations. Increase to 0.5-0.6 only if results are too broad\n"
        "• limit: 15 (recommended) - Number of results to return. Use 10-15 for quote searches, 20-30 for conversation context\n\n"
        "SEARCH STRATEGY: For verbatim quotes and exact dialogue, use speaker names and key phrases in queries. "
        "Lower similarity_threshold (0.4-0.5) helps find conversations where the same topic was discussed with different wording. "
        "For thorough dialogue research, try queries with character names, specific topics, and key vocabulary from the discussion."
    )
    
    logger.info(
        f"Created 3 campaign-specific RAG tools for guild {guild_id}, campaign {campaign_id} "
        f"(sanitized: {sanitized_campaign_id}) - Documents: {narratives_count} narratives, "
        f"{details_count} details, {transcript_count} transcripts"
    )
    return narratives_tool, details_tool, transcript_tool


# Production: tracks indexed content via a manifest file to avoid re-processing
# unchanged session files on subsequent runs. Manifest logic removed for showcase.
def index_session_content(
    session_dir: Path,
    session_id: str,
    narratives_tool: RagTool | None = None,
    details_tool: RagTool | None = None,
    transcript_tool: RagTool | None = None,
) -> dict:
    """
    Index session content into the vector database.
    Call this AFTER session notes are generated.

    Parameters:
    -----------
    session_dir: Path
        Path to session directory containing files
    session_id: str
        Session identifier for metadata
    narratives_tool: RagTool | None
        Tool for narrative story content
    details_tool: RagTool | None
        Tool for detailed factual content
    transcript_tool: RagTool | None
        Tool for raw transcript content

    Returns:
    --------
    dict: Results of indexing operation with keys: indexed_files, skipped_files, errors
    """
    if narratives_tool is None or details_tool is None or transcript_tool is None:
        error_msg = "Requires narratives_tool, details_tool, and transcript_tool parameters"
        logger.error(error_msg)
        return {"indexed_files": [], "skipped_files": [], "errors": [error_msg]}

    results = {"indexed_files": [], "skipped_files": [], "errors": []}

    campaign_id = session_dir.parent.parent.name

    def _index_file(source_path: Path, tool: RagTool, target: str, temp_name: str) -> None:
        if not source_path.exists():
            return

        try:
            enhanced_content = _add_metadata_header(source_path, session_id, source_path.name)
            temp_path = session_dir / temp_name
            metadata = {
                "session_id": session_id,
                "campaign_id": campaign_id,
                "target": target,
                "source_path": str(source_path.relative_to(session_dir)),
            }

            try:
                with open(temp_path, "w", encoding="utf-8") as handle:
                    handle.write(enhanced_content)
                tool.add(str(temp_path), metadata=metadata)
                results["indexed_files"].append(f"{target}: {source_path.name}")
                logger.info("Indexed %s to %s tool", source_path.name, target)
            finally:
                if temp_path.exists():
                    temp_path.unlink()
        except Exception as exc:
            error_msg = f"Failed to index {source_path.name}: {exc}"
            results["errors"].append(error_msg)
            logger.error(error_msg)

    try:
        _index_file(session_dir / "narrative_notes.txt", narratives_tool, "narratives", "temp_narrative_notes.txt")
        _index_file(session_dir / "detail_notes.txt", details_tool, "details", "temp_detail_notes.txt")

        session_notes_path = session_dir / "session_notes.md"
        if session_notes_path.exists():
            _index_file(session_notes_path, narratives_tool, "narratives", "temp_session_notes_narratives.md")
            _index_file(session_notes_path, details_tool, "details", "temp_session_notes_details.md")

        _index_file(session_dir / "master_transcript.md", transcript_tool, "transcript", "temp_master_transcript.md")
    except Exception as exc:
        error_msg = f"Failed to access session directory: {exc}"
        results["errors"].append(error_msg)
        logger.error(error_msg)

    return results



def _add_metadata_header(file_path: Path, session_id: str, filename: str) -> str:
    """Add metadata header to file content for better search context."""
    
    # Extract session date from session_id (format: session-YYYYMMDD-XXXXXX)
    session_date = "Unknown"
    if session_id.startswith("session-") and len(session_id) > 16:
        try:
            date_part = session_id[8:16]  # Extract YYYYMMDD
            session_date = f"{date_part[:4]}-{date_part[4:6]}-{date_part[6:8]}"
        except:
            pass
    
    # Read original content
    with open(file_path, 'r', encoding='utf-8') as f:
        original_content = f.read()
    
    # Create enhanced content with metadata header
    doc_type = filename.replace('.md', '').replace('.txt', '').replace('_', ' ').title()
    
    header = f"""# {doc_type} - {session_id}

**Session ID**: {session_id}  
**Date**: {session_date}  
**Document Type**: {doc_type}  
**Source File**: {filename}

---

"""
    
    return header + original_content



