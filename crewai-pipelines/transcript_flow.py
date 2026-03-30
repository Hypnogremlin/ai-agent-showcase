# Multi-phase transcript processing pipeline using CrewAI Flow.
# Handles longer sessions via segment analysis then synthesis.
# Shorter sessions use session_crew.py instead (single-crew path).
"""
DnD Transcript Flow

This module provides a CrewAI Flow for processing D&D session transcripts,
orchestrating the analysis and synthesis of session notes.

Note: AgentOps is initialized in start_bot.py at bot startup.
This automatically tracks all CrewAI Flow operations.
"""

import logging
import asyncio
import time
from pathlib import Path
from typing import Dict, Any
from uuid import uuid4

from pydantic import BaseModel, Field
from crewai.flow.flow import Flow, listen, start

from .segment_analysis_crew import SegmentAnalysisCrew
from .synthesis_crew import SynthesisCrew
from .llm_config import reset_to_primary_mode, with_llm_fallback, get_current_provider_info

# Configure logging
logger = logging.getLogger("TranscriptFlow")


class TranscriptFlowState(BaseModel):
    """State for transcript processing flow"""
    id: str = Field(default_factory=lambda: str(uuid4()))
    analysis_complete: bool = False
    synthesis_complete: bool = False
    narrative_path: str = ""
    detail_path: str = ""
    session_notes: str = ""
    database_updates: str = ""
    segment_input_tokens: int = 0
    segment_output_tokens: int = 0
    synthesis_input_tokens: int = 0
    synthesis_output_tokens: int = 0


class DnDTranscriptFlow(Flow[TranscriptFlowState]):
    """
    Flow for processing D&D transcripts.
    
    This flow orchestrates:
    1. Segment analysis to create narrative and detail notes files
    2. Synthesis to create final session notes and database updates
    """
    
    initial_state = TranscriptFlowState
    
    def __init__(self, session_data: Dict[str, Any]):
        """
        Initialize flow with session data.
        
        Parameters:
        -----------
        session_data: Dict[str, Any]
            Complete session data including transcript, campaign info, etc.
        """
        super().__init__()
        self._session_data = session_data
        
        # Extract key identifiers
        self._guild_id = session_data.get('guild_id')
        self._session_id = session_data.get('session_id')
        self._campaign_id = session_data.get('campaign_id')
        
        # Reset to primary LLM mode for each new session (TEMPORARILY DISABLED FOR TESTING)
        # TODO: Re-enable this after testing manual mode switching
        try:
            reset_to_primary_mode()  
            provider_info = get_current_provider_info()
            logger.info(f"DnDTranscriptFlow initialized for session {self._session_id} using {provider_info['provider']} - {provider_info['model']}")
        except Exception as e:
            logger.warning(f"Error getting LLM info: {e}, continuing with current settings")
            logger.info(f"DnDTranscriptFlow initialized for session {self._session_id}")
    
    # @start kicks off the flow — triggers parallel segment analysis across all transcript chunks
    @start()
    def analyze_transcript_segments(self) -> Dict[str, Any]:
        """
        Step 1: Analyze all transcript segments and create analysis files.
        """
        logger.info("Starting transcript segment analysis")
        
        # Define crew factory and operation for LLM fallback wrapper
        def segment_crew_factory():
            return SegmentAnalysisCrew()
        
        def execute_segment_analysis(crew):
            result = crew.process_all_segments(self._session_data)
            if result.get('status') != 'success':
                raise Exception(f"Segment analysis failed: {result.get('error', 'Unknown error')}")
            return result
        
        # Execute with LLM fallback logic
        result = with_llm_fallback(
            segment_crew_factory,
            execute_segment_analysis,
            "segment analysis"
        )
        
        # Store file paths in state
        self.state.narrative_path = result.get('narrative_path', '')
        self.state.detail_path = result.get('detail_path', '')
        self.state.analysis_complete = True

        # Capture segment analysis token usage
        try:
            self.state.segment_input_tokens = result.get('input_tokens', 0)
            self.state.segment_output_tokens = result.get('output_tokens', 0)
        except Exception:
            pass  # Never let metrics break processing
        
        logger.info("Segment analysis completed successfully")
        
        # Rate-limit pause: prevents API quota exhaustion between analysis and synthesis phases.
        # Segment analysis can consume significant token budgets; this cooldown lets quotas reset.
        logger.info("Pausing for 60 seconds before synthesis to avoid rate limiting...")
        time.sleep(60)
        
        return result
    
    # @listen waits for segment analysis to complete before starting synthesis.
    # CrewAI Flow manages the dependency graph automatically.
    @listen(analyze_transcript_segments)
    def synthesize_final_outputs(self, analysis_result: Dict[str, Any]) -> Dict[str, Any]:
        """
        Step 2: Synthesize the analysis files into final session notes and database updates.
        """
        logger.info("Starting synthesis of final outputs")
        
        narrative_path = Path(self.state.narrative_path)
        detail_path = Path(self.state.detail_path)
        
        if not narrative_path.exists() or not detail_path.exists():
            raise FileNotFoundError("Analysis files not found")
        
        with open(narrative_path, 'r', encoding='utf-8') as f:
            narrative_notes = f.read()
        
        with open(detail_path, 'r', encoding='utf-8') as f:
            detail_notes = f.read()
        
        # Prepare inputs for synthesis crew
        synthesis_inputs = {
            'narrative_notes': narrative_notes,
            'detail_notes': detail_notes,
            'context': {
                'campaign_data': self._session_data.get('campaign_data', {}),
                'characters': self._session_data.get('characters', []),
                'npcs': self._session_data.get('npcs', []),
                'world': self._session_data.get('world', {}),
                'locations': self._session_data.get('locations', []),
                'quests': self._session_data.get('quests', []),
                'factions': self._session_data.get('factions', []),
                'previous_session_notes': self._session_data.get('previous_session_notes', [])
            },
            'guild_id': self._guild_id,
            'session_id': self._session_id,
            'campaign_id': self._campaign_id
        }
        
        # Define crew factory and operation for LLM fallback wrapper
        def synthesis_crew_factory():
            return SynthesisCrew(guild_id=self._guild_id, campaign_id=self._campaign_id)
        
        def execute_synthesis(crew):
            results = crew.crew().kickoff(inputs=synthesis_inputs)
            # Extract outputs (TEMPORARY: only 2 tasks while database curator disabled)
            if not (results and results.tasks_output and len(results.tasks_output) >= 2):
                raise Exception("Unexpected synthesis result format")
            return results
        
        # Execute with LLM fallback logic
        results = with_llm_fallback(
            synthesis_crew_factory,
            execute_synthesis,
            "synthesis"
        )
        
        # Extract outputs
        refined_context = results.tasks_output[0].raw
        session_notes = results.tasks_output[1].raw
        # Production: indexes new session content to ChromaDB for future RAG retrieval
        database_updates = "Database curator temporarily disabled for vector DB testing"
        
        logger.info("Extracted refined context for session notes")
        
        logger.info("Session notes generated")

        # Extract CrewAI token usage from synthesis crew result and store in state
        try:
            tu = getattr(results, "token_usage", None)
            if tu:
                self.state.synthesis_input_tokens = getattr(tu, "prompt_tokens", 0)
                self.state.synthesis_output_tokens = getattr(tu, "completion_tokens", 0)
        except Exception:
            pass  # Never let metrics break processing

        # Update state
        self.state.session_notes = session_notes
        self.state.database_updates = database_updates
        self.state.synthesis_complete = True

        return {
            "status": "success",
            "session_notes": session_notes,
            "database_updates": database_updates,
        }
    
    async def process_session(self, session_data: Dict[str, Any]) -> Dict[str, Any]:
        """
        Process a D&D session - async wrapper for Discord bot compatibility.
        
        Note: The session_data parameter is not used (data comes from constructor),
        but is kept for API compatibility with existing code.
        
        Parameters:
        -----------
        session_data: Dict[str, Any]
            Not used - kept for compatibility
            
        Returns:
        --------
        Dict[str, Any]: Results formatted for the bot's expectations
        """
        logger.info(f"Starting transcript flow processing for session {self._session_id}")
        
        try:
            # Run the flow in a thread pool to avoid blocking
            loop = asyncio.get_event_loop()
            await loop.run_in_executor(None, self.kickoff)

            # Read token usage from flow state — sum segment analysis + synthesis phases
            token_usage: Dict[str, Any] = {}
            try:
                total_input = self.state.segment_input_tokens + self.state.synthesis_input_tokens
                total_output = self.state.segment_output_tokens + self.state.synthesis_output_tokens
                if total_input or total_output:
                    token_usage = {
                        "input_tokens": total_input,
                        "output_tokens": total_output,
                    }
            except Exception:
                pass  # Never let metrics break processing

            # Format results for bot compatibility
            return {
                "notes_result": {
                    "tasks_output": [{"raw": self.state.session_notes}]
                },
                "campaign_result": {
                    "tasks_output": [{"raw": self.state.database_updates}]
                },
                "session_notes": self.state.session_notes,
                "token_usage": token_usage,
            }

        except Exception as e:
            logger.error(f"Error in process_session: {str(e)}")
            raise
