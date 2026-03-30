# Parallel segment analysis crew. Processes transcript segments independently
# using kickoff_for_each for concurrent execution. Each segment gets both a
# narrative analyst (story/character focus) and a detail analyst (mechanics/facts).
"""
Segment Analysis Crew

This crew handles the analysis of transcript segments,
providing both narrative and detailed analysis of the content.

Note: AgentOps is initialized in start_bot.py at bot startup.
This automatically tracks all CrewAI operations.
"""

import logging
from pathlib import Path
from typing import Dict, Any

from crewai import Agent, Crew, Process, Task
from crewai.project import CrewBase, agent, crew, task, before_kickoff

from .llm_config import get_llm_for_role

# Configure logging
logger = logging.getLogger("SegmentAnalysisCrew")


@CrewBase
class SegmentAnalysisCrew:
    """Crew for analyzing transcript segments"""
    
    agents_config = 'config/segment_agents.yaml'
    tasks_config = 'config/segment_tasks.yaml'
    
    @before_kickoff
    def load_llm_settings(self, inputs):
        """Assign per-role LLMs to all agents before crew execution.

        Builds a reverse map from YAML role text -> config key name so
        get_llm_for_role() receives the correct key for override lookups.
        """
        try:
            # Build reverse map: YAML role text -> config key name
            role_text_to_key = {}
            for key, cfg in self.agents_config.items():
                role_text = cfg.get("role", "").strip()
                if role_text:
                    role_text_to_key[role_text] = key

            for agent in self.agents:
                config_key = role_text_to_key.get(agent.role.strip(), agent.role)
                agent.llm = get_llm_for_role(config_key)
                logger.debug(f"Agent '{config_key}' assigned LLM: {agent.llm.model}")
        except Exception as e:
            logger.error(f"Error loading LLM settings for SegmentAnalysisCrew: {e}")

        return inputs
    
    @agent
    def narrative_analyst(self) -> Agent:
        """Agent focused on story progression, character development, and social interactions"""
        return Agent(
            config=self.agents_config['narrative_analyst'],
            llm=get_llm_for_role("narrative_analyst"),
            verbose=False
        )

    @agent
    def detail_analyst(self) -> Agent:
        """Agent focused on game mechanics, combat details, and factual elements"""
        return Agent(
            config=self.agents_config['detail_analyst'],
            llm=get_llm_for_role("detail_analyst"),
            verbose=False
        )
    
    @task
    def narrative_analysis(self) -> Task:
        """Analyze narrative elements of the transcript segment"""
        return Task(
            config=self.tasks_config['narrative_analysis'],
            agent=self.narrative_analyst()
        )
    
    @task
    def detail_analysis(self) -> Task:
        """Analyze detailed mechanics and factual elements of the transcript segment"""
        return Task(
            config=self.tasks_config['detail_analysis'],
            agent=self.detail_analyst()
        )
    
    @crew
    def crew(self) -> Crew:
        """Creates the Segment Analysis Crew"""
        return Crew(
            agents=self.agents,
            tasks=self.tasks,
            process=Process.sequential,
            verbose=True
        )
    
    def process_all_segments(self, session_data: Dict[str, Any]) -> Dict[str, Any]:
        """
        Process all transcript segments using CrewAI's built-in iteration.
        
        Parameters:
        -----------
        session_data: Dict[str, Any]
            Complete session data including transcript and context
            
        Returns:
        --------
        Dict with status and file paths
        """
        logger.info("Starting segment analysis processing")
        
        # Extract essential data
        guild_id = session_data.get('guild_id')
        session_id = session_data.get('session_id')
        campaign_id = session_data.get('campaign_id')
        total_segments = session_data.get('total_segments', 1)
        has_segments = session_data.get('has_segments', False)
        
        # Prepare context that will be shared across all segments
        # Now includes RAG-based relevant context
        shared_context = {
            'campaign_data': session_data.get('campaign_data', {}),
            'characters': session_data.get('characters', []),
            'npcs': session_data.get('npcs', []),
            'world': session_data.get('world', {}),
            'locations': session_data.get('locations', []),
            'quests': session_data.get('quests', []),
            'factions': session_data.get('factions', []),
            'previous_session_notes': session_data.get('previous_session_notes', []),
            'relevant_context': session_data.get('relevant_context', {})  # NEW: RAG-based context
        }
        
        # Build inputs array for all segments
        segments_to_process = total_segments if has_segments else 1
        logger.info(f"Processing {segments_to_process} segment(s)")
        
        inputs_array = []
        for segment_index in range(segments_to_process):
            if has_segments:
                transcript = session_data.get('transcript', '')
            else:
                transcript = session_data.get('transcript', '')
            
            # Prepare crew inputs for this segment
            crew_inputs = {
                'transcript': transcript,
                'context': shared_context,
                'segment_index': segment_index,
                'total_segments': segments_to_process
            }
            inputs_array.append(crew_inputs)
        
        # kickoff_for_each is CrewAI's built-in parallel execution — each segment
        # gets its own crew run, enabling concurrent processing of long sessions
        results = self.crew().kickoff_for_each(inputs=inputs_array)
        
        # Process results and accumulate outputs
        all_narrative_notes = ""
        all_detail_notes = ""
        total_input_tokens = 0
        total_output_tokens = 0

        for i, result in enumerate(results):
            if result and result.tasks_output and len(result.tasks_output) >= 2:
                narrative_output = result.tasks_output[0].raw
                detail_output = result.tasks_output[1].raw

                # Accumulate the outputs
                all_narrative_notes += narrative_output + "\n\n"
                all_detail_notes += detail_output + "\n\n"

                # Aggregate token costs across all parallel segments for billing accuracy
                try:
                    tu = getattr(result, "token_usage", None)
                    if tu:
                        total_input_tokens += getattr(tu, "prompt_tokens", 0)
                        total_output_tokens += getattr(tu, "completion_tokens", 0)
                except Exception:
                    pass  # Never let metrics break processing

                logger.info(f"Successfully processed segment {i + 1}/{segments_to_process}")
            else:
                logger.error(f"Unexpected result format for segment {i + 1}")
                raise Exception(f"Segment {i + 1} returned unexpected result format")
        
        return {
            "status": "success",
            "narrative_notes": all_narrative_notes.strip(),
            "detail_notes": all_detail_notes.strip(),
            "input_tokens": total_input_tokens,
            "output_tokens": total_output_tokens,
        }
