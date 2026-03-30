# Synthesis crew that combines segment analyses into final session notes.
# Supports RAG-enhanced context retrieval and campaign database updates.
# This is the second phase of the multi-phase transcript processing pipeline.
"""
Synthesis Crew

This crew handles the final synthesis of all segment analyses into
session notes and database updates.

Note: AgentOps is initialized in start_bot.py at bot startup.
This automatically tracks all CrewAI operations.
"""

import logging
from crewai import Agent, Crew, Process, Task
from crewai.project import CrewBase, agent, crew, task, before_kickoff

from .llm_config import get_llm_for_role
from .tools.pc_update_tool import PCUpdateTool
from .tools.npc_update_tool import NPCUpdateTool
from .tools.campaign_update_tool import CampaignUpdateTool

# Configure logging
logger = logging.getLogger("SynthesisCrew")


@CrewBase
class SynthesisCrew:
    """Crew for synthesizing segment analyses into final outputs"""
    
    agents_config = 'config/synthesis_agents.yaml'
    tasks_config = 'config/synthesis_tasks.yaml'
    
    def __init__(self, guild_id: int, campaign_id: str):
        """Initialize with guild and campaign IDs for tool creation."""
        super().__init__()
        self.guild_id = guild_id
        self.campaign_id = campaign_id
    
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
            logger.error(f"Error loading LLM settings for SynthesisCrew: {e}")

        return inputs
    
    @agent
    def context_specialist(self) -> Agent:
        """Agent responsible for retrieving deeper campaign context using enhanced RAG"""
        # Production: gets three RAG tools (narrative_search, details_search, transcript_search)
        # for querying campaign-specific ChromaDB vector collections
        tools = []
        return Agent(
            config=self.agents_config['context_specialist'],
            tools=tools,
            llm=get_llm_for_role("context_specialist"),
            verbose=True
        )
    
    @agent
    def chronicler(self) -> Agent:
        """Agent responsible for compiling final session notes"""
        return Agent(
            config=self.agents_config['chronicler'],
            llm=get_llm_for_role("chronicler"),
            verbose=True
        )

    @agent
    def database_curator(self) -> Agent:
        """Agent responsible for updating campaign database with new information.
        This agent uses CrewAI tools to write new NPCs, locations, quests, and
        other entities discovered during the session into campaign XML databases."""
        return Agent(
            config=self.agents_config['database_curator'],
            llm=get_llm_for_role("database_curator"),
            verbose=True,
            tools=[PCUpdateTool(), NPCUpdateTool(), CampaignUpdateTool()]
        )
    
    @task
    def extract_refined_context(self) -> Task:
        """Extract refined context from campaign history"""
        return Task(
            config=self.tasks_config['extract_refined_context'],
            agent=self.context_specialist()
        )
    
    @task
    def compile_session_notes(self) -> Task:
        """Compile all segment analyses into final session notes"""
        return Task(
            config=self.tasks_config['compile_session_notes'],
            agent=self.chronicler()
        )
    
    @task
    def update_campaign_database(self) -> Task:
        """Update campaign database with information from segment analyses"""
        return Task(
            config=self.tasks_config['update_campaign_database'],
            agent=self.database_curator()
        )
    
    @crew
    def crew(self) -> Crew:
        """Creates the Synthesis Crew"""
        # TEMPORARY: Exclude database curator task while testing vector DB
        active_tasks = [self.extract_refined_context(), self.compile_session_notes()]
        # Uncomment to re-enable database updates:
        # active_tasks.append(self.update_campaign_database())
        
        return Crew(
            agents=self.agents,  # Automatically created by the @agent decorator
            tasks=active_tasks,  # Only use context and notes tasks for now
            process=Process.sequential,
            verbose=True
        )
