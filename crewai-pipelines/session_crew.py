# Single-crew session processor for shorter TTRPG sessions.
# Uses CrewAI @CrewBase with per-role LLM overrides to route different agents
# to different providers (e.g., expensive model for synthesis, cheaper for context retrieval).
# For longer sessions, the multi-phase transcript_flow.py pipeline is used instead.

import logging
from pathlib import Path
import asyncio

from crewai import Agent, Crew, Process, Task, LLM
from crewai.project import CrewBase, agent, crew, task, before_kickoff
from .tools.pc_update_tool import PCUpdateTool
from .tools.npc_update_tool import NPCUpdateTool
from .tools.campaign_update_tool import CampaignUpdateTool
from .llm_config import get_llm_for_role, with_llm_fallback

# Configure logging
logger = logging.getLogger("DnDSessionCrew")

@CrewBase
class DnDSessionCrew():
	"""D&D Session Processing Crew - Creates session notes with RAG context when available"""

	agents_config = 'config/synthesis_agents.yaml'
	tasks_config = 'config/short_session_tasks.yaml'
	
	def __init__(self, guild_id: int = None, campaign_id: str = None):
		"""Initialize with guild and campaign IDs for RAG tool creation."""
		super().__init__()
		self.guild_id = guild_id
		self.campaign_id = campaign_id
	
	# @before_kickoff runs before crew execution — this is where per-role LLM assignments happen at runtime,
	# mapping each agent to its configured provider (primary, backup, or role-specific override).
	@before_kickoff
	def prepare_inputs(self, inputs):
		"""
		Load LLM settings and restructure inputs to provide simpler access patterns.
		
		Creates:
		- transcript: Just the transcript text
		- context: All campaign-related data grouped together
		
		Original input keys are preserved for backward compatibility.
		"""
		if inputs is None:
			return {}
			
		# Assign per-role LLMs: build reverse map from YAML role text -> config key
		try:
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
			logger.error(f"Error loading LLM settings for DnDSessionCrew: {e}")
			
		# Extract transcript to its own top-level key
		transcript = inputs.get('transcript', '')
		
		# Extract campaign_id if available, use what's provided
		campaign_id = inputs.get('campaign_id')
		guild_id = inputs.get('guild_id')
		
		# Extract previous session notes if available
		previous_notes = inputs.get('previous_session_notes', [])
		
		# Group all campaign/world data under 'context'
		context = {
			'campaign_id': campaign_id,  # Include campaign_id explicitly
			'guild_id': guild_id,  # Make guild_id more prominent
			'campaign_data': inputs.get('campaign_data', {}),
			'characters': inputs.get('characters', []),
			'npcs': inputs.get('npcs', []),
			'world': inputs.get('world', {}),
			'locations': inputs.get('locations', []),
			'quests': inputs.get('quests', []),
			'factions': inputs.get('factions', []),
			'previous_session_notes': previous_notes  # Add previous notes to context
		}
		
		# Add guild_id and campaign_id to inputs for tasks (from instance variables if not in inputs)
		if self.guild_id and not guild_id:
			context['guild_id'] = self.guild_id
		if self.campaign_id and not campaign_id:
			context['campaign_id'] = self.campaign_id
		
		# Create a new inputs dictionary with simplified structure
		new_inputs = {
			'transcript': transcript,
			'context': context,
			# Keep original keys for backward compatibility
			**inputs
		}
		
		return new_inputs

	@agent
	def chronicler(self) -> Agent:
		return Agent(
			config=self.agents_config['chronicler'],
			llm=get_llm_for_role("chronicler"),
			verbose=True
		)

	@agent
	def context_specialist(self) -> Agent:
		"""Agent responsible for retrieving deeper campaign context using RAG when available"""
		# Production: tools list is populated with campaign-specific RAG tools
		# (narrative_search, details_search, transcript_search) for vector DB retrieval
		tools = []
		return Agent(
			config=self.agents_config['context_specialist'],
			tools=tools,
			llm=get_llm_for_role("context_specialist"),
			verbose=True
		)

	@task
	def compile_session_notes(self) -> Task:
		return Task(
			config=self.tasks_config['compile_session_notes'],
		)

	@task
	def extract_refined_context(self) -> Task:
		"""Extract refined context from campaign history using RAG or data_loader"""
		return Task(
			config=self.tasks_config['extract_refined_context'],
			agent=self.context_specialist()
		)
	

	@crew
	def crew(self) -> Crew:
		"""Creates the D&D Session Processing Crew with RAG context retrieval"""
		# Task sequence: extract context -> compile notes (no database updates per PRP requirements)
		return Crew(
			agents=self.agents, # Automatically created by the @agent decorator
			tasks=[self.extract_refined_context(), self.compile_session_notes()], # Context first, then notes
			process=Process.sequential,
			verbose=True,
		)

	# Production: writes session notes to disk, indexes content to ChromaDB vector store,
	# and fires cost tracking events for LLM usage monitoring.
	async def process_session(self, session_data):
		"""
		Process a D&D session using the data loaded from the data_loader.
		This method runs asynchronously to avoid blocking the Discord bot.
		
		Parameters:
		-----------
		session_data: dict
			Dictionary containing all session and campaign data
			
		Returns:
		--------
		Dictionary containing results from both tasks
		"""
		# Extract guild_id, session_id, and campaign_id for later use
		guild_id = session_data.get('guild_id')
		session_id = session_data.get('session_id')
		campaign_id = session_data.get('campaign_id')
		
		logger.info(f"Starting session processing for session {session_id}")
		
		# Define crew factory for LLM fallback wrapper
		def crew_factory():
			return DnDSessionCrew(guild_id=guild_id, campaign_id=campaign_id)
		
		# Define operation function
		def execute_crew(crew):
			return crew.crew().kickoff(session_data)
		
		# Run the crew with LLM fallback logic
		loop = asyncio.get_event_loop()
		result = await loop.run_in_executor(
			None,
			with_llm_fallback,
			crew_factory,
			execute_crew,
			"single segment session processing"
		)
		
		# Extract session notes and save to file
		session_notes = None
		if result and result.tasks_output and len(result.tasks_output) >= 2:
			# First task is extract_refined_context, second is compile_session_notes
			session_notes = result.tasks_output[1].raw  # Second task is compile_session_notes
			logger.info(f"Session notes generated for session {session_id}")
		
		# Extract CrewAI token usage (CrewOutput.token_usage in CrewAI >= 0.100)
		token_usage = {}
		try:
			tu = getattr(result, "token_usage", None)
			if tu:
				token_usage = {
					"input_tokens": getattr(tu, "prompt_tokens", 0),
					"output_tokens": getattr(tu, "completion_tokens", 0),
				}
		except Exception:
			pass  # Never let metrics break processing

		# Format results for compatibility
		combined_results = {
			"context_result": {"tasks_output": [result.tasks_output[0]]} if result.tasks_output else None,
			"notes_result": {"tasks_output": [result.tasks_output[1]]} if result.tasks_output and len(result.tasks_output) > 1 else None,
			"session_notes": session_notes,
			"token_usage": token_usage,
		}

		return combined_results
		
