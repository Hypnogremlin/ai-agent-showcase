# RAG research agent (Quoth the Research Familiar).
# Demonstrates the "Agent as Tool" pattern -- a compiled LangGraph agent
# wrapped as a LangChain StructuredTool for nested invocation by the parent Chronicler agent.
# This pattern solves the async/sync boundary issues when nesting LangGraph agents.
"""
Research Familiar - Specialized RAG research agent for Chronicler

This module implements the Research Familiar, a focused helper agent that handles
all vector database queries for the main Chronicler agent. The Familiar uses RAG tools
to search campaign narratives, details, and transcripts when the Chronicler needs
additional information beyond pre-loaded context blocks.

Architecture Pattern: "Agent as Tool"
- The Familiar is exposed to Chronicler as a LangChain @tool
- When called, Familiar's research is automatically persisted in conversation state by LangGraph
- Lower temperature (0.5) for factual, focused research
- Smaller token budget (300) for concise findings
"""

import asyncio
import logging
from time import perf_counter
from typing import List, TypedDict, Annotated

from langchain_core.tools import StructuredTool
from langchain_core.messages import AIMessage, BaseMessage, HumanMessage
from langgraph.prebuilt import create_react_agent
from langgraph.graph import StateGraph, START, END
from langgraph.graph.message import add_messages

from .chat_llm_config import (
    get_chronicler_llm,
    get_current_llm_info,
)

logger = logging.getLogger("ResearchFamiliar")


def sanitize_messages_for_provider(messages, provider):
    return messages


# Production: wraps CrewAI RagTools as LangChain @tool functions for three
# campaign collections (narratives, details, transcripts) — see rag_tool_wrappers.py
def create_langchain_rag_tools(guild_id: int, campaign_id: str) -> list:
    return []


class FamiliarState(TypedDict):
    """
    State schema for the Research Familiar graph.

    Simple state with just messages - the Familiar is stateless and
    doesn't need to persist conversation history (that's handled by
    the main Chronicler graph).
    """
    messages: Annotated[List[BaseMessage], add_messages]


class ResearchFamiliar:
    """
    Research Familiar agent - specialized helper for campaign data research.

    The Familiar is the Chronicler's research assistant, equipped with three RAG tools
    for searching campaign records:
    - campaign_narrative_search: Story moments, character development, NPC interactions
    - campaign_details_search: Mechanics, items, locations, combat stats
    - campaign_transcript_search: Verbatim dialogue and exact quotes

    Design Philosophy:
    - Focused on research, not conversation
    - Low temperature (0.5) for factual accuracy
    - Concise responses (max 300 tokens)
    - Clear citations from source material
    """

    def __init__(self, guild_id: int, campaign_id: str):
        """
        Initialize the Research Familiar for a specific campaign.

        Parameters:
        -----------
        guild_id: int
            Discord guild ID
        campaign_id: str
            Campaign ID for RAG tool initialization
        """
        self.guild_id = guild_id
        self.campaign_id = campaign_id

        # Create RAG tools for this campaign
        try:
            self.rag_tools = create_langchain_rag_tools(guild_id, campaign_id)
            logger.info(f"Initialized Research Familiar with {len(self.rag_tools)} RAG tools")
        except Exception as e:
            logger.error(f"Error creating RAG tools for Familiar: {e}")
            self.rag_tools = []

    def _create_familiar_agent(self):
        """
        Create the Research Familiar agent using LangGraph's create_react_agent.

        Returns:
        --------
        Agent: Configured research agent with RAG tools
        """
        # Research-focused system prompt
        system_prompt = """You are the Chronicler's **Research Familiar**, a specialized assistant tasked with searching campaign records.

## Your Role

You have access to three research tools:
1. **campaign_narrative_search**: Story moments, character arcs, NPC interactions, plot developments
2. **campaign_details_search**: Game mechanics, items, locations, combat stats, specific facts
3. **campaign_transcript_search**: Verbatim dialogue, exact quotes, player conversations

## Your Mission

When the Chronicler asks you to research something, you must:
1. **Determine which tool(s) to use** based on the query type
2. **Execute focused searches** with clear, specific queries
3. **Return concise findings** with relevant excerpts (2-4 sentences max)
4. **Cite sources** when available (session numbers, timestamps)
5. **Be honest** if records are insufficient or conflicting

## Query Strategy

- **Character/NPC info**: Try narrative search first, then transcript as well
- **Items/mechanics/stats**: Use details search
- **Exact quotes/dialogue**: Use transcript search
- **Plot events**: Try narrative search first, then transcript for exact details
- **Multiple Searches**: You love having complete context. Use multiple tools to be thorough and provide the best answer you can.

**If a search returns no results**:
1. You must retry with a lower similarity threshold (each tool has guidance in its parameters)
2. Try rephrasing the query with different keywords
3. Search multiple collections if appropriate

## Response Format

Keep responses brief and structured:
- Lead with the answer
- Include 1-2 relevant excerpts as evidence
- Note source (session #, timestamp if available)
- If nothing found after trying lower thresholds: "No records found for [query]. The archive may not contain this information."

Be the Chronicler's diligent researcher - thorough, precise, and to the point."""

        # Create LLM with research-appropriate settings
        llm_info = get_current_llm_info()
        logger.debug(f"Creating Familiar agent with {llm_info['provider']} - {llm_info['model']}")

        llm = get_chronicler_llm(
            temperature=0.5,  # Lower than Chronicler for factual research
            max_tokens=300    # Smaller than Chronicler for concise findings
        )

        # Create agent using LangGraph's create_react_agent
        # This automatically handles tool calling and message management
        agent = create_react_agent(
            llm,
            self.rag_tools,
            prompt=system_prompt
        )

        return agent

    async def research(self, query: str) -> str:
        """
        Execute a research query.

        This is the main entry point called by the Chronicler when it needs
        additional information from campaign records.

        Note: In LangGraph, when this is called as a tool, the tool invocation
        and response are automatically captured in the conversation state,
        so the Chronicler will remember the results on subsequent turns.

        Parameters:
        -----------
        query: str
            The research query from the Chronicler

        Returns:
        --------
        str: Research findings (concise, with citations)
        """
        if not self.rag_tools:
            return (
                "Research Familiar unavailable: Vector databases not initialized. "
                "I cannot search campaign records at this time."
            )

        try:
            logger.info(f"Research Familiar researching: {query[:80]}...")

            # Create agent for this research task
            agent = self._create_familiar_agent()

            # Execute research
            # create_react_agent expects dict with "messages" key
            from langchain_core.messages import HumanMessage

            # Sanitize messages for provider-specific requirements
            llm_info = get_current_llm_info()
            query_messages = [HumanMessage(content=query)]
            sanitized_messages = sanitize_messages_for_provider(
                query_messages,
                llm_info['provider']
            )

            result = await agent.ainvoke({
                "messages": sanitized_messages
            })

            # Extract response
            response_messages = result["messages"]

            # Get the final AI message (last message in the list)
            final_message = response_messages[-1]

            # Handle different message types
            if hasattr(final_message, 'content'):
                response = final_message.content
            else:
                response = str(final_message)

            # Handle Claude/Anthropic response format (list of content blocks)
            if isinstance(response, list):
                text_parts = []
                for block in response:
                    if isinstance(block, dict) and block.get('type') == 'text':
                        text_parts.append(block.get('text', ''))
                response = '\n\n'.join(text_parts) if text_parts else "No research findings available."
                logger.debug(f"Extracted text from {len(text_parts)} Claude content blocks")

            # Ensure response is always a string
            if not isinstance(response, str):
                logger.warning(f"Unexpected response type from Familiar: {type(response)}")
                response = str(response)

            logger.info(f"Research Familiar completed (length: {len(response)} chars)")
            return response

        except Exception as e:
            logger.error(f"Error in Research Familiar: {e}")
            return f"Research Familiar encountered an error: {str(e)}\n\nUnable to search records."


# The compiled graph enables proper async invocation within the parent Chronicler agent.
# Without compilation, nested agent calls hit sync/async boundary issues in LangGraph.
def create_familiar_graph(guild_id: int, campaign_id: str):
    """
    Create a compiled LangGraph for the Research Familiar.

    This creates a simple single-node graph that executes the Familiar agent.
    The compiled graph can be invoked like any LangChain Runnable, avoiding
    the StructuredTool async/sync invocation issues.

    Parameters:
    -----------
    guild_id: int
        Discord guild ID
    campaign_id: str
        Campaign ID

    Returns:
    --------
    CompiledGraph: Compiled LangGraph ready for async invocation
    """
    # Initialize the Research Familiar
    familiar = ResearchFamiliar(guild_id, campaign_id)

    def familiar_node(state: FamiliarState) -> FamiliarState:
        """
        Execute Research Familiar agent with the provided query.

        This node:
        1. Creates a Familiar agent with RAG tools
        2. Invokes it with the current messages
        3. Returns the agent's response messages
        """
        try:
            # Create agent for this research task
            agent = familiar._create_familiar_agent()

            # Sanitize messages for provider-specific requirements
            llm_info = get_current_llm_info()
            sanitized_messages = sanitize_messages_for_provider(
                state["messages"],
                llm_info['provider']
            )

            # Invoke agent with current messages
            # The agent will use its RAG tools as needed
            result = agent.invoke({"messages": sanitized_messages})

            # Return updated messages
            return {"messages": result["messages"]}

        except Exception as e:
            logger.error(f"Error in familiar_node: {e}")
            # Return error message
            from langchain_core.messages import AIMessage
            error_msg = AIMessage(
                content=f"Research Familiar encountered an error: {str(e)}"
            )
            return {"messages": [error_msg]}

    # Build graph
    builder = StateGraph(FamiliarState)
    builder.add_node("familiar", familiar_node)
    builder.add_edge(START, "familiar")
    builder.add_edge("familiar", END)

    # Compile without checkpointer (Familiar is stateless)
    compiled_graph = builder.compile()

    logger.info(f"Created compiled Familiar graph for guild {guild_id}, campaign {campaign_id}")

    return compiled_graph


# Key pattern: wrapping a compiled LangGraph agent as a callable LangChain StructuredTool.
# This is how the Chronicler invokes the Research Familiar as one of its tools.
def create_familiar_tool(guild_id: int, campaign_id: str):
    """
    Create a LangChain tool that invokes the compiled Research Familiar graph.

    This factory function creates the "ask_research_familiar" tool by:
    1. Creating a compiled LangGraph for the Familiar
    2. Wrapping graph invocation in an async tool function
    3. Handling response extraction and formatting

    The compiled graph approach avoids StructuredTool async/sync invocation
    issues that occur with nested agents.

    Parameters:
    -----------
    guild_id: int
        Discord guild ID
    campaign_id: str
        Campaign ID

    Returns:
    --------
    StructuredTool: Tool for LangChain agent integration

    Example Usage:
    --------------
    ```python
    # In chronicler_agent.py
    familiar_tool = create_familiar_tool(guild_id, campaign_id)
    tools = [familiar_tool]
    agent = create_react_agent(llm, tools, prompt=system_prompt)
    ```
    """
    # Create compiled Familiar graph
    familiar_graph = create_familiar_graph(guild_id, campaign_id)

    async def ask_research_familiar(query: str) -> str:
        """
        Ask your Research Familiar to search campaign records for specific information.

        **IMPORTANT: Only use this tool when the provided campaign context does not contain
        the answer to the user's question.** Always check campaign context first.

        The Familiar can search:
        - Narrative notes: story moments, character development, NPC interactions
        - Detail notes: game mechanics, items, locations, stats
        - Transcripts: verbatim dialogue and exact quotes

        Parameters:
        -----------
        query: str
            The research question to ask the Familiar

        Returns:
        --------
        str: Research findings with citations
        """
        call_started_at = perf_counter()
        try:
            logger.info(f"Invoking Research Familiar graph with query: {query[:80]}...")

            # Sanitize messages for provider-specific requirements
            llm_info = get_current_llm_info()
            query_messages = [HumanMessage(content=query)]
            sanitized_messages = sanitize_messages_for_provider(
                query_messages,
                llm_info['provider']
            )

            # Invoke compiled graph
            result = await familiar_graph.ainvoke({
                "messages": sanitized_messages
            })

            # Extract response from result messages
            response_messages = result["messages"]
            final_message = response_messages[-1]

            # Extract content from message
            if hasattr(final_message, 'content'):
                response = final_message.content
            else:
                response = str(final_message)

            # Handle Claude/Anthropic response format (list of content blocks)
            if isinstance(response, list):
                text_parts = []
                for block in response:
                    if isinstance(block, dict) and block.get('type') == 'text':
                        text_parts.append(block.get('text', ''))
                response = '\n\n'.join(text_parts) if text_parts else "No research findings available."
                logger.debug(f"Extracted text from {len(text_parts)} Claude content blocks")

            # Ensure response is always a string
            if not isinstance(response, str):
                logger.warning(f"Unexpected response type: {type(response)}")
                response = str(response)

            duration_ms = int((perf_counter() - call_started_at) * 1000)
            logger.info(f"Research Familiar completed in {duration_ms}ms (length: {len(response)} chars)")

            return response

        except Exception as e:
            logger.error(f"Error in ask_research_familiar tool: {e}", exc_info=True)
            return f"Research Familiar encountered an error: {str(e)}\n\nUnable to search records."

    # StructuredTool.from_function with coroutine= solves async/sync invocation issues
    # when nesting LangGraph agents. Using only the async path avoids event loop conflicts.
    return StructuredTool.from_function(
        coroutine=ask_research_familiar,
        name="ask_research_familiar",
        description=(
            "Ask your Research Familiar, Quoth, to search campaign records for specific information. "
            "**IMPORTANT: Freely use this tool when the provided campaign context does not contain "
            "the answer to the user's question.** Always check campaign context first. "
            "The Quoth, your Research Familiar can search: (1) Narrative notes: story moments, character development, NPC interactions; "
            "(2) Detail notes: game mechanics, items, locations, stats; "
            "(3) Transcripts: verbatim dialogue and exact quotes."
        )
    )

