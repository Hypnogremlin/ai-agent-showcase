# Design Patterns Catalog

A reference of the key AI agent design patterns used in the DnD Chronicler system. Each pattern includes its purpose, where it is implemented, and a representative code snippet.

---

## 1. CrewAI @CrewBase with Per-Role LLM Overrides

**Description:** CrewAI's `@CrewBase` decorator provides a declarative way to define agents and tasks via YAML configuration. The `@before_kickoff` hook runs before crew execution, giving us a place to dynamically assign different LLM providers to different agent roles at runtime.

**Why it matters:** Not all agents need the same model. A context retrieval agent can use a cheaper, faster model while the synthesis agent uses a premium model for quality. Per-role overrides enable cost optimization without code changes -- just edit the JSON config.

**File:** `crewai-pipelines/session_crew.py` (lines 33-62), config in `crewai-pipelines/config/notes_llm_config.json`

```python
@before_kickoff
def prepare_inputs(self, inputs):
    # Build reverse map: YAML role text -> config key name
    role_text_to_key = {}
    for key, cfg in self.agents_config.items():
        role_text = cfg.get("role", "").strip()
        if role_text:
            role_text_to_key[role_text] = key

    for agent in self.agents:
        config_key = role_text_to_key.get(agent.role.strip(), agent.role)
        agent.llm = get_llm_for_role(config_key)
```

The `get_llm_for_role()` function checks `role_overrides` in the config JSON. If a role has an override (e.g., `"context_specialist": "backup"`), it uses that profile; otherwise it falls back to `current_mode`.

**Config example (`notes_llm_config.json`):**
```json
{
  "primary": { "provider": "anthropic", "model": "claude-haiku-4-5" },
  "backup":  { "provider": "xai", "model": "grok-4-1-fast" },
  "role_overrides": {
    "context_specialist": "backup",
    "narrative_analyst": "backup",
    "detail_analyst": "backup"
  }
}
```

This routes the expensive chronicler role to Claude while cheaper analysis roles use Grok.

---

## 2. CrewAI Flow for Multi-Phase Pipelines

**Description:** CrewAI's `Flow` class provides a state machine where `@start` and `@listen` decorators define a dependency graph between processing phases. A Pydantic `BaseModel` tracks state across phases, enabling token accumulation and intermediate file paths to persist between crews.

**Why it matters:** Long sessions require a two-phase pipeline (segment analysis then synthesis) with a rate-limit cooldown between them. The Flow pattern makes this explicit and type-safe, with the Pydantic state model serving as the contract between phases.

**File:** `crewai-pipelines/transcript_flow.py` (lines 32-44 for state, lines 86-129 for phase wiring)

```python
class TranscriptFlowState(BaseModel):
    """Pydantic state tracks progress and accumulates metrics across phases."""
    analysis_complete: bool = False
    synthesis_complete: bool = False
    narrative_path: str = ""
    detail_path: str = ""
    session_notes: str = ""
    segment_input_tokens: int = 0
    segment_output_tokens: int = 0
    synthesis_input_tokens: int = 0
    synthesis_output_tokens: int = 0


class DnDTranscriptFlow(Flow[TranscriptFlowState]):

    @start()
    def analyze_transcript_segments(self) -> Dict[str, Any]:
        """Phase 1: parallel segment analysis."""
        result = with_llm_fallback(segment_crew_factory, execute_segment_analysis, "segment analysis")
        self.state.narrative_path = result.get('narrative_path', '')
        self.state.segment_input_tokens = result.get('input_tokens', 0)
        # ... 60s cooldown ...
        return result

    @listen(analyze_transcript_segments)
    def synthesize_final_outputs(self, analysis_result) -> Dict[str, Any]:
        """Phase 2: runs only after analysis completes."""
        # Reads files from self.state.narrative_path / detail_path
        # Writes final notes to self.state.session_notes
```

After both phases complete, `process_session()` sums token counts from both state fields for a single cost total.

---

## 3. Three-Tier LLM Fallback

**Description:** The `with_llm_fallback()` wrapper implements a retry strategy for CrewAI crew execution: (1) try primary provider, (2) wait 60 seconds and retry primary, (3) switch to backup provider. Each retry creates a fresh crew instance so `@before_kickoff` re-assigns LLMs from the updated config.

**Why it matters:** LLM API providers have transient failures (rate limits, overloaded servers). This pattern ensures session processing completes even when the primary provider is down, without manual intervention.

**File:** `crewai-pipelines/llm_config.py` (lines 460-535)

```python
def with_llm_fallback(crew_factory, operation_func, operation_name):
    try:
        # Tier 1: Primary attempt
        crew = crew_factory()
        return operation_func(crew)
    except Exception as e:
        if should_switch_to_backup(e):
            time.sleep(60)  # Tier 2: Wait and retry primary
            try:
                retry_crew = crew_factory()  # Fresh instance -> @before_kickoff re-runs
                return operation_func(retry_crew)
            except Exception:
                # Tier 3: Switch config to backup, create fresh crew
                switch_to_backup_mode()
                backup_crew = crew_factory()  # Loads backup LLM via @before_kickoff
                return operation_func(backup_crew)
        else:
            raise  # Non-recoverable error
```

The fresh crew instance on each retry is critical. Since `@before_kickoff` reads `current_mode` from the config file, switching to backup mode only takes effect when a new crew is instantiated.

---

## 4. Agent-as-Tool (Nested Agent Invocation)

**Description:** A compiled LangGraph agent is wrapped as a LangChain `StructuredTool`, allowing a parent agent to invoke a child agent through its normal tool-calling mechanism. The child agent is compiled into a `StateGraph` first, then its `ainvoke()` method is wrapped in an async function exposed via `StructuredTool.from_function(coroutine=...)`.

**Why it matters:** The Chronicler agent needs to delegate research queries to the Research Familiar. Direct function calls between LangGraph agents hit async/sync boundary issues. The compiled graph + StructuredTool approach solves this by keeping everything in the async path.

**File:** `langgraph-agents/research_familiar.py` (lines 252-445)

```python
def create_familiar_graph(guild_id, campaign_id):
    """Step 1: Build a compiled StateGraph containing the Familiar agent."""
    familiar = ResearchFamiliar(guild_id, campaign_id)

    def familiar_node(state: FamiliarState) -> FamiliarState:
        agent = familiar._create_familiar_agent()
        result = agent.invoke({"messages": state["messages"]})
        return {"messages": result["messages"]}

    builder = StateGraph(FamiliarState)
    builder.add_node("familiar", familiar_node)
    builder.add_edge(START, "familiar")
    builder.add_edge("familiar", END)
    return builder.compile()  # Compiled graph is a Runnable


def create_familiar_tool(guild_id, campaign_id):
    """Step 2: Wrap the compiled graph as a StructuredTool."""
    familiar_graph = create_familiar_graph(guild_id, campaign_id)

    async def ask_research_familiar(query: str) -> str:
        result = await familiar_graph.ainvoke(
            {"messages": [HumanMessage(content=query)]}
        )
        return result["messages"][-1].content

    # coroutine= ensures only the async path is used, avoiding event loop conflicts
    return StructuredTool.from_function(
        coroutine=ask_research_familiar,
        name="ask_research_familiar",
        description="Ask your Research Familiar to search campaign records..."
    )
```

The parent Chronicler agent then uses this tool like any other:
```python
familiar_tool = create_familiar_tool(guild_id, campaign_id)
agent = create_react_agent(llm, [familiar_tool], prompt=system_prompt)
```

---

## 5. Smart Context Caching with Staleness Detection

**Description:** The Chronicler agent caches loaded campaign context in the LangGraph state with a timestamp. On each invocation, it checks three conditions before reloading: (1) campaign ID mismatch (rapid campaign switch), (2) missing timestamp, (3) context older than 1 minute. If none trigger, the cached context is reused.

**Why it matters:** Loading campaign context requires reading multiple XML files from disk. In a chat conversation with rapid successive messages, reloading on every turn wastes time. But the context must stay fresh enough that campaign edits (via slash commands) are reflected promptly.

**File:** `langgraph-agents/chronicler_agent.py` (lines 454-500)

```python
# State carries three cache-related fields
campaign_context = state.get("campaign_context") or {}
campaign_context_loaded_at = state.get("campaign_context_loaded_at") or ""
campaign_context_for_campaign_id = state.get("campaign_context_for_campaign_id") or ""

should_reload = False

# Check 1: Campaign mismatch (catches rapid campaign switches)
if campaign_context_for_campaign_id != state["campaign_id"]:
    should_reload = True
# Check 2: No timestamp
elif not campaign_context_loaded_at:
    should_reload = True
# Check 3: Staleness (>1 minute)
else:
    loaded_at = datetime.fromisoformat(campaign_context_loaded_at)
    if (datetime.now() - loaded_at) > timedelta(minutes=1):
        should_reload = True

if should_reload:
    campaign_context = await context_builder.load_campaign_context(
        state["guild_id"], state["campaign_id"]
    )
    campaign_context_loaded_at = datetime.now().isoformat()
```

The three fields (`campaign_context`, `campaign_context_loaded_at`, `campaign_context_for_campaign_id`) are defined in `langgraph-agents/graph_state.py` as part of the `ChroniclerState` TypedDict.

---

## 6. Sliding Window Message Retention

**Description:** After each agent invocation, the conversation history is trimmed to the last 20 messages before checkpoint persistence. Base64 image data is also stripped from messages post-processing.

**Why it matters:** LangGraph's SQLite checkpointer persists the full message list at every node. Without trimming, conversation history grows unboundedly, inflating database size and increasing context window costs on subsequent turns. The 20-message window keeps the last several exchanges while discarding stale history.

**File:** `langgraph-agents/chronicler_agent.py` (lines 554-563)

```python
# After agent invocation, trim before checkpoint save
final_messages = response_messages
if len(response_messages) > 20:
    final_messages = response_messages[-20:]
    trimmed = True
    logger.info("Trimmed checkpoint to last 20 messages")

# Strip base64 images (LLM has already processed them)
final_messages = strip_base64_from_messages(final_messages)
```

The trimming happens after the agent has generated its response (so the full history was available during reasoning) but before the state is returned (so the checkpointer only stores the trimmed window).

---

## 7. Content-Aware RAG with Campaign Isolation

**Description:** The RAG system creates three ChromaDB collections per campaign, each with chunk sizes optimized for its content type. Collection names include a sanitized campaign ID for isolation. A dual tool interface wraps the same vector database for both CrewAI and LangChain consumers.

**Why it matters:** Narrative text needs larger chunks (600 tokens) to preserve story flow across paragraphs. Transcript dialogue needs smaller chunks (300 tokens) because individual speaker turns are short. Campaign isolation prevents cross-contamination between different game groups sharing the same bot instance.

**File:** `rag-system/rag_utils.py` (lines 77-355), `rag-system/rag_tool_wrappers.py` (lines 24-182)

```python
# Three collections with content-aware chunk sizes
narratives_config = {"chunker": {"chunk_size": 600, "chunk_overlap": 100}}  # Story flow
details_config    = {"chunker": {"chunk_size": 400, "chunk_overlap": 80}}   # Mechanics/stats
transcript_config = {"chunker": {"chunk_size": 300, "chunk_overlap": 60}}   # Dialogue matching

# Campaign-isolated collection names
sanitized_id = _sanitize_collection_name(campaign_id)  # Handles unicode campaign names
narratives_collection_name = f"campaign_{sanitized_id}_narratives"
details_collection_name    = f"campaign_{sanitized_id}_details"
transcript_collection_name = f"campaign_{sanitized_id}_transcripts"
```

The bridge layer (`rag_tool_wrappers.py`) wraps CrewAI `RagTool._run()` inside LangChain `@tool` functions via closure:

```python
@tool
def campaign_narrative_search(query: str, similarity_threshold: float = 0.5, limit: int = 5) -> str:
    """LangChain tool that delegates to the CrewAI RagTool."""
    return narratives_rag._run(query=query, similarity_threshold=similarity_threshold, limit=limit)
```

---

## 8. YAML-Driven Agent Configuration

**Description:** Agent personalities, roles, goals, and backstories are externalized to YAML files. Task descriptions and expected output formats live in separate YAML files. CrewAI's `@CrewBase` decorator loads these automatically via the `agents_config` and `tasks_config` class attributes.

**Why it matters:** Separating agent behavior from code enables prompt iteration without touching Python. Different crews can share agent definitions (e.g., `chronicler` appears in both `agents.yaml` and `synthesis_agents.yaml`). Task templates use `{transcript}`, `{context}`, and `{segment_index}` placeholders that CrewAI interpolates at runtime.

**Files:** `crewai-pipelines/config/agents.yaml`, `crewai-pipelines/config/segment_agents.yaml`, `crewai-pipelines/config/synthesis_agents.yaml`, `crewai-pipelines/config/segment_tasks.yaml`

```yaml
# crewai-pipelines/config/segment_agents.yaml
narrative_analyst:
  role: >
    TTRPG Session Narrative & Social Analyst
  goal: >
    Analyze session transcript segments to extract story progression, character
    developments, social interactions, and memorable moments.
  backstory: >
    You are a veteran game master with a keen eye for both storytelling and
    group dynamics in tabletop roleplaying games...
  max_rpm: 3

detail_analyst:
  role: >
    TTRPG Session Detail & Mechanics Analyst
  goal: >
    Extract precise game details from session transcript segments, including
    mechanics usage, combat specifics, treasure and loot acquisition...
  backstory: >
    You are a meticulous rules expert and treasurer with years of experience...
  max_rpm: 3
```

The crew class references these with a simple path:
```python
@CrewBase
class SegmentAnalysisCrew:
    agents_config = 'config/segment_agents.yaml'
    tasks_config = 'config/segment_tasks.yaml'
```

---

## 9. Multi-Provider LLM Factory

**Description:** Two factory functions produce LLM instances for five providers (Anthropic, OpenAI, Google Gemini, XAI Grok, Vertex AI). One factory (`llm_config.py`) produces CrewAI `LLM` objects for batch processing. The other (`chat_llm_config.py`) produces LangChain `BaseChatModel` objects for interactive chat. Both read from JSON config files with the same structure (`primary`, `backup`, `current_mode`).

**Why it matters:** The project needs to switch providers without code changes (JSON config only), support different models for different use cases (batch vs. chat), and handle provider-specific quirks (e.g., Gemini's base URL format, XAI's LiteLLM prefix). A single factory per framework centralizes this complexity.

**Files:** `crewai-pipelines/llm_config.py` (lines 109-294), `langgraph-agents/chat_llm_config.py` (lines 60-244)

**CrewAI factory** (produces `crewai.LLM` instances):
```python
def get_current_llm(profile=None) -> LLM:
    settings = load_llm_settings()
    config = settings["backup"] if resolved_profile == "backup" else settings["primary"]

    if config["provider"] == "xai":
        return LLM(model=f"xai/{config['model']}", api_key=os.getenv("XAI_API_KEY"))
    elif config["provider"] == "google":
        return LLM(model=config["model"], base_url=f"{base_url}/models/{model_name}")
    elif config["provider"] == "anthropic":
        return LLM(model=config["model"], base_url="https://api.anthropic.com")
    # ... vertex_ai, openai
```

**LangChain factory** (produces `BaseChatModel` subclasses):
```python
def get_chronicler_llm(temperature=0.7, max_tokens=1500) -> BaseChatModel:
    settings = load_llm_settings()
    config = settings["backup"] if mode == "backup" else settings["primary"]

    if config["provider"] == "xai":
        return ChatXAI(model=config["model"], temperature=temperature, streaming=True)
    elif config["provider"] == "google":
        return ChatGoogleGenerativeAI(model=model_name, temperature=temperature)
    elif config["provider"] == "anthropic":
        return ChatAnthropic(model=config["model"], temperature=temperature)
    # ... vertex_ai, openai
```

The separation is deliberate: CrewAI uses LiteLLM internally (requiring provider prefixes like `xai/`), while LangChain uses native provider packages with their own initialization conventions.
