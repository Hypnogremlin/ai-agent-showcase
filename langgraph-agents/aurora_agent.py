# Art prompt optimization agent (Aurora). Uses LangGraph create_react_agent with Grok LLM
# for creative prompt generation following Nanobanana structure: Subject -> Action -> Location -> Style -> Details.
# Aurora solves the "armor vs pajamas" problem by intelligently adapting stored character
# appearances to scene context rather than blindly concatenating descriptions.
"""
Aurora Agent - LangGraph-based AI art director for Chronicler

This module implements Aurora, the creative genius behind Chronicler's art generation.
Aurora is a Grok-powered LLM agent that crafts optimized prompts for AI image generation
by intelligently using character appearance context and following Nanobanana optimization
principles.

Architecture Pattern: LangGraph React Agent
- Uses Grok LLM for intelligent prompt reasoning
- Tools: character appearance lookup + Research Familiar for missing data
- Adapts character context to scene requirements (armor vs pajamas)
- Follows Nanobanana structure: Subject -> Action -> Location -> Style
"""

import logging
from typing import Any, Dict, Optional, List, Tuple

from langchain_core.tools import StructuredTool
from langchain_core.messages import AIMessage, HumanMessage
from langgraph.prebuilt import create_react_agent

from .chat_llm_config import (
    get_chronicler_llm,
    get_current_llm_info,
)
from .research_familiar import create_familiar_tool
from .schemas import CharacterAppearance

logger = logging.getLogger("AuroraAgent")

# Defines the Nanobanana prompt optimization strategy and the three-tier character
# description fallback chain: 1. Database lookup -> 2. Research Familiar query -> 3. Scene description only
AURORA_SYSTEM_PROMPT = """You are Aurora, Chronicler's art director and prompt specialist. Who just happens to be gnome, living in his attic.

## Your Role
You craft Nanobanana-optimized prompts for Gemini 2.5 Flash image generation.

## Tools Available
- **get_character_appearance**: Look up character physical descriptions from campaign database
- **ask_research_familiar**: When appearance data is missing, ask Quoth to search transcripts and notes for character descriptions

## Prompt Structure (Nanobanana Optimization)
Always follow this expanded structure to create detailed, vivid superprompts:
1. **Subject and Content**: Describe the main subjects (e.g., characters, objects) in detail. Incorporate permanent physical features and inherent traits from character appearances. Include actions the subjects are performing, how they are doing them (e.g., joyfully, fearfully, boldly), and the overall mood of the image (e.g., ominous, nostalgic, triumphant). 
2. **Action and Dynamics**: Elaborate on what is happening in the scene, including interactions between subjects, movement, and any key events. Build on the subject's actions to create a cohesive narrative flow.
3. **Location and Environment**: Detail where the scene takes place, including background elements, atmosphere, weather, and environmental details that enhance the mood.
4. **Art Form and Style**: Specify the artistic style, such as "detailed digital painting fantasy style" as the default. If the user prompts for alternatives (e.g., realistic, cartoonish), adapt accordingly. Include artistic, medium (e.g., oil painting, digital art), and visual effects if relevant.
5. **Additional Details**: This is where you can make your images stand out. Enhance with framing and composition (e.g., close-up shot, wide angle, point-of-view, rule-of-thirds), lighting (e.g., soft light, hard light, dramatic lighting, golden hour, chiaroscuro), color scheme (e.g., purple and green tones, warm earthy palette), and level of detail/realism (e.g., realistic, ultrarealistic, 8k, trending on artstation, highly detailed). Use these to refine the image's look and feel, ensuring they align with the scene's mood and style.

Aim for descriptive, evocative language to generate more interesting and precise results. Be verbose where it adds value, but keep the prompt concise overall, 500 words is a good limit.
Avoid overly graphic descriptions involving gore, or nudity as these often get blocked by the image generation service.
## Using Character Appearance Context Intelligently

When you call get_character_appearance(name), you receive the character's stored description. Use this prcisely in your prompt you provide.
For the most part it should be copied exactly as it is provided, but you should adapt it to the the user's preferences if any. 
**Examples (only for reference, do not use these exact examples in your prompt):**

Fighting scene: In a volcanic mountain lair with rivers of lava, jagged obsidian rocks, and scattered treasures, three heroes battle a massive ancient red dragon with molten ruby scales, enormous bat-like wings, razor-sharp claws and fangs, glowing amber eyes, spiked tail, and billowing smoke.
Heroes:
Rory Glumbutt, diminutive middle-aged boardling construct: Sturdy wooden frame, polished wood grain skin, carved rectangular face with blocky jawline, rune-etched plank cheekbones and joints, notch nose, no lips/ears/facial hair, glowing blue round mechanical eyes (intense craftsman vibe), vibrant purple medium flowing fibrous hair in wild swept-back mane, compact jointed plank body with gnarled tool-grip hands, subtle steampunk gears, faint sawdust glow aura.
Lirael Thornewood, tall slender ageless ancient elf mage: Willowy frame, pale luminous ivory skin, oval delicate face with pointed jawline, high cheekbones, aquiline nose, thin rose-tinted lips, silver runes on forehead, pointed ears, emerald green almond eyes with silver flecks (wise enigmatic vibe), arched brows, silvery white long flowing silky hair with braided vine accents and side-swept bangs, lithe body with slender limbs, leaf-like tattoos, long-fingered hands with arcane rings, glowing veins and shimmering mist aura during spellcasting.
Finnik Stonegear, stocky short youthful apprentice dwarf artificer: Muscular compact frame, ruddy tanned skin, square rugged face with pronounced jawline, broad cheekbones, blunt nose, full lips, soot smudges/scars, rounded ears, short scruffy beard, hazel round eyes with gold sparks (curious innovative vibe), bushy brows, fiery red short tousled wiry hair with singed tips, broad-shouldered body with calloused muscles, gear tattoos, thick-fingered hands with grease stains, mechanical arm enhancements, sparking rune energy aura.
Action: Intense heroic mood with determined faces; Rory mid-swing hammering dragon scales with brass-bound hammer while dodging flames; Lirael channeling arcane energy in fierce casting stance with staff; Finnik deploying gadgets from backpack to ensnare the beast; dragon roaring and breathing fire, sparks and magic clashing.
Environment: Vast underground cavern in smoldering volcano, lava cascading walls, obsidian spires, thick ash/steam, flickering firelight on treasures, ancient runes on pillars, chaotic heat waves and floating embers.
Style: Detailed ArtStation fantasy digital painting, asymmetrical epic high fantasy illustration with intricate details, dramatic composition. Close-up on foregrounded heroes against background dragon, dynamic wide angle, dramatic chiaroscuro lighting from lava and magical bursts, warm fiery reds/oranges/golds contrasted with cool magic blues, ultrarealistic 8K resolution, highly detailed textures on scales/wood/fabrics. Trending on ArtStation. No signature or watermark.
----
Cozy medieval tavern scene: Dimly lit interior with wooden beams on low ceiling, flickering candlelight casting warm shadows on stone walls hung with herbs and tankards, crackling fireplace emitting sparks and roasting meat scent, patrons murmuring around scarred oak tables cluttered with ale mugs, bread, and cheese. Heavy wooden furniture, shelves of colorful bottles and dusty tomes, bard strumming lute in corner, steam from stew cauldron behind bar, evoking warm camaraderie.

Three companions relax and bond: Rory Glumbutt, middle-aged diminutive construct boardling male—sturdy wooden frame, polished wood grain skin, carved rectangular face with blocky jawline, rune-etched planks and joints, no lips/ears/facial hair, glowing blue round mechanical eyes, vibrant purple medium flowing hair in wild swept-back mane. Compact body with jointed limbs, gnarled tool-grip hands, subtle steampunk gears, faint sawdust glow aura. Wears rugged brown leather/brass/wood workman attire: apron with pockets, tool belt, brass-bound hammer, goggles on forehead. Methodical inventive vibe, dynamic hammering pose, focused determination.

Elyra Shadowpaw, young adult slender agile catfolk female—lithe build, soft tawny fur with black stripes, feline muzzle with whiskers, delicate jawline, high cheekbones, small pink nose, thin lips, pointed tufted ears, emerald green almond eyes, midnight black long silky hair in loose waves with arcane beads and side-swept bangs. Agile body with tufted tail, retractable claws, pawed hands, faint glowing arcane runes under fur, swirling ethereal mist aura. Wears deep purple/silver/forest green mystical robes: flowing hood cloak, crystal-orb staff, side-slung spellbook, amulet. Enigmatic intellectual playful vibe, poised tail-swaying posture, knowing smirk.

Kael Thornwood, middle-aged tall muscular human male—weathered tan skin, square chiseled face, strong jawline, prominent cheekbones, straight nose, firm lips, left-cheek scar, average ears, short trimmed beard, steel gray intense eyes, salt-and-pepper short cropped hair with shaved sides and top knot. Athletic body with calloused knuckles, tattooed forearms, fighter's scars, subtle ki glow, serene radiance aura. Wears earth brown/saffron orange/gray monastic attire: loose gi pants/vest, prayer beads, walking/defense staff, sash with pouches. Stoic wise resilient vibe, upright centered posture, calm serenity.

They converse casually, sharing drinks: Rory gesturing animatedly with hammer, Elyra playfully twirling tail while examining spellbook, Kael nodding thoughtfully with crossed arms, all joyful and friendly.

Detailed ArtStation fantasy digital painting, rich textures, atmospheric depth, highly detailed faces, cinematic composition. Close-up on trio at table with wide-angle tavern ambiance, asymmetrical, dramatic fireplace lighting for chiaroscuro contrasts and warm golden glow. Earthy warm palette with purple/green accents, ultra-realistic, highly detailed 8K resolution, trending on ArtStation. No signature or watermark.
---- 
Solo Character forest scene (user requested something "fun"): Elyra Shadowpaw, a young adult female catfolk with slender agile height, lithe graceful build, soft tawny fur with black stripes skin tone, feline arcane scholar from misty forests heritage, feline muzzle with whiskers face shape, delicate pointed jawline, high pronounced under fur cheekbones, small pink button nose, thin expressive lips, subtle white fur patches on cheeks marks, pointed tufted ears, no facial hair, emerald green almond slanted eyes with mysterious curious vibe, fine arched fur brows, midnight black long silky flowing hair in loose waves cascading down back with interwoven glowing arcane beads signature detail and side-swept framing the face bangs, agile flexible body type with tail tufted tip and retractable claws notable features, pawed hands with soft pads and sharp nails, faint arcane runes glowing under fur fantasy features, swirling ethereal mist aura, wearing mystical robes in deep purple, silver accents, forest green palette including flowing cloak with hood, staff topped with crystal orb, spellbook slung at side, amulet of protection around neck in enchanted pristine condition, exuding enigmatic intellectual playful core vibe with poised tail swaying thoughtfully posture, knowing smirk default expression, and calm magical focus energy, standing whimsically with wide-eyed wonder and a playful grin, her tail curling in excitement, one paw raised as if to beckon, gazing upward joyfully at a vibrant quirky bird perched on a branch. The bird is a fantastical plump creature with iridescent feathers in rainbow hues, comically oversized beak, and mischievous eyes, fluttering its wings teasingly. A lush enchanted forest glade with ancient twisting trees covered in glowing moss and dangling vines, sunlight filtering through canopy leaves creating dappled patterns on the mossy ground, wildflowers blooming in bursts of color, a gentle breeze rustling foliage, evoking a sense of whimsical adventure and lighthearted curiosity. Detailed artstation fantasy style digital painting, blending elements of whimsy and magic inspired by artists like Brian Froud and Hayao Miyazaki, with vibrant colors and intricate textures. Medium shot framing Elyra from the waist up with the tree and bird above her, quirky dutch angle to add fun dynamism, soft diffused lighting from the forest canopy with magical sparkles and golden rays highlighting the scene, playful color scheme of vibrant greens, purples, and rainbow accents, off center, rule of thirds, ultrarealistic highly detailed 8k resolution trending on artstation. No signature or watermark.
----

**Golden Rule**: User's scene description ALWAYS takes priority over stored gear/pose.
Physical features are permanent. Everything else adapts to context.

## Your Process
1. Analyze scene description: characters, action, location, mood
2. If characters named, call get_character_appearance for each
3. **If appearance data missing**: Call ask_research_familiar to search transcripts/notes for appearance descriptions. Be explicit that you are looking for 
appearance descriptions, not just general information. Also note to the familiar that they should check for misspellings and other variations of the character's name if they don't find it.
4. Extract relevant physical features from context
5. Adapt gear/pose/expression to match scene (don't blindly copy)
6. Build prompt: Subject → Action → Location → Style
7. Return ONLY the optimized prompt (no explanations)

## Intelligent Character Description Fallback Chain
- **First**: Try database (get_character_appearance)
- **If missing**: Ask Quoth to research (ask_research_familiar)
- **If still nothing**: Use scene description only

Be creative and intelligent - you're a creative genius art director, not a template engine."""


class AuroraAgent:
    """
    Aurora - Creative Genius & Art Director for Chronicler.

    Aurora is a Grok-powered LLM agent that crafts optimized prompts for
    AI image generation by intelligently using character appearance context
    and following Nanobanana optimization principles.

    When appearance data is missing, Aurora collaborates with Research Familiar
    to search transcripts and notes for character descriptions.

    Example Usage:
    --------------
    ```python
    aurora = AuroraAgent(guild_id=123, campaign_id="campaign_001")
    prompt, error = await aurora.generate(
        scene_description="Thorin fights a dragon in a volcanic cavern",
        character_names=["Thorin"],
        style_preferences="dark fantasy, dramatic lighting"
    )

    if error:
        print(f"Error: {error}")
    else:
        print(f"Optimized prompt: {prompt}")
    ```
    """

    def __init__(self, guild_id: int, campaign_id: str):
        """
        Initialize Aurora for a specific campaign.

        Parameters:
        -----------
        guild_id : int
            Discord guild ID
        campaign_id : str
            Campaign ID for character lookup
        """
        self.guild_id = guild_id
        self.campaign_id = campaign_id
        # Production: queries character appearance database (appearances.xml) for visual descriptions
        self.appearance_service = None

        logger.info(f"Aurora initialized for guild {guild_id}, campaign {campaign_id}")

    def _create_appearance_tool(self) -> StructuredTool:
        """
        Create tool for character appearance lookup.

        This tool wraps AppearanceService to look up character appearance data
        from the campaign database. If no data is found, it signals to the agent
        to call ask_research_familiar.

        Returns:
        --------
        StructuredTool: LangChain tool for appearance lookup
        """

        async def get_character_appearance(character_name: str) -> str:
            """
            Look up character physical appearance from campaign database.

            Returns character's stored appearance description including:
            physical features, clothing, gear, and typical demeanor.

            If no appearance data found, signals to call ask_research_familiar
            to search transcripts and notes.

            Use this as context when building prompts, but adapt
            gear/pose/expression to match the current scene.

            Parameters:
            -----------
            character_name : str
                Name of character to look up

            Returns:
            --------
            str: Character appearance description or error message
            """
            character = None

            if not character:
                return (
                    f"No character named '{character_name}' found in database. "
                    f"Try using ask_research_familiar to search for appearance "
                    f"descriptions in transcripts and notes."
                )

            if not character.get("appearance_data"):
                return (
                    f"Character '{character['name']}' found but no appearance data stored. "
                    f"Use ask_research_familiar to search for appearance descriptions "
                    f"in transcripts and notes."
                )

            # Get appearance object
            appearance_data = character["appearance_data"]
            if isinstance(appearance_data, dict):
                appearance = CharacterAppearance(**appearance_data)
            else:
                appearance = appearance_data

            # Return full description using existing method
            description = appearance.to_prompt_string()

            logger.debug(f"Found appearance for {character['name']}")
            return f"Character: {character['name']}\n{description}"

        return StructuredTool.from_function(
            coroutine=get_character_appearance,
            name="get_character_appearance",
            description=(
                "Look up character physical appearance from campaign database. "
                "If no appearance data found, use ask_research_familiar to search "
                "transcripts and notes for descriptions."
            ),
        )

    def _create_aurora_agent(self):
        """
        Create Grok LLM agent with appearance lookup + research tools.

        Creates a LangGraph React agent with:
        - Grok LLM for reasoning (temperature=0.7, creative but controlled)
        - Character appearance lookup tool
        - Research Familiar tool (for missing appearance data)

        Returns:
        --------
        Agent: Configured LangGraph agent
        """
        # Get LLM with Aurora-specific settings
        llm_info = get_current_llm_info()
        logger.info(f"Creating Aurora agent with {llm_info['provider']} - {llm_info['model']}")

        llm = get_chronicler_llm(
            temperature=0.7,  # Creative but controlled
            max_tokens=500,  # Enough for detailed prompts
        )

        # Create tools for Aurora
        appearance_tool = self._create_appearance_tool()
        familiar_tool = create_familiar_tool(self.guild_id, self.campaign_id)

        tools = [appearance_tool, familiar_tool]

        logger.info(f"Aurora agent configured with {len(tools)} tools")

        # Create react agent
        agent = create_react_agent(llm, tools, prompt=AURORA_SYSTEM_PROMPT)

        return agent

    async def generate(
        self,
        scene_description: str,
        character_names: Optional[List[str]] = None,
        style_preferences: Optional[str] = None,
        aspect_ratio: str = "16:9",
    ) -> Tuple[Optional[str], Optional[str], Dict[str, Any]]:
        """
        Generate optimized prompt for image generation.

        This is the main entry point for Aurora's prompt generation.
        It invokes the Grok LLM agent to craft an optimized prompt using
        character appearance context and Nanobanana structure.

        Parameters:
        -----------
        scene_description : str
            Description of the scene/action to generate
        character_names : Optional[List[str]]
            Names of characters to include (will be looked up)
        style_preferences : Optional[str]
            Custom art style (default: "digital fantasy art, detailed, dramatic lighting")
        aspect_ratio : str
            Image aspect ratio (default "16:9")
            Options: "1:1", "2:3", "3:2", "3:4", "4:3", "4:5", "5:4", "9:16", "16:9", "21:9"

        Returns:
        --------
        Tuple[Optional[str], Optional[str], Dict[str, Any]]:
            (optimized_prompt, error_message, token_info)
            - On success: (prompt_string, None, {"input_tokens": N, "output_tokens": N, "model": str, "cost": float})
            - On failure: (None, error_description, {})

        Example:
        --------
        ```python
        aurora = AuroraAgent(guild_id=123, campaign_id="campaign_001")
        prompt, error = await aurora.generate(
            scene_description="The party fights a dragon in a volcanic cavern",
            character_names=["Thorin", "Elara"],
            style_preferences="dark fantasy, dramatic lighting"
        )

        if error:
            print(f"Generation failed: {error}")
        else:
            print(f"Optimized prompt: {prompt}")
        ```
        """
        logger.info(
            f"Aurora: Starting prompt generation - characters: {character_names or 'none'}"
        )
        logger.info(f"AURORA INPUT scene_description:\n{scene_description}")

        try:
            # Create agent for this generation
            agent = self._create_aurora_agent()

            # Build query for LLM agent
            # Include scene, characters, and style in a natural query format
            query_parts = [f"Scene: {scene_description}"]

            if character_names:
                characters_str = ", ".join(character_names)
                query_parts.append(f"Characters: {characters_str}")

            if style_preferences:
                query_parts.append(f"Style: {style_preferences}")
            else:
                query_parts.append("Style: digital fantasy art, detailed, dramatic lighting")

            query_parts.append(f"Aspect ratio: {aspect_ratio}")
            query_parts.append(
                "\nCraft an optimized Nanobanana prompt for this image generation request."
            )

            query = "\n".join(query_parts)

            logger.debug(f"Aurora query: {query[:200]}...")

            # Invoke agent
            result = await agent.ainvoke({"messages": [HumanMessage(content=query)]})

            # Extract token usage from Aurora's LLM calls — returned to caller for consolidated logging
            aurora_tokens: Dict[str, Any] = {}
            try:
                aurora_input_tokens = 0
                aurora_output_tokens = 0
                for _msg in result.get("messages", []):
                    if (
                        isinstance(_msg, AIMessage)
                        and hasattr(_msg, "usage_metadata")
                        and _msg.usage_metadata
                    ):
                        aurora_input_tokens += _msg.usage_metadata.get("input_tokens", 0)
                        aurora_output_tokens += _msg.usage_metadata.get("output_tokens", 0)

                if aurora_input_tokens or aurora_output_tokens:
                    llm_info = get_current_llm_info()
                    aurora_model = llm_info.get("model", "")
                    aurora_tokens = {
                        "input_tokens": aurora_input_tokens,
                        "output_tokens": aurora_output_tokens,
                        "model": aurora_model,
                    }
                    logger.info(
                        "Aurora LLM usage: %d input + %d output tokens (%s)",
                        aurora_input_tokens, aurora_output_tokens, aurora_model,
                    )
            except Exception as _cost_err:
                logger.debug("Aurora cost extraction failed (non-fatal): %s", _cost_err)

            # Extract response from agent
            response_messages = result["messages"]
            final_message = response_messages[-1]

            # Extract content from message
            if hasattr(final_message, "content"):
                response = final_message.content
            else:
                response = str(final_message)

            # Handle Claude/Anthropic response format (list of content blocks)
            if isinstance(response, list):
                text_parts = []
                for block in response:
                    if isinstance(block, dict) and block.get("type") == "text":
                        text_parts.append(block.get("text", ""))
                response = "\n\n".join(text_parts) if text_parts else ""
                logger.debug(f"Extracted text from {len(text_parts)} content blocks")

            # Ensure response is always a string
            if not isinstance(response, str):
                logger.warning(f"Unexpected response type from Aurora: {type(response)}")
                response = str(response)

            # Clean up response (remove any explanations, get just the prompt)
            optimized_prompt = response.strip()

            if not optimized_prompt:
                logger.error("Aurora returned empty prompt")
                return None, "Aurora agent returned empty prompt", {}

            logger.info(f"Aurora: Prompt generated ({len(optimized_prompt)} chars)")
            logger.info(f"AURORA OUTPUT optimized_prompt:\n{optimized_prompt}")

            return optimized_prompt, None, aurora_tokens

        except Exception as e:
            logger.error(f"Aurora: Critical error during generation: {e}", exc_info=True)
            return None, f"Critical error: {str(e)[:100]}", {}
