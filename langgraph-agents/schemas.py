# Pydantic models for the character appearance system.
# Structured data enables intelligent art prompt generation — Aurora reads these models
# and adapts physical features to scene context rather than blindly copying descriptions.
# The 7-section framework (identity, face, eyes, hair, body, gear, personality) ensures
# visual consistency across generated images.
"""
Pydantic Schemas for Character Appearance

This module defines structured character appearance data for consistent AI art generation.
Based on the 7-section character description framework.
"""

from typing import Optional
from pydantic import BaseModel, Field


class IdentityAnchors(BaseModel):
    """
    Core identifying features that should appear in every character image.

    These are the fundamental traits that define who the character is at a glance.
    All fields except name and species are optional to allow flexible character creation.
    """
    name: str = Field(description="Character name or handle")
    gender_presentation: Optional[str] = Field(
        default=None,
        description="Gender presentation: masculine, feminine, androgynous, other"
    )
    age_descriptor: Optional[str] = Field(
        default=None,
        description="Age: exact number or range like 'late 20s', 'ancient', 'adolescent'"
    )
    species: str = Field(
        default="human",
        description="Species/race: human, elf, dwarf, halfling, orc, tiefling, dragonborn, aasimar, etc."
    )
    heritage_details: Optional[str] = Field(
        default=None,
        description="Lineage, clan, infernal mark, draconic ancestry color, fey-touched, etc."
    )
    height: Optional[str] = Field(
        default=None,
        description="Height descriptor: short, average, tall + approximate measurement"
    )
    build: Optional[str] = Field(
        default=None,
        description="Build: lean, athletic, stocky, curvy, lanky, burly"
    )
    skin_tone: Optional[str] = Field(
        default=None,
        description="Skin tone: fair, olive, tan, deep, ebony + undertone (warm/cool/neutral)"
    )
    distinctive_silhouette: Optional[str] = Field(
        default=None,
        description="Distinctive silhouette features: broad shoulders, long limbs, hunched posture, etc."
    )

    class Config:
        validate_assignment = True


class FaceHead(BaseModel):
    """
    Facial structure and head features.

    Defines the character's facial features and structure that remain consistent
    across different images.
    """
    face_shape: Optional[str] = Field(
        default=None,
        description="Face shape: oval, heart, square, angular, round"
    )
    jawline: Optional[str] = Field(
        default=None,
        description="Jawline: sharp, soft, pronounced, narrow"
    )
    cheekbones: Optional[str] = Field(
        default=None,
        description="Cheekbones: high, subtle, prominent"
    )
    nose: Optional[str] = Field(
        default=None,
        description="Nose: straight, hooked, broad, small, aquiline"
    )
    lips: Optional[str] = Field(
        default=None,
        description="Lips: thin, full, cupid's bow, downturned"
    )
    marks_and_moles: Optional[str] = Field(
        default=None,
        description="Freckles, moles, marks with location (e.g., 'mole under left eye')"
    )
    scars: Optional[str] = Field(
        default=None,
        description="Scars: shape + placement (e.g., 'thin scar through right eyebrow')"
    )
    ears: Optional[str] = Field(
        default=None,
        description="Ears: human, pointed, long, notched, pierced"
    )
    facial_hair: Optional[str] = Field(
        default=None,
        description="Facial hair: stubble, short beard, braided beard, mustache style"
    )

    class Config:
        validate_assignment = True


class Eyes(BaseModel):
    """
    Eye features including color, shape, and emotional presence.

    Combines hard descriptors (color, shape) with soft vibe tags to capture
    the character's expressive nature.
    """
    color: Optional[str] = Field(
        default=None,
        description="Eye color: emerald, amber, ice-blue, heterochromia (specify which eye)"
    )
    shape: Optional[str] = Field(
        default=None,
        description="Eye shape: almond, round, hooded, upturned, deep-set"
    )
    brows: Optional[str] = Field(
        default=None,
        description="Brows: thick, thin, arched, straight + grooming"
    )
    soft_vibe_tags: Optional[list[str]] = Field(
        default=None,
        description="Soft eye vibe tags (1-3): smoldering, mischievous, kind, predatory, tired-but-bright, haunted, playful, regal, wide-eyed wonder, piercing, dreamy, calculating"
    )

    class Config:
        validate_assignment = True


class Hair(BaseModel):
    """
    Hair color, style, texture, and signature details.

    Defines the character's hairstyle and distinctive hair features that help
    maintain visual consistency.
    """
    color: Optional[str] = Field(
        default=None,
        description="Hair color including undertone (e.g., 'ash brown', 'warm chestnut')"
    )
    length: Optional[str] = Field(
        default=None,
        description="Hair length: buzz, short, shoulder, waist-length"
    )
    texture: Optional[str] = Field(
        default=None,
        description="Hair texture: straight, wavy, curly, coiled"
    )
    style: Optional[str] = Field(
        default=None,
        description="Hair style: undercut, side part, messy bun, braided crown, twin braids, ponytail, loose waves"
    )
    signature_detail: Optional[str] = Field(
        default=None,
        description="Signature detail: white streak, shaved rune line, beadwork, ribbon color"
    )
    bangs: Optional[str] = Field(
        default=None,
        description="Hairline/bangs: yes/no + style"
    )

    class Config:
        validate_assignment = True


class BodyDetails(BaseModel):
    """
    Body physique, features, and decorative elements.

    Covers overall body type, notable physical features, hands, tattoos, piercings,
    and species-specific features like horns, tails, scales, or magical auras.
    """
    body_type: Optional[str] = Field(
        default=None,
        description="Overall body type and physique: athletic with defined muscles, curvy hourglass, lean and wiry, stocky and muscular, etc."
    )
    notable_features: Optional[str] = Field(
        default=None,
        description="Notable body features: broad shoulders, long legs, muscular arms, narrow waist, etc."
    )
    hands: Optional[str] = Field(
        default=None,
        description="Hands: delicate, calloused, clawed, gloved"
    )
    tattoos_runes: Optional[str] = Field(
        default=None,
        description="Tattoos/runes: location + motif"
    )
    piercings: Optional[str] = Field(
        default=None,
        description="Piercings: location + metal type (e.g., 'nose ring, silver')"
    )
    fantasy_features: Optional[str] = Field(
        default=None,
        description="Fantasy features: horns, tail, scales, fangs with details (shape, color, length)"
    )
    aura_magic_residue: Optional[str] = Field(
        default=None,
        description="Aura/magic residue: subtle glow, frost breath, ember motes, shadow haze"
    )

    class Config:
        validate_assignment = True


class GearOutfit(BaseModel):
    """
    Clothing and equipment with integrated material descriptions.

    Defines core equipment and clothing that help maintain character identity across
    different images. Materials are integrated into equipment descriptions.
    """
    clothing_style: Optional[str] = Field(
        default=None,
        description="Primary clothing style: noble, ranger leathers, pirate captain, scholar robes, street rogue"
    )
    color_palette: Optional[list[str]] = Field(
        default=None,
        description="2-4 core colors (e.g., ['black', 'deep purple', 'silver accents'])"
    )
    equipment: Optional[list[str]] = Field(
        default=None,
        description="Equipment with materials integrated: 'steel rapier with leather-wrapped hilt', 'worn leather cloak with silver clasp', 'iron-banded oak shield', 'bronze amulet'"
    )
    condition: Optional[str] = Field(
        default=None,
        description="Overall condition: pristine, travel-worn, patched, bloodstained, well-maintained"
    )

    class Config:
        validate_assignment = True


class PersonalityPresence(BaseModel):
    """
    Visual personality markers and presence.

    Captures the character's vibe, posture, and energy through visual cues that
    convey personality without dialogue.
    """
    core_vibe: Optional[list[str]] = Field(
        default=None,
        description="Core vibe (1-2): jaunty, grim, serene, cocky, nervy, brooding, radiant, feral, regal, bookish, roguish"
    )
    posture: Optional[str] = Field(
        default=None,
        description="Posture: upright military, relaxed slouch, prowling, dancer-light, guarded"
    )
    default_expression: Optional[str] = Field(
        default=None,
        description="Default expression: half-smile, scowl, calm neutrality, amused smirk"
    )
    energy: Optional[str] = Field(
        default=None,
        description="Energy: welcoming, intimidating, chaotic, disciplined, magnetic"
    )

    class Config:
        validate_assignment = True


class CharacterAppearance(BaseModel):
    """
    Complete character appearance schema combining all 7 sections.

    This is the main model for character appearance data. Only the identity section
    is required, all others are optional to allow flexible character creation.

    The to_prompt_string() method formats this data into a prompt suitable for
    AI image generation.
    """
    identity: IdentityAnchors = Field(description="Core identifying features (required)")
    face_head: Optional[FaceHead] = Field(default=None, description="Facial structure and features")
    eyes: Optional[Eyes] = Field(default=None, description="Eye features and expression")
    hair: Optional[Hair] = Field(default=None, description="Hair color, style, and details")
    body_details: Optional[BodyDetails] = Field(default=None, description="Body physique and decorative features")
    gear_outfit: Optional[GearOutfit] = Field(default=None, description="Clothing and equipment")
    personality_presence: Optional[PersonalityPresence] = Field(default=None, description="Visual personality markers")

    class Config:
        validate_assignment = True

    # Converts structured appearance data into natural language for image generation models.
    # Physical features are always included; gear/pose adapt to scene context via Aurora.
    def to_prompt_string(self) -> str:
        """
        Format character appearance as an AI image generation prompt.

        Combines all character appearance sections into a detailed prompt
        suitable for AI image generation.

        Returns:
        --------
        str
            Formatted prompt string for AI image generation
        """
        sections = []

        # Section 1: Identity Anchors (always included)
        identity_parts = [f"Name: {self.identity.name}"]
        if self.identity.species:
            identity_parts.append(f"{self.identity.species}")
        if self.identity.gender_presentation:
            identity_parts.append(f"{self.identity.gender_presentation} presentation")
        if self.identity.age_descriptor:
            identity_parts.append(f"age {self.identity.age_descriptor}")
        if self.identity.heritage_details:
            identity_parts.append(f"{self.identity.heritage_details}")
        if self.identity.height and self.identity.build:
            identity_parts.append(f"{self.identity.height}, {self.identity.build} build")
        elif self.identity.height:
            identity_parts.append(f"{self.identity.height}")
        elif self.identity.build:
            identity_parts.append(f"{self.identity.build} build")
        if self.identity.skin_tone:
            identity_parts.append(f"{self.identity.skin_tone} skin")
        if self.identity.distinctive_silhouette:
            identity_parts.append(f"{self.identity.distinctive_silhouette}")

        sections.append(", ".join(identity_parts))

        # Section 2: Face & Head
        if self.face_head:
            face_parts = []
            if self.face_head.face_shape:
                face_parts.append(f"{self.face_head.face_shape} face")
            if self.face_head.jawline:
                face_parts.append(f"{self.face_head.jawline} jawline")
            if self.face_head.cheekbones:
                face_parts.append(f"{self.face_head.cheekbones} cheekbones")
            if self.face_head.nose:
                face_parts.append(f"{self.face_head.nose} nose")
            if self.face_head.lips:
                face_parts.append(f"{self.face_head.lips} lips")
            if self.face_head.marks_and_moles:
                face_parts.append(f"{self.face_head.marks_and_moles}")
            if self.face_head.scars:
                face_parts.append(f"{self.face_head.scars}")
            if self.face_head.ears:
                face_parts.append(f"{self.face_head.ears} ears")
            if self.face_head.facial_hair:
                face_parts.append(f"{self.face_head.facial_hair}")

            if face_parts:
                sections.append(", ".join(face_parts))

        # Section 3: Eyes
        if self.eyes:
            eye_parts = []
            if self.eyes.color:
                eye_parts.append(f"{self.eyes.color} eyes")
            if self.eyes.shape:
                eye_parts.append(f"{self.eyes.shape} eye shape")
            if self.eyes.brows:
                eye_parts.append(f"{self.eyes.brows} brows")
            if self.eyes.soft_vibe_tags:
                eye_parts.append(f"eyes: {', '.join(self.eyes.soft_vibe_tags)}")

            if eye_parts:
                sections.append(", ".join(eye_parts))

        # Section 4: Hair
        if self.hair:
            hair_parts = []
            if self.hair.color:
                hair_parts.append(f"{self.hair.color} hair")
            if self.hair.length:
                hair_parts.append(f"{self.hair.length} length")
            if self.hair.texture:
                hair_parts.append(f"{self.hair.texture}")
            if self.hair.style:
                hair_parts.append(f"styled {self.hair.style}")
            if self.hair.signature_detail:
                hair_parts.append(f"{self.hair.signature_detail}")
            if self.hair.bangs:
                hair_parts.append(f"{self.hair.bangs}")

            if hair_parts:
                sections.append(", ".join(hair_parts))

        # Section 5: Body Details
        if self.body_details:
            body_parts = []
            if self.body_details.body_type:
                body_parts.append(f"{self.body_details.body_type}")
            if self.body_details.notable_features:
                body_parts.append(f"{self.body_details.notable_features}")
            if self.body_details.hands:
                body_parts.append(f"{self.body_details.hands} hands")
            if self.body_details.tattoos_runes:
                body_parts.append(f"{self.body_details.tattoos_runes}")
            if self.body_details.piercings:
                body_parts.append(f"{self.body_details.piercings}")
            if self.body_details.fantasy_features:
                body_parts.append(f"{self.body_details.fantasy_features}")
            if self.body_details.aura_magic_residue:
                body_parts.append(f"{self.body_details.aura_magic_residue}")

            if body_parts:
                sections.append(", ".join(body_parts))

        # Section 6: Gear & Outfit
        if self.gear_outfit:
            gear_parts = []
            if self.gear_outfit.clothing_style:
                gear_parts.append(f"wearing {self.gear_outfit.clothing_style}")
            if self.gear_outfit.color_palette:
                gear_parts.append(f"colors: {', '.join(self.gear_outfit.color_palette)}")
            if self.gear_outfit.equipment:
                gear_parts.append(f"equipped with {', '.join(self.gear_outfit.equipment)}")
            if self.gear_outfit.condition:
                gear_parts.append(f"{self.gear_outfit.condition} condition")

            if gear_parts:
                sections.append(", ".join(gear_parts))

        # Section 7: Personality & Presence
        if self.personality_presence:
            presence_parts = []
            if self.personality_presence.core_vibe:
                presence_parts.append(f"{', '.join(self.personality_presence.core_vibe)} demeanor")
            if self.personality_presence.posture:
                presence_parts.append(f"{self.personality_presence.posture} posture")
            if self.personality_presence.default_expression:
                presence_parts.append(f"{self.personality_presence.default_expression} expression")
            if self.personality_presence.energy:
                presence_parts.append(f"{self.personality_presence.energy} energy")

            if presence_parts:
                sections.append(", ".join(presence_parts))

        # Combine all sections
        prompt = ". ".join(sections) + "."

        return prompt
