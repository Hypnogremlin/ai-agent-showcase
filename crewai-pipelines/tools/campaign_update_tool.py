# CrewAI tool for campaign world-building CRUD operations.
# Manages locations, factions, quests, world info, and campaign metadata.
# AI agents use this after session processing to persist new story elements.
# The Pydantic schemas define the contract between AI agents and the database layer.
"""
Campaign Update Tool for CrewAI

This module provides a custom tool for CrewAI agents to manage Campaign elements
in the XML database by interfacing with the CampaignService.
"""

import logging
from typing import Dict, List, Any, Optional, Union, Type, Literal
from enum import Enum

from pydantic import BaseModel, Field, validator
from crewai.tools import BaseTool

logger = logging.getLogger("CampaignUpdateTool")

class CampaignUpdateInput(BaseModel):
    """Input schema for CampaignUpdateTool."""
    guild_id: int = Field(..., description="Discord guild ID - identifies which Discord server the data belongs to")
    entity_type: Literal["campaign_info", "world", "location", "faction", "quest"] = Field(..., description="Type of campaign entity to work with")
    operation: Literal["create", "update", "delete"] = Field(..., description="Operation to perform on the entity")
    entity_id: Optional[str] = Field(None, description="Entity ID (required for 'update' and 'delete' operations except for 'campaign_info' and 'world')")
    entity_data: Optional[Dict[str, Any]] = Field(None, description="Entity data for create/update operations")
    campaign_id: Optional[str] = Field(None, description="Campaign ID (if None, will use active campaign)")
    parent_quest_id: Optional[str] = Field(None, description="Parent quest ID (for 'create' operation with 'quest' entity type)")
    
    @validator('entity_id')
    def validate_entity_id(cls, v, values):
        """Validate that entity_id is provided for update/delete operations when needed."""
        if 'operation' in values and values['operation'] in ["update", "delete"]:
            if 'entity_type' in values and values['entity_type'] not in ["campaign_info", "world"]:
                if not v:
                    raise ValueError(f"entity_id is required for {values['operation']} operations on {values['entity_type']}")
        return v
    
    @validator('entity_data')
    def validate_entity_data(cls, v, values):
        """Validate that entity_data is provided for create and update operations."""
        if 'operation' in values and values['operation'] in ["create", "update"]:
            if not v:
                raise ValueError(f"entity_data is required for {values['operation']} operations")
        return v

class CampaignUpdateTool(BaseTool):
    """Tool for managing Campaign elements in the XML database."""
    
    name: str = "Campaign Update Tool"
    description: str = """
    Create, update, or delete campaign elements including campaign info, world details, locations, factions, and quests.
    This tool helps you manage all aspects of the campaign world and storyline.
    
    You must specify:
    - guild_id: The Discord server ID
    - entity_type: What type of entity to work with (campaign_info, world, location, faction, quest)
    - operation: What operation to perform (create, update, delete)
    - entity_id: ID of the entity (required for update/delete operations on location, faction, quest)
    - campaign_id: Optional. If not provided, the active campaign will be used
    
    For all 'create' and 'update' operations, you must provide:
    - entity_data: Dictionary with data fields appropriate for the entity type
    
    For 'quest' operations:
    - parent_quest_id: Optional parent quest ID for creating subquests
    
    Specific entity_data fields for each entity_type:
    
    1. campaign_info:
       - setting: Campaign setting
       - theme: Campaign theme
       - summary: Campaign summary
       - system: Game system (D&D 5e, Pathfinder, etc.)
       - game_master: Name of the Game Master
       
    2. world:
       - name: Name of the world
       - description: Description of the world
       - setting: Setting of the world
       - technology_level: Technology level of the world
       
    3. location:
       - name: The name of the location
       - type: Type of location (city, dungeon, etc.)
       - region: Region where the location is situated
       - description: Description of the location
       - features: List of features or points of interest
       
    4. faction:
       - name: The name of the faction
       - description: Description of the faction
       - leadership: List of leaders with name and title
       - goals: List of faction goals
       - associated_npcs: List of NPCs associated with the faction
       
    5. quest:
       - title: The title of the quest
       - description: Description of the quest
       - giver: NPC or entity who gave the quest
       - status: Status of the quest (Active, Complete, Failed)
       - related_npcs: List of NPCs related to this quest
       - related_locations: List of locations related to this quest
    
    Examples:
    1. To update campaign information:
       guild_id=123456, entity_type="campaign_info", operation="update", entity_data={"setting": "Forgotten Realms", "theme": "Dark Fantasy"}
       
    2. To create a new location:
       guild_id=123456, entity_type="location", operation="create", entity_data={"name": "Waterdeep", "type": "City", "region": "Sword Coast", "description": "City of Splendors", "features": ["Harbor", "Castle Waterdeep"]}
       
    3. To update a quest status:
       guild_id=123456, entity_type="quest", operation="update", entity_id="quest-123", entity_data={"status": "Complete"}
    """
    
    args_schema: Type[BaseModel] = CampaignUpdateInput
    
    def _run(self, 
             guild_id: int,
             entity_type: str,
             operation: str,
             entity_id: Optional[str] = None,
             entity_data: Optional[Dict[str, Any]] = None,
             campaign_id: Optional[str] = None,
             parent_quest_id: Optional[str] = None) -> str:
        """
        Execute the Campaign update with the specified parameters.
        
        Returns:
            A string with the result of the update operation.
        """
        # Production: wraps XML-based database service for campaign entity CRUD operations
        raise NotImplementedError("Showcase only")
