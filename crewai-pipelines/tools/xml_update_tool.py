# Generic XML update tool for CrewAI agents.
# Provides a unified interface for all XML-backed entity operations (PC, NPC, campaign).
# Pydantic schemas for each entity type define the contract between AI agents and the database layer.
"""
XML Update Tool for CrewAI

This module provides a custom tool for CrewAI agents to safely update XML files
by interfacing with the existing XML services.
"""

import logging
from typing import Dict, List, Any, Optional, Union, Type
from enum import Enum
from pydantic import BaseModel, Field, validator

from crewai.tools import BaseTool

logger = logging.getLogger("XMLUpdateTool")

class ServiceType(str, Enum):
    """Enumeration of available XML service types."""
    PC = "pc"
    NPC = "npc"
    CAMPAIGN = "campaign"

class UpdateType(str, Enum):
    """Enumeration of available update operation types."""
    ADD = "add"
    MODIFY = "modify"
    DELETE = "delete"

class NPCUpdate(BaseModel):
    """Schema for NPC updates."""
    name: Optional[str] = Field(None, description="The name of the NPC")
    race: Optional[str] = Field(None, description="Race or species of the NPC")
    occupation: Optional[str] = Field(None, description="Role or occupation of the NPC")
    location: Optional[str] = Field(None, description="Current location of the NPC")
    status: Optional[str] = Field(None, description="Status of the NPC (Active, Deceased, etc.)")
    notes: Optional[str] = Field(None, description="Additional notes about the NPC")
    relationship_to_party: Optional[str] = Field(None, description="Relationship to the player party")

class PCUpdate(BaseModel):
    """Schema for PC updates."""
    name: Optional[str] = Field(None, description="The name of the PC")
    player_discord_name: Optional[str] = Field(None, description="Discord name of the player")
    player_nickname: Optional[str] = Field(None, description="Nickname of the player")
    class_name: Optional[str] = Field(None, description="Class of the character")
    race: Optional[str] = Field(None, description="Race of the character")
    background: Optional[str] = Field(None, description="Background of the character")
    level: Optional[int] = Field(None, description="Current level of the character")
    items: Optional[List[str]] = Field(None, description="Items owned by the character")
    notes: Optional[str] = Field(None, description="Additional notes about the character")
    
    @validator('items', pre=True)
    def validate_items(cls, v):
        """Convert various item formats to a list of strings."""
        if v is None:
            return None
            
        # If it's already a list of strings, return it
        if isinstance(v, list):
            # Make sure all items are strings
            result = []
            for item in v:
                if isinstance(item, str):
                    result.append(item)
                elif isinstance(item, dict):
                    # Handle complex item objects by converting to string
                    if 'name' in item and item['name']:
                        name = item['name']
                        if 'description' in item and item['description']:
                            desc = item['description']
                            result.append(f"{name} ({desc})")
                        else:
                            result.append(name)
                    else:
                        # If no name, use str representation
                        result.append(str(item))
                else:
                    # Convert any other type to string
                    result.append(str(item))
            return result
            
        # If it's a string that looks like a JSON array, parse it
        if isinstance(v, str):
            try:
                # Try to parse as JSON
                import json
                items = json.loads(v)
                if isinstance(items, list):
                    return [str(item) for item in items]
            except:
                # If not valid JSON, treat it as a single item
                return [v]
                
        # Handle any other format by converting to string and returning as single item
        return [str(v)]

class LocationUpdate(BaseModel):
    """Schema for location updates."""
    name: str = Field(description="The name of the location")
    description: Optional[str] = Field(None, description="Description of the location")
    type: Optional[str] = Field(None, description="Type of location (city, dungeon, etc.)")
    region: Optional[str] = Field(None, description="Region where the location is situated")
    notable_npcs: Optional[List[str]] = Field(None, description="NPCs associated with this location")
    points_of_interest: Optional[List[str]] = Field(None, description="Points of interest within this location")
    notes: Optional[str] = Field(None, description="Additional notes about the location")

class FactionUpdate(BaseModel):
    """Schema for faction updates."""
    name: str = Field(description="The name of the faction")
    description: Optional[str] = Field(None, description="Description of the faction")
    leadership: Optional[List[Dict[str, str]]] = Field(None, description="Leaders of the faction, with name and title")
    goals: Optional[List[str]] = Field(None, description="Goals of the faction")
    associated_npcs: Optional[List[Dict[str, str]]] = Field(None, description="NPCs associated with the faction, with name and role")
    notes: Optional[str] = Field(None, description="Additional notes about the faction")

class QuestUpdate(BaseModel):
    """Schema for quest updates."""
    title: Optional[str] = Field(None, description="The title of the quest")
    name: Optional[str] = Field(None, description="The name of the quest (alias for title)")
    description: Optional[str] = Field(None, description="Description of the quest")
    giver: Optional[str] = Field(None, description="NPC or entity who gave the quest")
    status: Optional[str] = Field(None, description="Status of the quest (Active, Complete, Failed)")
    related_npcs: Optional[List[Dict[str, str]]] = Field(None, description="NPCs related to this quest")
    related_locations: Optional[List[str]] = Field(None, description="Locations related to this quest")
    parent_quest_id: Optional[str] = Field(None, description="ID of parent quest if this is a subquest")
    notes: Optional[str] = Field(None, description="Additional notes about the quest")
    
    @validator('title', always=True)
    def validate_title(cls, v, values):
        """Use name as title if title is not provided."""
        if v is None and 'name' in values and values['name'] is not None:
            return values['name']
        return v

class CampaignUpdate(BaseModel):
    """Schema for campaign info updates."""
    setting: Optional[str] = Field(None, description="Campaign setting")
    theme: Optional[str] = Field(None, description="Campaign theme")
    summary: Optional[str] = Field(None, description="Campaign summary")
    system: Optional[str] = Field(None, description="Game system (D&D 5e, Pathfinder, etc.)")
    game_master: Optional[str] = Field(None, description="Name of the Game Master")
    gm_discord_id: Optional[str] = Field(None, description="Discord ID of the Game Master")

class WorldUpdate(BaseModel):
    """Schema for world info updates."""
    name: Optional[str] = Field(None, description="Name of the world")
    description: Optional[str] = Field(None, description="Description of the world")
    setting: Optional[str] = Field(None, description="Setting of the world")
    technology_level: Optional[str] = Field(None, description="Technology level of the world")
    magic_level: Optional[str] = Field(None, description="Magic level in the world")
    deities: Optional[List[str]] = Field(None, description="Deities in the world")
    major_events: Optional[List[str]] = Field(None, description="Major historical events")
    notes: Optional[str] = Field(None, description="Additional notes about the world")

class XMLUpdateInput(BaseModel):
    """Input schema for the XML Update Tool."""
    guild_id: int = Field(..., description="Discord guild ID - identifies which Discord server the data belongs to")
    service_type: ServiceType = Field(..., description="Type of service to use (pc, npc, campaign)")
    update_type: UpdateType = Field(..., description="Type of update to perform (add, modify, delete)")
    entity_id: Optional[str] = Field(None, description="ID of the entity to update (required for modify and delete operations)")
    campaign_id: Optional[str] = Field(None, description="ID of the campaign to work with (if None, will use active campaign)")
    
    # Entity-specific update data - only one of these should be provided
    pc_data: Optional[PCUpdate] = Field(None, description="Data for PC updates")
    npc_data: Optional[NPCUpdate] = Field(None, description="Data for NPC updates")
    location_data: Optional[LocationUpdate] = Field(None, description="Data for location updates")
    faction_data: Optional[FactionUpdate] = Field(None, description="Data for faction updates")
    quest_data: Optional[QuestUpdate] = Field(None, description="Data for quest updates")
    campaign_data: Optional[CampaignUpdate] = Field(None, description="Data for campaign updates")
    world_data: Optional[WorldUpdate] = Field(None, description="Data for world updates")
    
    @validator('entity_id')
    def validate_entity_id_required(cls, v, values):
        """Validate that entity_id is provided for modify and delete operations."""
        # Special exception for campaign data and world data which doesn't need entity_id
        if 'update_type' in values and values['update_type'] == UpdateType.MODIFY:
            if 'service_type' in values and values['service_type'] == ServiceType.CAMPAIGN:
                if values.get('campaign_data') is not None or values.get('world_data') is not None:
                    return v  # Skip validation for campaign/world updates
        
        if 'update_type' in values and values['update_type'] in [UpdateType.MODIFY, UpdateType.DELETE] and not v:
            raise ValueError(f"entity_id is required for {values['update_type']} operations")
        return v
    
    @validator('pc_data', 'npc_data', 'location_data', 'faction_data', 'quest_data', 'campaign_data', 'world_data')
    def validate_data_matches_service(cls, v, values):
        """Validate that the provided data matches the specified service type."""
        if 'service_type' not in values:
            return v
            
        service_type = values['service_type']
        update_type = values.get('update_type')
        field_name = f"{service_type}_data"
        
        # For delete operations, we don't need any data
        if update_type == UpdateType.DELETE:
            return v
            
        # Check if the appropriate data field is provided based on service_type
        if service_type == ServiceType.PC and field_name == 'pc_data' and v is not None:
            return v
        elif service_type == ServiceType.NPC and field_name == 'npc_data' and v is not None:
            return v
        elif service_type == ServiceType.CAMPAIGN:
            if ((field_name == 'campaign_data' and v is not None) or 
                (field_name == 'faction_data' and v is not None) or 
                (field_name == 'quest_data' and v is not None) or
                (field_name == 'world_data' and v is not None) or 
                (field_name == 'location_data' and v is not None)):
                return v
                
        # If we get here and the field matches the service_type, return v
        if field_name == f"{service_type}_data":
            return v
            
        # If we're checking a field that doesn't match the service type, it should be None
        if v is not None and field_name != f"{service_type}_data":
            raise ValueError(f"Cannot provide {field_name} when service_type is {service_type}")
            
        return v

class XMLUpdateTool(BaseTool):
    """Tool for updating XML files safely."""
    
    name: str = "XML Update Tool"
    description: str = """
    Safely update XML data files for D&D campaigns. This tool can add, modify, or delete 
    entries in various XML files storing campaign data.
    
    You must specify:
    - guild_id: The Discord server ID
    - service_type: What kind of data to update (pc, npc, campaign)
    - update_type: The operation to perform (add, modify, delete)
    - entity_id: ID of the entity to update (for modify/delete operations)
             NOTE: entity_id is NOT required for campaign_data and world_data updates
    - campaign_id: Optional. If not provided, the active campaign will be used
    
    You must also provide the appropriate data field for the service type:
    - For PC updates: pc_data
    - For NPC updates: npc_data
    - For Campaign updates: 
        - campaign_data (campaign info)
        - faction_data (factions)
        - quest_data (quests)
        - world_data (world info)
        - location_data (locations)
    
    Examples:
    1. To update an NPC's description:
       guild_id, service_type="npc", update_type="modify", entity_id="npc-123", 
       npc_data={"name": "Gandalf", "description": "A wise wizard with a long grey beard."}
       
    2. To create a new location:
       guild_id, service_type="campaign", update_type="add",
       location_data={"name": "Rivendell", "type": "City", "description": "Elven sanctuary"}
       
    3. To update campaign information (no entity_id needed):
       guild_id, service_type="campaign", update_type="modify",
       campaign_data={"setting": "Forgotten Realms", "theme": "High Fantasy"}
       
    4. To add items to a PC:
       guild_id, service_type="pc", update_type="modify", entity_id="pc-123",
       pc_data={"items": ["Magic Sword", "Healing Potion"]}
    """
    
    args_schema: Type[BaseModel] = XMLUpdateInput
    
    def _run(self, 
             guild_id: int, 
             service_type: ServiceType, 
             update_type: UpdateType, 
             entity_id: Optional[str] = None,
             campaign_id: str = None,
             pc_data: Optional[Dict[str, Any]] = None,
             npc_data: Optional[Dict[str, Any]] = None,
             location_data: Optional[Dict[str, Any]] = None,
             faction_data: Optional[Dict[str, Any]] = None,
             quest_data: Optional[Dict[str, Any]] = None,
             campaign_data: Optional[Dict[str, Any]] = None, 
             world_data: Optional[Dict[str, Any]] = None) -> str:
        """
        Execute the XML update with the specified parameters.
        
        Returns:
            A string with the result of the update operation.
        """
        # Production: routes to appropriate XML service (PCService, NPCService, CampaignService)
        # based on service_type and executes the CRUD operation against campaign XML files
        raise NotImplementedError("Showcase only")
