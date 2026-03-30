# CrewAI tool for Player Character CRUD operations.
# AI agents use this tool to create, update, and delete PCs discovered during session processing.
# The Pydantic schemas define the contract between AI agents and the database layer.
"""
PC Update Tool for CrewAI

This module provides a custom tool for CrewAI agents to manage Player Characters
in the XML database by interfacing with the PCService.
"""

import logging
from typing import Dict, List, Any, Optional, Union, Type, Literal
from enum import Enum

from pydantic import BaseModel, Field, validator
from crewai.tools import BaseTool

logger = logging.getLogger("PCUpdateTool")

class PCUpdateInput(BaseModel):
    """Input schema for PCUpdateTool."""
    guild_id: int = Field(..., description="Discord guild ID - identifies which Discord server the data belongs to")
    operation: Literal["create", "update", "delete", "add_item", "remove_item", "add_relationship", "remove_relationship"] = Field(..., description="Operation to perform on the PC")
    character_id: Optional[str] = Field(None, description="Character ID (required for all operations except 'create')")
    character_data: Optional[Dict[str, Any]] = Field(None, description="Character data for create/update operations")
    campaign_id: Optional[str] = Field(None, description="Campaign ID (if None, will use active campaign)")
    player_id: Optional[int] = Field(0, description="Player's Discord ID (for 'create' operation)")
    player_name: Optional[str] = Field("", description="Player's Discord username (for 'create' operation)")
    player_nickname: Optional[str] = Field("", description="Player's Discord nickname (for 'create' operation)")
    item_name: Optional[str] = Field(None, description="Item name (for 'add_item'/'remove_item' operations)")
    relationship_text: Optional[str] = Field(None, description="Relationship text (for 'add_relationship'/'remove_relationship' operations)")
    
    @validator('character_id')
    def validate_character_id(cls, v, values):
        """Validate that character_id is provided for non-create operations."""
        if 'operation' in values and values['operation'] != "create" and not v:
            raise ValueError(f"character_id is required for {values['operation']} operations")
        return v
    
    @validator('character_data')
    def validate_character_data(cls, v, values):
        """Validate that character_data is provided for create and update operations."""
        if 'operation' in values and values['operation'] in ["create", "update"] and not v:
            raise ValueError(f"character_data is required for {values['operation']} operations")
        return v
    
    @validator('item_name')
    def validate_item_name(cls, v, values):
        """Validate that item_name is provided for item operations."""
        if 'operation' in values and values['operation'] in ["add_item", "remove_item"] and not v:
            raise ValueError(f"item_name is required for {values['operation']} operations")
        return v
    
    @validator('relationship_text')
    def validate_relationship_text(cls, v, values):
        """Validate that relationship_text is provided for relationship operations."""
        if 'operation' in values and values['operation'] in ["add_relationship", "remove_relationship"] and not v:
            raise ValueError(f"relationship_text is required for {values['operation']} operations")
        return v

class PCUpdateTool(BaseTool):
    """Tool for managing Player Characters in the XML database."""
    
    name: str = "Player Character Update Tool"
    description: str = """
    Create, update, or delete player characters in the D&D campaign database.
    This tool helps you manage character information including basic details, items, and relationships.
    
    You must specify:
    - guild_id: The Discord server ID
    - operation: What operation to perform (create, update, delete, add_item, remove_item, add_relationship, remove_relationship)
    - character_id: ID of the character (required for all operations except 'create')
    - campaign_id: Optional. If not provided, the active campaign will be used
    
    For 'create' operation, you must provide:
    - character_data: Dictionary with 'name', 'race', 'class', 'level', 'backstory' fields
    - player_name: Player's Discord username
    - player_nickname: Player's Discord nickname
    - player_id: Player's Discord ID (defaults to 0 if not provided)
    
    For 'update' operation, you must provide:
    - character_data: Dictionary with fields to update
    
    For 'add_item' and 'remove_item' operations, you must provide:
    - item_name: Name of the item to add or remove
    
    For 'add_relationship' and 'remove_relationship' operations, you must provide:
    - relationship_text: Description of the relationship to add or remove
    
    Examples:
    1. To create a new character:
       guild_id=123456, operation="create", character_data={"name": "Gandalf", "race": "Human", "class": "Wizard", "level": 10, "backstory": "A wise old wizard"}, player_name="JohnDoe", player_nickname="Johnny"
       
    2. To update a character's level:
       guild_id=123456, operation="update", character_id="pc-gandalf-123", character_data={"level": 11}
       
    3. To add an item to a character:
       guild_id=123456, operation="add_item", character_id="pc-gandalf-123", item_name="Staff of Power"
    """
    
    args_schema: Type[BaseModel] = PCUpdateInput
    
    def _run(self, 
             guild_id: int,
             operation: str,
             character_id: Optional[str] = None,
             character_data: Optional[Dict[str, Any]] = None,
             campaign_id: Optional[str] = None,
             player_id: Optional[int] = 0,
             player_name: Optional[str] = "",
             player_nickname: Optional[str] = "",
             item_name: Optional[str] = None,
             relationship_text: Optional[str] = None) -> str:
        """
        Execute the PC update with the specified parameters.
        
        Returns:
            A string with the result of the update operation.
        """
        # Production: wraps XML-based database service for PC CRUD operations
        raise NotImplementedError("Showcase only")
