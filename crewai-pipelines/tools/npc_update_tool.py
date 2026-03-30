# CrewAI tool for NPC CRUD operations.
# AI agents use this tool to track NPCs encountered during sessions, including
# status changes, location updates, and relationship connections.
# The Pydantic schemas define the contract between AI agents and the database layer.
"""
NPC Update Tool for CrewAI

This module provides a custom tool for CrewAI agents to manage Non-Player Characters
in the XML database by interfacing with the NPCService.
"""

import logging
from typing import Dict, List, Any, Optional, Union, Type, Literal
from enum import Enum

from pydantic import BaseModel, Field, validator
from crewai.tools import BaseTool

logger = logging.getLogger("NPCUpdateTool")

class NPCUpdateInput(BaseModel):
    """Input schema for NPCUpdateTool."""
    guild_id: int = Field(..., description="Discord guild ID - identifies which Discord server the data belongs to")
    operation: Literal["create", "update", "delete", "add_connection", "remove_connection", "update_status", "update_location"] = Field(..., description="Operation to perform on the NPC")
    npc_id: Optional[str] = Field(None, description="NPC ID (required for all operations except 'create')")
    npc_data: Optional[Dict[str, Any]] = Field(None, description="NPC data for create/update operations")
    campaign_id: Optional[str] = Field(None, description="Campaign ID (if None, will use active campaign)")
    connection_text: Optional[str] = Field(None, description="Connection text (for 'add_connection'/'remove_connection' operations)")
    status: Optional[str] = Field(None, description="New status (for 'update_status' operation)")
    location: Optional[str] = Field(None, description="New location (for 'update_location' operation)")
    
    @validator('npc_id')
    def validate_npc_id(cls, v, values):
        """Validate that npc_id is provided for non-create operations."""
        if 'operation' in values and values['operation'] != "create" and not v:
            raise ValueError(f"npc_id is required for {values['operation']} operations")
        return v
    
    @validator('npc_data')
    def validate_npc_data(cls, v, values):
        """Validate that npc_data is provided for create and update operations."""
        if 'operation' in values and values['operation'] in ["create", "update"] and not v:
            raise ValueError(f"npc_data is required for {values['operation']} operations")
        return v
    
    @validator('connection_text')
    def validate_connection_text(cls, v, values):
        """Validate that connection_text is provided for connection operations."""
        if 'operation' in values and values['operation'] in ["add_connection", "remove_connection"] and not v:
            raise ValueError(f"connection_text is required for {values['operation']} operations")
        return v
    
    @validator('status')
    def validate_status(cls, v, values):
        """Validate that status is provided for update_status operation."""
        if 'operation' in values and values['operation'] == "update_status" and not v:
            raise ValueError("status is required for update_status operation")
        return v
    
    @validator('location')
    def validate_location(cls, v, values):
        """Validate that location is provided for update_location operation."""
        if 'operation' in values and values['operation'] == "update_location" and not v:
            raise ValueError("location is required for update_location operation")
        return v

class NPCUpdateTool(BaseTool):
    """Tool for managing Non-Player Characters in the XML database."""
    
    name: str = "NPC Update Tool"
    description: str = """
    Create, update, or delete NPCs in the D&D campaign database.
    This tool helps you manage NPC information including basic details, status, location, and connections.
    
    You must specify:
    - guild_id: The Discord server ID
    - operation: What operation to perform (create, update, delete, add_connection, remove_connection, update_status, update_location)
    - npc_id: ID of the NPC (required for all operations except 'create')
    - campaign_id: Optional. If not provided, the active campaign will be used
    
    For 'create' operation, you must provide:
    - npc_data: Dictionary with 'name', 'race', 'occupation', 'location', 'notes', 'status', 'relationship_to_party' fields
    
    For 'update' operation, you must provide:
    - npc_data: Dictionary with fields to update
    
    For 'add_connection' and 'remove_connection' operations, you must provide:
    - connection_text: Description of the connection to add or remove
    
    For 'update_status' operation, you must provide:
    - status: New status of the NPC (e.g., "Active", "Deceased", "Missing")
    
    For 'update_location' operation, you must provide:
    - location: New location of the NPC
    
    Examples:
    1. To create a new NPC:
       guild_id=123456, operation="create", npc_data={"name": "Elminster", "race": "Human", "occupation": "Archmage", "location": "Shadowdale", "notes": "Powerful wizard", "status": "Active", "relationship_to_party": "Ally"}
       
    2. To update an NPC's information:
       guild_id=123456, operation="update", npc_id="npc-elminster-123", npc_data={"notes": "Currently away on business"}
       
    3. To update an NPC's status:
       guild_id=123456, operation="update_status", npc_id="npc-elminster-123", status="Missing"
    """
    
    args_schema: Type[BaseModel] = NPCUpdateInput
    
    def _run(self, 
             guild_id: int,
             operation: str,
             npc_id: Optional[str] = None,
             npc_data: Optional[Dict[str, Any]] = None,
             campaign_id: Optional[str] = None,
             connection_text: Optional[str] = None,
             status: Optional[str] = None,
             location: Optional[str] = None) -> str:
        """
        Execute the NPC update with the specified parameters.
        
        Returns:
            A string with the result of the update operation.
        """
        # Production: wraps XML-based database service for NPC CRUD operations
        raise NotImplementedError("Showcase only")
