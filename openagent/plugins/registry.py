"""
Plugin Registry for OpenAgent

Manages plugin metadata, discovery, and cataloging with persistent storage
and efficient querying capabilities.
"""

import asyncio
import json
import logging
from pathlib import Path
from typing import Dict, List, Optional, Set, Any
from datetime import datetime, timedelta, timezone
import sqlite3
import aiosqlite

from .base import PluginMetadata, PluginType, PluginStatus

logger = logging.getLogger(__name__)


class PluginRegistry:
    """Registry for managing plugin metadata and discovery."""
    
    def __init__(self, storage_path: Optional[Path] = None):
        """Initialize the plugin registry."""
        self.storage_path = storage_path or Path("plugins/registry.db")
        self._metadata_cache: Dict[str, PluginMetadata] = {}
        self._is_initialized = False
        
        logger.info(f"PluginRegistry initialized with storage: {self.storage_path}")
    
    async def initialize(self) -> None:
        """Initialize the registry and create necessary storage."""
        if self._is_initialized:
            return
        
        try:
            # Ensure storage directory exists
            self.storage_path.parent.mkdir(parents=True, exist_ok=True)
            
            # Initialize database
            await self._init_database()
            
            # Load cache
            await self._load_cache()
            
            self._is_initialized = True
            logger.info("PluginRegistry initialized successfully")
            
        except Exception as e:
            logger.error(f"Failed to initialize PluginRegistry: {e}")
            raise
    
    async def cleanup(self) -> None:
        """Cleanup registry resources."""
        self._metadata_cache.clear()
        self._is_initialized = False
        logger.info("PluginRegistry cleaned up")
    
    async def register_plugin(self, metadata: PluginMetadata) -> bool:
        """Register a plugin with the registry."""
        if not self._is_initialized:
            await self.initialize()
        
        try:
            # Update timestamp
metadata.updated_at = datetime.now(timezone.utc)
            
            # Store in database
            async with aiosqlite.connect(self.storage_path) as db:
                await db.execute("""
                    INSERT OR REPLACE INTO plugins 
                    (name, version, description, author, plugin_type, metadata_json, created_at, updated_at)
                    VALUES (?, ?, ?, ?, ?, ?, ?, ?)
                """, (
                    metadata.name,
                    metadata.version,
                    metadata.description,
                    metadata.author,
                    metadata.plugin_type.value,
                    json.dumps(metadata.to_dict()),
                    metadata.created_at.isoformat(),
                    metadata.updated_at.isoformat()
                ))
                await db.commit()
            
            # Update cache
            self._metadata_cache[metadata.name] = metadata
            
            logger.info(f"Registered plugin: {metadata.name} v{metadata.version}")
            return True
            
        except Exception as e:
            logger.error(f"Failed to register plugin {metadata.name}: {e}")
            return False
    
    async def unregister_plugin(self, plugin_name: str) -> bool:
        """Unregister a plugin from the registry."""
        if not self._is_initialized:
            await self.initialize()
        
        try:
            # Remove from database
            async with aiosqlite.connect(self.storage_path) as db:
                cursor = await db.execute("DELETE FROM plugins WHERE name = ?", (plugin_name,))
                await db.commit()
                
                if cursor.rowcount == 0:
                    logger.warning(f"Plugin {plugin_name} was not found in registry")
                    return False
            
            # Remove from cache
            if plugin_name in self._metadata_cache:
                del self._metadata_cache[plugin_name]
            
            logger.info(f"Unregistered plugin: {plugin_name}")
            return True
            
        except Exception as e:
            logger.error(f"Failed to unregister plugin {plugin_name}: {e}")
            return False
    
    async def get_plugin(self, plugin_name: str) -> Optional[PluginMetadata]:
        """Get plugin metadata by name."""
        if not self._is_initialized:
            await self.initialize()
        
        # Check cache first
        if plugin_name in self._metadata_cache:
            return self._metadata_cache[plugin_name]
        
        # Query database
        try:
            async with aiosqlite.connect(self.storage_path) as db:
                cursor = await db.execute(
                    "SELECT metadata_json FROM plugins WHERE name = ?",
                    (plugin_name,)
                )
                row = await cursor.fetchone()
                
                if row:
                    metadata_dict = json.loads(row[0])
                    metadata = PluginMetadata.from_dict(metadata_dict)
                    
                    # Cache the result
                    self._metadata_cache[plugin_name] = metadata
                    return metadata
                
                return None
                
        except Exception as e:
            logger.error(f"Failed to get plugin {plugin_name}: {e}")
            return None
    
    async def list_plugins(
        self,
        plugin_type: Optional[PluginType] = None,
        author: Optional[str] = None,
        keywords: Optional[List[str]] = None
    ) -> List[str]:
        """List plugin names with optional filtering."""
        if not self._is_initialized:
            await self.initialize()
        
        try:
            query = "SELECT name FROM plugins WHERE 1=1"
            params = []
            
            if plugin_type:
                query += " AND plugin_type = ?"
                params.append(plugin_type.value)
            
            if author:
                query += " AND author = ?"
                params.append(author)
            
            async with aiosqlite.connect(self.storage_path) as db:
                cursor = await db.execute(query, params)
                rows = await cursor.fetchall()
                
                plugin_names = [row[0] for row in rows]
                
                # Additional filtering for keywords (in-memory)
                if keywords:
                    filtered_names = []
                    for name in plugin_names:
                        metadata = await self.get_plugin(name)
                        if metadata and any(keyword in metadata.keywords for keyword in keywords):
                            filtered_names.append(name)
                    plugin_names = filtered_names
                
                return plugin_names
                
        except Exception as e:
            logger.error(f"Failed to list plugins: {e}")
            return []
    
    async def search_plugins(self, query: str) -> List[PluginMetadata]:
        """Search plugins by name, description, or keywords."""
        if not self._is_initialized:
            await self.initialize()
        
        try:
            search_query = f"%{query.lower()}%"
            
            async with aiosqlite.connect(self.storage_path) as db:
                cursor = await db.execute("""
                    SELECT metadata_json FROM plugins 
                    WHERE LOWER(name) LIKE ? 
                    OR LOWER(description) LIKE ?
                    OR LOWER(author) LIKE ?
                    ORDER BY name
                """, (search_query, search_query, search_query))
                
                rows = await cursor.fetchall()
                results = []
                
                for row in rows:
                    try:
                        metadata_dict = json.loads(row[0])
                        metadata = PluginMetadata.from_dict(metadata_dict)
                        
                        # Also check keywords
                        if any(query.lower() in keyword.lower() for keyword in metadata.keywords):
                            results.append(metadata)
                        elif query.lower() in metadata.name.lower() or query.lower() in metadata.description.lower():
                            results.append(metadata)
                            
                    except Exception as e:
                        logger.error(f"Error parsing plugin metadata in search: {e}")
                        continue
                
                return results
                
        except Exception as e:
            logger.error(f"Failed to search plugins: {e}")
            return []
    
    async def get_plugins_by_type(self, plugin_type: PluginType) -> List[PluginMetadata]:
        """Get all plugins of a specific type."""
        if not self._is_initialized:
            await self.initialize()
        
        try:
            async with aiosqlite.connect(self.storage_path) as db:
                cursor = await db.execute(
                    "SELECT metadata_json FROM plugins WHERE plugin_type = ? ORDER BY name",
                    (plugin_type.value,)
                )
                rows = await cursor.fetchall()
                
                results = []
                for row in rows:
                    try:
                        metadata_dict = json.loads(row[0])
                        metadata = PluginMetadata.from_dict(metadata_dict)
                        results.append(metadata)
                    except Exception as e:
                        logger.error(f"Error parsing plugin metadata: {e}")
                        continue
                
                return results
                
        except Exception as e:
            logger.error(f"Failed to get plugins by type {plugin_type}: {e}")
            return []
    
    async def get_plugin_dependencies(self, plugin_name: str) -> List[str]:
        """Get dependencies for a plugin."""
        metadata = await self.get_plugin(plugin_name)
        return metadata.dependencies if metadata else []
    
    async def find_dependents(self, plugin_name: str) -> List[str]:
        """Find plugins that depend on the given plugin."""
        if not self._is_initialized:
            await self.initialize()
        
        dependents = []
        
        try:
            async with aiosqlite.connect(self.storage_path) as db:
                cursor = await db.execute("SELECT name, metadata_json FROM plugins")
                rows = await cursor.fetchall()
                
                for row in rows:
                    try:
                        metadata_dict = json.loads(row[1])
                        metadata = PluginMetadata.from_dict(metadata_dict)
                        
                        if plugin_name in metadata.dependencies:
                            dependents.append(metadata.name)
                            
                    except Exception as e:
                        logger.error(f"Error checking dependencies for {row[0]}: {e}")
                        continue
                
                return dependents
                
        except Exception as e:
            logger.error(f"Failed to find dependents for {plugin_name}: {e}")
            return []
    
    async def validate_dependencies(self, plugin_name: str) -> Dict[str, bool]:
        """Validate that all dependencies for a plugin are available."""
        dependencies = await self.get_plugin_dependencies(plugin_name)
        validation_result = {}
        
        for dep in dependencies:
            metadata = await self.get_plugin(dep)
            validation_result[dep] = metadata is not None
        
        return validation_result
    
    async def get_statistics(self) -> Dict[str, Any]:
        """Get registry statistics."""
        if not self._is_initialized:
            await self.initialize()
        
        try:
            async with aiosqlite.connect(self.storage_path) as db:
                # Total plugins
                cursor = await db.execute("SELECT COUNT(*) FROM plugins")
                total = (await cursor.fetchone())[0]
                
                # Plugins by type
                cursor = await db.execute("""
                    SELECT plugin_type, COUNT(*) 
                    FROM plugins 
                    GROUP BY plugin_type
                """)
                by_type = {row[0]: row[1] for row in await cursor.fetchall()}
                
                # Recent registrations (last 7 days)
seven_days_ago = (datetime.now(timezone.utc) - timedelta(days=7)).isoformat()
                cursor = await db.execute(
                    "SELECT COUNT(*) FROM plugins WHERE created_at >= ?",
                    (seven_days_ago,)
                )
                recent = (await cursor.fetchone())[0]
                
                return {
                    "total_plugins": total,
                    "plugins_by_type": by_type,
                    "recent_registrations": recent,
                    "cache_size": len(self._metadata_cache)
                }
                
        except Exception as e:
            logger.error(f"Failed to get statistics: {e}")
            return {}
    
    async def export_registry(self, export_path: Path) -> bool:
        """Export registry to JSON file."""
        if not self._is_initialized:
            await self.initialize()
        
        try:
            plugins = []
            
            async with aiosqlite.connect(self.storage_path) as db:
                cursor = await db.execute("SELECT metadata_json FROM plugins ORDER BY name")
                rows = await cursor.fetchall()
                
                for row in rows:
                    try:
                        metadata_dict = json.loads(row[0])
                        plugins.append(metadata_dict)
                    except Exception as e:
                        logger.error(f"Error parsing plugin metadata during export: {e}")
                        continue
            
            with open(export_path, 'w') as f:
                json.dump({
"exported_at": datetime.now(timezone.utc).isoformat(),
                    "total_plugins": len(plugins),
                    "plugins": plugins
                }, f, indent=2)
            
            logger.info(f"Exported {len(plugins)} plugins to {export_path}")
            return True
            
        except Exception as e:
            logger.error(f"Failed to export registry: {e}")
            return False
    
    async def import_registry(self, import_path: Path) -> int:
        """Import plugins from JSON file."""
        if not self._is_initialized:
            await self.initialize()
        
        try:
            with open(import_path, 'r') as f:
                data = json.load(f)
            
            imported_count = 0
            plugins = data.get("plugins", [])
            
            for plugin_dict in plugins:
                try:
                    metadata = PluginMetadata.from_dict(plugin_dict)
                    if await self.register_plugin(metadata):
                        imported_count += 1
                except Exception as e:
                    logger.error(f"Error importing plugin {plugin_dict.get('name', 'unknown')}: {e}")
                    continue
            
            logger.info(f"Imported {imported_count} plugins from {import_path}")
            return imported_count
            
        except Exception as e:
            logger.error(f"Failed to import registry: {e}")
            return 0
    
    # Private methods
    async def _init_database(self) -> None:
        """Initialize the SQLite database."""
        async with aiosqlite.connect(self.storage_path) as db:
            await db.execute("""
                CREATE TABLE IF NOT EXISTS plugins (
                    name TEXT PRIMARY KEY,
                    version TEXT NOT NULL,
                    description TEXT,
                    author TEXT,
                    plugin_type TEXT NOT NULL,
                    metadata_json TEXT NOT NULL,
                    created_at TEXT NOT NULL,
                    updated_at TEXT NOT NULL
                )
            """)
            
            # Create indexes for better performance
            await db.execute("CREATE INDEX IF NOT EXISTS idx_plugin_type ON plugins(plugin_type)")
            await db.execute("CREATE INDEX IF NOT EXISTS idx_author ON plugins(author)")
            await db.execute("CREATE INDEX IF NOT EXISTS idx_created_at ON plugins(created_at)")
            
            await db.commit()
    
    async def _load_cache(self) -> None:
        """Load frequently accessed plugins into cache."""
        try:
            async with aiosqlite.connect(self.storage_path) as db:
                cursor = await db.execute("SELECT name, metadata_json FROM plugins")
                rows = await cursor.fetchall()
                
                for row in rows:
                    try:
                        metadata_dict = json.loads(row[1])
                        metadata = PluginMetadata.from_dict(metadata_dict)
                        self._metadata_cache[metadata.name] = metadata
                    except Exception as e:
                        logger.error(f"Error loading plugin {row[0]} to cache: {e}")
                        continue
                
                logger.info(f"Loaded {len(self._metadata_cache)} plugins to cache")
                
        except Exception as e:
            logger.error(f"Failed to load cache: {e}")


# Import required for statistics method
from datetime import timedelta
