"""
Plugin Loader for OpenAgent

Handles dynamic loading of plugins with security validation,
dependency resolution, and error handling.
"""

import asyncio
import importlib
import importlib.util
import inspect
import logging
import sys
from pathlib import Path
from typing import Any, Dict, List, Optional, Type, Set
import json
import ast

from .base import PluginBase, PluginMetadata, PluginType

logger = logging.getLogger(__name__)


class PluginLoader:
    """Dynamic plugin loader with security and validation."""
    
    def __init__(self, plugin_dir: Path):
        """Initialize the plugin loader."""
        self.plugin_dir = plugin_dir
        self._loaded_modules: Dict[str, Any] = {}
        
        logger.info(f"PluginLoader initialized for directory: {plugin_dir}")
    
    async def discover_plugin_files(self) -> List[Path]:
        """Discover all plugin files in the plugin directory."""
        plugin_files = []
        
        try:
            if not self.plugin_dir.exists():
                logger.warning(f"Plugin directory does not exist: {self.plugin_dir}")
                return plugin_files
            
            # Look for Python files and directories with __init__.py
            for item in self.plugin_dir.iterdir():
                if item.is_file() and item.suffix == '.py' and not item.name.startswith('_'):
                    plugin_files.append(item)
                elif item.is_dir() and (item / '__init__.py').exists():
                    plugin_files.append(item / '__init__.py')
            
            logger.info(f"Discovered {len(plugin_files)} plugin files")
            return plugin_files
            
        except Exception as e:
            logger.error(f"Error discovering plugin files: {e}")
            return []
    
    async def load_plugin_metadata(self, plugin_file: Path) -> Optional[PluginMetadata]:
        """Load plugin metadata from a plugin file."""
        try:
            # First, try to load from metadata file
            metadata_file = plugin_file.parent / f"{plugin_file.stem}_metadata.json"
            if metadata_file.exists():
                with open(metadata_file, 'r') as f:
                    metadata_dict = json.load(f)
                    return PluginMetadata.from_dict(metadata_dict)
            
            # Otherwise, extract from plugin source
            metadata = await self._extract_metadata_from_source(plugin_file)
            if metadata:
                return metadata
            
            logger.warning(f"No metadata found for plugin: {plugin_file}")
            return None
            
        except Exception as e:
            logger.error(f"Error loading metadata for {plugin_file}: {e}")
            return None
    
    async def load_plugin_class(self, metadata: PluginMetadata) -> Optional[Type[PluginBase]]:
        """Load a plugin class from its metadata."""
        try:
            # Construct module path
            module_path = self.plugin_dir / f"{metadata.name}.py"
            if not module_path.exists():
                # Try directory-based plugin
                module_path = self.plugin_dir / metadata.name / "__init__.py"
                if not module_path.exists():
                    logger.error(f"Plugin file not found for {metadata.name}")
                    return None
            
            # Load the module
            plugin_module = await self._load_module(metadata.name, module_path)
            if not plugin_module:
                return None
            
            # Find the plugin class
            plugin_class = await self._find_plugin_class(plugin_module, metadata)
            if not plugin_class:
                logger.error(f"Plugin class not found in {metadata.name}")
                return None
            
            logger.info(f"Loaded plugin class: {plugin_class.__name__}")
            return plugin_class
            
        except Exception as e:
            logger.error(f"Error loading plugin class for {metadata.name}: {e}")
            return None
    
    async def unload_plugin_module(self, plugin_name: str) -> bool:
        """Unload a plugin module from memory."""
        try:
            if plugin_name in self._loaded_modules:
                # Remove from sys.modules if present
                module_name = f"openagent_plugin_{plugin_name}"
                if module_name in sys.modules:
                    del sys.modules[module_name]
                
                # Remove from our cache
                del self._loaded_modules[plugin_name]
                
                logger.info(f"Unloaded plugin module: {plugin_name}")
                return True
            
            return True  # Already unloaded
            
        except Exception as e:
            logger.error(f"Error unloading plugin module {plugin_name}: {e}")
            return False
    
    async def validate_plugin_file(self, plugin_file: Path) -> bool:
        """Validate a plugin file for basic security and structure."""
        try:
            with open(plugin_file, 'r') as f:
                source_code = f.read()
            
            # Parse the AST to check for dangerous imports/operations
            tree = ast.parse(source_code)
            
            # Check for dangerous imports
            dangerous_imports = {
                'os', 'sys', 'subprocess', 'exec', 'eval',
                'importlib', '__import__', 'compile'
            }
            
            for node in ast.walk(tree):
                if isinstance(node, ast.Import):
                    for alias in node.names:
                        if alias.name in dangerous_imports:
                            logger.warning(f"Potentially dangerous import in {plugin_file}: {alias.name}")
                            # Don't reject, just warn for now
                
                elif isinstance(node, ast.ImportFrom):
                    if node.module in dangerous_imports:
                        logger.warning(f"Potentially dangerous import in {plugin_file}: {node.module}")
            
            return True
            
        except SyntaxError as e:
            logger.error(f"Syntax error in plugin file {plugin_file}: {e}")
            return False
        except Exception as e:
            logger.error(f"Error validating plugin file {plugin_file}: {e}")
            return False
    
    # Private methods
    async def _extract_metadata_from_source(self, plugin_file: Path) -> Optional[PluginMetadata]:
        """Extract metadata from plugin source code."""
        try:
            with open(plugin_file, 'r') as f:
                source_code = f.read()
            
            # Parse the AST
            tree = ast.parse(source_code)
            
            # Look for plugin metadata in class decorators or module variables
            metadata = None
            
            for node in ast.walk(tree):
                # Check for @plugin_metadata decorator
                if isinstance(node, ast.ClassDef):
                    for decorator in node.decorator_list:
                        if isinstance(decorator, ast.Call) and \
                           isinstance(decorator.func, ast.Name) and \
                           decorator.func.id == 'plugin_metadata':
                            
                            metadata = self._parse_metadata_decorator(decorator)
                            if metadata:
                                break
                
                # Check for module-level PLUGIN_METADATA variable
                elif isinstance(node, ast.Assign):
                    for target in node.targets:
                        if isinstance(target, ast.Name) and target.id == 'PLUGIN_METADATA':
                            if isinstance(node.value, ast.Dict):
                                metadata = self._parse_metadata_dict(node.value)
            
            return metadata
            
        except Exception as e:
            logger.error(f"Error extracting metadata from {plugin_file}: {e}")
            return None
    
    def _parse_metadata_decorator(self, decorator_node: ast.Call) -> Optional[PluginMetadata]:
        """Parse metadata from a decorator node."""
        try:
            metadata_dict = {}
            
            # Parse keyword arguments
            for keyword in decorator_node.keywords:
                if isinstance(keyword.value, ast.Constant):
                    metadata_dict[keyword.arg] = keyword.value.value
                elif isinstance(keyword.value, ast.Str):  # Python < 3.8 compatibility
                    metadata_dict[keyword.arg] = keyword.value.s
            
            # Parse positional arguments (name, version, description, author, plugin_type)
            if len(decorator_node.args) >= 5:
                arg_names = ['name', 'version', 'description', 'author', 'plugin_type']
                for i, arg in enumerate(decorator_node.args[:5]):
                    if isinstance(arg, ast.Constant):
                        if arg_names[i] == 'plugin_type':
                            # Handle PluginType enum
                            metadata_dict[arg_names[i]] = PluginType(arg.value)
                        else:
                            metadata_dict[arg_names[i]] = arg.value
            
            if 'name' in metadata_dict and 'version' in metadata_dict:
                return PluginMetadata(
                    name=metadata_dict['name'],
                    version=metadata_dict['version'],
                    description=metadata_dict.get('description', ''),
                    author=metadata_dict.get('author', ''),
                    plugin_type=metadata_dict.get('plugin_type', PluginType.CUSTOM)
                )
            
            return None
            
        except Exception as e:
            logger.error(f"Error parsing metadata decorator: {e}")
            return None
    
    def _parse_metadata_dict(self, dict_node: ast.Dict) -> Optional[PluginMetadata]:
        """Parse metadata from a dictionary node."""
        try:
            metadata_dict = {}
            
            for key, value in zip(dict_node.keys, dict_node.values):
                if isinstance(key, ast.Constant) and isinstance(value, ast.Constant):
                    metadata_dict[key.value] = value.value
                elif isinstance(key, ast.Str) and isinstance(value, ast.Str):
                    metadata_dict[key.s] = value.s
            
            if 'name' in metadata_dict and 'version' in metadata_dict:
                return PluginMetadata(
                    name=metadata_dict['name'],
                    version=metadata_dict['version'],
                    description=metadata_dict.get('description', ''),
                    author=metadata_dict.get('author', ''),
                    plugin_type=PluginType(metadata_dict.get('plugin_type', 'custom'))
                )
            
            return None
            
        except Exception as e:
            logger.error(f"Error parsing metadata dictionary: {e}")
            return None
    
    async def _load_module(self, plugin_name: str, module_path: Path) -> Optional[Any]:
        """Load a Python module from file path."""
        try:
            # Check if already loaded
            if plugin_name in self._loaded_modules:
                return self._loaded_modules[plugin_name]
            
            # Validate file first
            if not await self.validate_plugin_file(module_path):
                return None
            
            # Create module spec
            module_name = f"openagent_plugin_{plugin_name}"
            spec = importlib.util.spec_from_file_location(module_name, module_path)
            
            if not spec or not spec.loader:
                logger.error(f"Could not create module spec for {plugin_name}")
                return None
            
            # Create and execute module
            module = importlib.util.module_from_spec(spec)
            sys.modules[module_name] = module
            spec.loader.exec_module(module)
            
            # Cache the module
            self._loaded_modules[plugin_name] = module
            
            logger.info(f"Loaded plugin module: {plugin_name}")
            return module
            
        except Exception as e:
            logger.error(f"Error loading module for {plugin_name}: {e}")
            return None
    
    async def _find_plugin_class(self, module: Any, metadata: PluginMetadata) -> Optional[Type[PluginBase]]:
        """Find the main plugin class in a module."""
        try:
            # Look for classes that inherit from PluginBase
            plugin_classes = []
            
            for name in dir(module):
                obj = getattr(module, name)
                if (inspect.isclass(obj) and 
                    issubclass(obj, PluginBase) and 
                    obj != PluginBase):
                    plugin_classes.append(obj)
            
            if not plugin_classes:
                logger.error(f"No plugin classes found in {metadata.name}")
                return None
            
            # If multiple classes, try to find one with matching name
            if len(plugin_classes) > 1:
                for cls in plugin_classes:
                    if cls.__name__.lower() == metadata.name.lower() or \
                       cls.__name__.lower().endswith('plugin'):
                        return cls
                
                # Fallback to first class
                logger.warning(f"Multiple plugin classes found in {metadata.name}, using {plugin_classes[0].__name__}")
            
            return plugin_classes[0]
            
        except Exception as e:
            logger.error(f"Error finding plugin class in {metadata.name}: {e}")
            return None
    
    def get_loaded_modules(self) -> Dict[str, Any]:
        """Get all loaded plugin modules."""
        return self._loaded_modules.copy()
    
    async def reload_module(self, plugin_name: str, module_path: Path) -> Optional[Any]:
        """Reload a plugin module."""
        try:
            # Unload existing module
            await self.unload_plugin_module(plugin_name)
            
            # Load again
            return await self._load_module(plugin_name, module_path)
            
        except Exception as e:
            logger.error(f"Error reloading module {plugin_name}: {e}")
            return None
