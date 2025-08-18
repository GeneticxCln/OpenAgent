Plugin Development Guide
========================

OpenAgent features a powerful plugin system that allows you to extend its functionality with custom tools and integrations.

Plugin Architecture
-------------------

OpenAgent's plugin system is built around the concept of **Tools**. Each plugin provides one or more tools that can be used by the AI agent to perform specific tasks.

Creating Your First Plugin
---------------------------

1. **Define Your Tool Class**

.. code-block:: python

   from openagent.core.base import BaseTool, ToolResult
   from typing import Dict, Any

   class WeatherTool(BaseTool):
       def __init__(self):
           super().__init__(
               name="weather",
               description="Get current weather information for a location"
           )
       
       async def execute(self, location: str) -> ToolResult:
           # Your weather API integration here
           weather_data = await fetch_weather(location)
           
           return ToolResult(
               success=True,
               content=f"Weather in {location}: {weather_data['condition']}, {weather_data['temp']}°C",
               metadata={"location": location, "temp": weather_data['temp']}
           )

2. **Register Your Plugin**

.. code-block:: python

   from openagent.plugins.registry import register_plugin

   @register_plugin
   class WeatherPlugin:
       def get_tools(self):
           return [WeatherTool()]

3. **Install and Use**

.. code-block:: bash

   # Install your plugin
   openagent plugin install ./my-weather-plugin

   # Use in chat
   openagent chat
   > What's the weather like in London?

Plugin Configuration
--------------------

Plugins can define configuration schemas for user customization:

.. code-block:: python

   from pydantic import BaseModel

   class WeatherConfig(BaseModel):
       api_key: str
       default_units: str = "celsius"

   class WeatherTool(BaseTool):
       def __init__(self, config: WeatherConfig):
           super().__init__(name="weather", description="...")
           self.config = config

Best Practices
--------------

1. **Error Handling**: Always wrap your tool execution in proper error handling
2. **Async Support**: Use async/await for I/O operations
3. **Documentation**: Provide clear descriptions and examples
4. **Testing**: Include comprehensive tests for your plugin
5. **Security**: Validate all inputs and handle sensitive data securely

Plugin Examples
---------------

See the ``examples/plugins/`` directory for complete plugin implementations:

* **GitHub Plugin**: Interact with GitHub repositories
* **Docker Plugin**: Manage Docker containers
* **Database Plugin**: Query and manage databases

Distribution
------------

Plugins can be distributed as:

1. **Local Plugins**: Installed from local directory
2. **Git Repositories**: Installed directly from Git
3. **PyPI Packages**: Standard Python packages
4. **Plugin Registry**: Official OpenAgent plugin marketplace (coming soon)

Plugin API Reference
--------------------

For detailed API documentation, see:

* :doc:`../api/plugins`
* :doc:`../api/registry`
* :doc:`../api/loader`
