"""
Weather Plugin for OpenAgent

This plugin demonstrates how to create custom tools for OpenAgent.
It provides weather information using a mock weather service.
"""

import asyncio
from typing import Dict, Any, Optional
from pydantic import BaseModel, field_validator

from openagent.core.base import BaseTool, ToolResult


class WeatherConfig(BaseModel):
    """Configuration for the weather plugin."""
    
    api_key: Optional[str] = None
    default_units: str = "celsius"
    timeout: int = 10
    
    @field_validator("default_units")
    @classmethod
    def validate_units(cls, v: str) -> str:
        valid_units = ["celsius", "fahrenheit", "kelvin"]
        if v.lower() not in valid_units:
            raise ValueError(f"Invalid units: {v}. Must be one of {valid_units}")
        return v.lower()


class WeatherTool(BaseTool):
    """Tool for getting weather information."""
    
    def __init__(self, config: Optional[WeatherConfig] = None):
        super().__init__(
            name="weather",
            description="Get current weather information for a specific location"
        )
        self.config = config or WeatherConfig()
    
    async def execute(self, location: str) -> ToolResult:
        """
        Get weather information for a location.
        
        Args:
            location: The location to get weather for (e.g., "London", "New York")
            
        Returns:
            ToolResult containing weather information
        """
        try:
            # Simulate API call with timeout
            weather_data = await asyncio.wait_for(
                self._fetch_weather(location),
                timeout=self.config.timeout
            )
            
            # Format the response
            temp_unit = "°C" if self.config.default_units == "celsius" else "°F"
            content = (
                f"🌤️ Weather in {location}:\n"
                f"Condition: {weather_data['condition']}\n"
                f"Temperature: {weather_data['temperature']}{temp_unit}\n"
                f"Humidity: {weather_data['humidity']}%\n"
                f"Wind: {weather_data['wind_speed']} km/h"
            )
            
            return ToolResult(
                success=True,
                content=content,
                metadata={
                    "location": location,
                    "temperature": weather_data["temperature"],
                    "condition": weather_data["condition"],
                    "units": self.config.default_units
                }
            )
            
        except asyncio.TimeoutError:
            return ToolResult(
                success=False,
                content=f"❌ Weather request timed out for {location}"
            )
        except Exception as e:
            return ToolResult(
                success=False,
                content=f"❌ Failed to get weather for {location}: {str(e)}"
            )
    
    async def _fetch_weather(self, location: str) -> Dict[str, Any]:
        """
        Mock weather API call.
        In a real implementation, this would call a weather service.
        """
        # Simulate API delay
        await asyncio.sleep(0.5)
        
        # Mock weather data based on location
        mock_data = {
            "london": {"condition": "Cloudy", "temperature": 15, "humidity": 80, "wind_speed": 10},
            "new york": {"condition": "Sunny", "temperature": 22, "humidity": 65, "wind_speed": 8},
            "tokyo": {"condition": "Rainy", "temperature": 18, "humidity": 85, "wind_speed": 12},
            "sydney": {"condition": "Sunny", "temperature": 25, "humidity": 60, "wind_speed": 15},
        }
        
        location_key = location.lower().strip()
        if location_key in mock_data:
            return mock_data[location_key]
        
        # Default weather for unknown locations
        return {
            "condition": "Partly Cloudy",
            "temperature": 20,
            "humidity": 70,
            "wind_speed": 5
        }


# Plugin registration (this would be handled by the plugin system)
class WeatherPlugin:
    """Weather plugin for OpenAgent."""
    
    def __init__(self, config: Optional[Dict[str, Any]] = None):
        self.config = WeatherConfig(**(config or {}))
    
    def get_tools(self) -> list[BaseTool]:
        """Return the tools provided by this plugin."""
        return [WeatherTool(self.config)]
    
    def get_info(self) -> Dict[str, Any]:
        """Return plugin information."""
        return {
            "name": "weather",
            "version": "1.0.0",
            "description": "Get weather information for locations",
            "author": "OpenAgent Contributors",
            "tags": ["weather", "api", "information"]
        }


# Factory function for plugin loading
def create_plugin(config: Optional[Dict[str, Any]] = None) -> WeatherPlugin:
    """Create and return a WeatherPlugin instance."""
    return WeatherPlugin(config)
