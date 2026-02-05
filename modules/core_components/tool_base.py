"""
Base utilities for tab modules.

Provides common patterns and helpers for all tabs.
"""

from dataclasses import dataclass
from typing import Dict, Any, Callable, Optional


@dataclass
class TabConfig:
    """Configuration for a tab."""
    name: str  # Tab name/title
    module_name: str  # Python module name (e.g., 'tab_voice_clone')
    description: str  # Tab description for settings
    enabled: bool = True  # Whether tab is enabled
    category: str = "general"  # Tab category: "generation", "training", "utility", "settings"


class Tab:
    """Base class for tab modules."""

    config: TabConfig

    @classmethod
    def create_tab(cls, shared_state: Dict[str, Any]):
        """
        Create and return the tab UI.

        Args:
            shared_state: Dictionary with shared globals
                - 'active_emotions': Active emotions dict
                - 'user_config': User configuration
                - 'get_*': Helper functions

        Returns:
            dict with all component references for event wiring
        """
        raise NotImplementedError

    @classmethod
    def setup_events(cls, components: Dict[str, Any], shared_state: Dict[str, Any]):
        """
        Wire up all event handlers for this tab.

        Args:
            components: Component references returned by create_tab()
            shared_state: Dictionary with shared globals
        """
        raise NotImplementedError


def get_helper(shared_state: Dict[str, Any], helper_name: str) -> Callable:
    """Get a helper function from shared state."""
    if helper_name not in shared_state:
        raise KeyError(f"Helper '{helper_name}' not found in shared state")
    return shared_state[helper_name]
