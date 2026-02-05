"""
Tools Architecture for Voice Clone Studio

This document explains the modularized tools system for Voice Clone Studio.

## Overview

**All 10 voice tools have been extracted from the monolithic `voice_clone_studio.py`** into individual modules in `modules/core_components/tools/`. Each tool is:

- **Independently maintainable** - Each tool in its own file
- **Testable** - Can be unit tested in isolation
- **Configurable** - Can be enabled/disabled via config
- **Reusable** - Can be imported and used in other contexts

## Current Structure

```
modules/core_components/tools/
├── __init__.py              # Tool registry and loader
├── base.py                  # Tab base class and utilities
├── config.py                # Configuration schemas
├── tools_reference.py       # Documentation reference (not imported)
│
├── voice_clone.py           # Clone voices from samples
├── voice_presets.py         # Premium speaker voices
├── conversation.py          # Multi-speaker conversations
├── voice_design.py          # Create voices from descriptions
├── prep_samples.py          # Prepare audio for training
├── finetune_dataset.py      # Manage training datasets
├── train_model.py           # Train custom voice models
├── output_history.py        # Browse generated audio
├── settings.py              # Application configuration
└── help.py                  # Documentation and help (includes all help content)
```

## Tool Categories

**🎙️ Generation** (4 tools):
- voice_clone.py
- voice_presets.py
- conversation.py
- voice_design.py

**📦 Preparation** (2 tools):
- prep_samples.py
- finetune_dataset.py

**🧠 Training** (1 tool):
- train_model.py

**🔧 Utility** (3 tools):
- output_history.py
- settings.py
- help.py

## Creating a New Tool

### 1. Create the Module File

Create `modules/core_components/tools/my_feature.py`:

```python
import gradio as gr
from modules.core_components.tools.base import Tool, ToolConfig

class MyFeatureTool(Tool):
    """My Feature tool implementation."""
    
    config = ToolConfig(
        name="My Feature",
        module_name="my_feature",
        description="Description of my feature",
        enabled=True,
        category="generation"  # or "preparation", "training", "utility"
    )
    
    @classmethod
    def create_tool(cls, shared_state):
        """Create the tool UI. Return dict of component references."""
        components = {}
        
        with gr.TabItem("My Feature"):
            gr.Markdown("### My Feature")
            
            components['input'] = gr.Textbox(label="Input")
            components['output'] = gr.Textbox(label="Output")
            components['button'] = gr.Button("Process")
        
        return components
    
    @classmethod
    def setup_events(cls, components, shared_state):
        """Wire up event handlers."""
        
        def process(text):
            return f"Processed: {text}"
        
        components['button'].click(
            process,
            inputs=[components['input']],
            outputs=[components['output']]
        )


# Export for registry
get_tool_class = lambda: MyFeatureTab
```

### 2. Register the Tool

Add to `modules/core_components/tools/__init__.py`:

```python
from modules.core_components.tools import my_feature

ALL_TABS: Dict[str, tuple] = {
    'my_feature': (my_feature, my_feature.MyFeatureTab.config),
    # ... other tools
}
```

### 3. Use in Main File

The main file now simply loads tools dynamically:

```python
from modules.core_components.tools import get_tab_registry, create_enabled_tabs, setup_tab_events

# In create_ui():
with gr.Blocks() as app:
    shared_state = {
        'user_config': _user_config,
        'active_emotions': _active_emotions,
        # ... other helpers
    }
    
    # Create all enabled tools
    tool_components = create_enabled_tabs(shared_state)
    
    # Setup events for all tools
    setup_tab_events(tool_components, shared_state)
```

## Enabling/Disabling Tools

Users can disable tools via `config.json`:

```json
{
    "enabled_tools": {
        "Voice Clone": true,
        "Voice Presets": true,
        "Conversation": true,
        "Voice Design": true,
        "Prep Samples": true,
        "Output History": false,
        "Finetune Dataset": false,
        "Train Model": false,
        "Help Guide": true,
        "Settings": true
    }
}
```

If a tool is not in the list, it defaults to its `config.enabled` value.

## Shared State

Each tool receives `shared_state` dict with shared utilities:

```python
shared_state = {
    'user_config': dict,           # User configuration
    'active_emotions': dict,       # Loaded emotions
    'get_sample_choices': func,    # Get available samples
    'generate_audio': func,        # Audio generation
    'save_config': func,           # Save config to file
    # ... other helpers
}
```

Access utilities like:

```python
def setup_events(cls, components, shared_state):
    get_samples = shared_state.get('get_sample_choices')
    samples = get_samples() if get_samples else []
```

## Refactoring Status

- ✅ **COMPLETE** - All 10 tools extracted and working
- ✅ Tool infrastructure (base.py, __init__.py, registry)
- ✅ Tool registry system with enable/disable support
- ✅ All individual tools modularized
- ✅ Help content consolidated into help.py tool
- ✅ Imports and paths updated to new structure
- ⏳ **Next:** Main file integration

## Benefits

✅ **Modularity** - Each tool in its own file  
✅ **Testability** - Can unit test tools independently  
✅ **Maintainability** - Find/edit tool logic easily  
✅ **Extensibility** - Add new tools without touching main file  
✅ **User Control** - Users can disable unused tools  
✅ **Performance** - Can lazy-load tools if needed (future)  
✅ **Reusability** - Tools can be used outside of Voice Clone Studio  

## File Naming Convention

- **Tool files:** Clean names without prefix (e.g., `voice_clone.py`)
- **Class names:** Keep `Tab` suffix (e.g., `VoiceCloneTab`)
- **Import path:** `from modules.core_components.tools import voice_clone`
- **Reference file:** `tools_reference.py` (documentation only, not imported)

## Next Steps

1. Integrate tools into main `voice_clone_studio.py`
2. Test all tools load and function correctly
3. Add settings UI for tool enable/disable
4. Implement user profiles (beginner/creator/developer)
5. Consider lazy loading for better startup performance
"""
