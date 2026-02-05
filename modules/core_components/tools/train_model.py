"""
Train Model Tab

Train custom voice models using finetuning datasets.
"""

import gradio as gr
from textwrap import dedent
from modules.core_components.tool_base import Tool, ToolConfig
# format_help_html comes from shared_state
from pathlib import Path


class TrainModelTool(Tool):
    """Train Model tool implementation."""

    config = ToolConfig(
        name="Train Model",
        module_name="tool_train_model",
        description="Train custom voice models",
        enabled=True,
        category="training"
    )

    @classmethod
    def create_tool(cls, shared_state):
        """Create Train Model tool UI."""
        components = {}

        # Get helper functions and config
        format_help_html = shared_state['format_help_html']
        get_dataset_folders = shared_state['get_dataset_folders']
        get_dataset_files = shared_state['get_dataset_files']
        get_trained_model_names = shared_state['get_trained_model_names']
        train_model = shared_state['train_model']
        DATASETS_DIR = shared_state['DATASETS_DIR']

        with gr.TabItem("Train Model"):
            gr.Markdown("Train a custom voice model using your finetuning dataset")
            with gr.Row():
                # Left column - Dataset selection and validation
                with gr.Column(scale=1):
                    gr.Markdown("### Dataset Selection")

                    components['train_folder_dropdown'] = gr.Dropdown(
                        choices=["(Select Dataset)"] + get_dataset_folders(),
                        value="(Select Dataset)",
                        label="Training Dataset",
                        info="Select prepared subfolder",
                        interactive=True
                    )

                    components['refresh_train_folder_btn'] = gr.Button("Refresh Datasets", size="sm")

                    components['ref_audio_dropdown'] = gr.Dropdown(
                        choices=[],
                        label="Select Reference Audio Track",
                        info="Select one sample from your dataset as reference",
                        interactive=True
                    )

                    components['ref_audio_preview'] = gr.Audio(
                        label="Preview",
                        type="filepath",
                        interactive=False
                    )

                    components['start_training_btn'] = gr.Button("Start Training", variant="primary", size="lg")

                    train_quick_guide = dedent("""\
                        **Quick Guide:**
                        1. Select dataset folder
                        2. Enter speaker name
                        3. Choose reference audio from dataset
                        4. Configure parameters & start training (defaults work well for most cases)

                        *See Help Guide tab → Train Model for detailed instructions*
                    """)
                    gr.HTML(
                        value=format_help_html(train_quick_guide),
                        container=True,
                        padding=True)

                # Right column - Training configuration
                with gr.Column(scale=1):
                    gr.Markdown("### Training Parameters")

                    components['batch_size_slider'] = gr.Slider(
                        minimum=1,
                        maximum=8,
                        value=2,
                        step=1,
                        label="Batch Size",
                        info="Reduce if you get out of memory errors"
                    )

                    components['learning_rate_slider'] = gr.Slider(
                        minimum=1e-6,
                        maximum=1e-4,
                        value=2e-6,
                        label="Learning Rate",
                        info="Default: 2e-6"
                    )

                    components['num_epochs_slider'] = gr.Slider(
                        minimum=1,
                        maximum=100,
                        value=5,
                        step=1,
                        label="Number of Epochs",
                        info="How many times to train on the full dataset"
                    )

                    components['save_interval_slider'] = gr.Slider(
                        minimum=0,
                        maximum=10,
                        value=5,
                        step=1,
                        label="Save Interval (Epochs)",
                        info="Save checkpoint every N epochs (0 = save every epoch)"
                    )

                    components['training_status'] = gr.Textbox(
                        label="Status",
                        lines=20,
                        interactive=False
                    )

        return components

    @classmethod
    def setup_events(cls, components, shared_state):
        """Wire up Train Model tab events."""

        # Get helper functions
        get_dataset_files = shared_state['get_dataset_files']
        get_trained_model_names = shared_state['get_trained_model_names']
        train_model = shared_state['train_model']
        input_trigger = shared_state['input_trigger']
        DATASETS_DIR = shared_state['DATASETS_DIR']

        def update_ref_audio_dropdown(folder):
            """Update reference audio dropdown when folder changes."""
            files = get_dataset_files(folder)
            return gr.update(choices=files, value=None), None

        def load_ref_audio_preview(folder, filename):
            """Load reference audio preview."""
            if not folder or not filename or folder == "(No folders)" or folder == "(Select Dataset)":
                return None
            audio_path = DATASETS_DIR / folder / filename
            if audio_path.exists():
                return str(audio_path)
            return None

        components['train_folder_dropdown'].change(
            update_ref_audio_dropdown,
            inputs=[components['train_folder_dropdown']],
            outputs=[components['ref_audio_dropdown'], components['ref_audio_preview']]
        )

        components['refresh_train_folder_btn'].click(
            lambda: gr.update(choices=["(Select Dataset)"] + shared_state['get_dataset_folders'](), value="(Select Dataset)"),
            outputs=[components['train_folder_dropdown']]
        )

        components['ref_audio_dropdown'].change(
            load_ref_audio_preview,
            inputs=[components['train_folder_dropdown'], components['ref_audio_dropdown']],
            outputs=[components['ref_audio_preview']]
        )

        # Hidden JSON for existing models list (JS-accessible)
        components['existing_models_json'] = gr.JSON(value=[], visible=False)

        # Function to show modal with current model list
        def show_training_modal():
            """Fetch current model list and prepare modal."""
            existing_models = get_trained_model_names()
            return existing_models

        components['start_training_btn'].click(
            fn=show_training_modal,
            inputs=None,
            outputs=[components['existing_models_json']]
        ).then(
            fn=None,
            inputs=[components['existing_models_json']],
            outputs=None,
            js="""
            (existingModels) => {
                const overlay = document.getElementById('input-modal-overlay');
                if (!overlay) return;

                const titleEl = document.getElementById('input-modal-title');
                const messageEl = document.getElementById('input-modal-message');
                const inputField = document.getElementById('input-modal-field');
                const submitBtn = document.getElementById('input-modal-submit-btn');
                const cancelBtn = document.getElementById('input-modal-cancel-btn');
                const errorEl = document.getElementById('input-modal-error');

                if (titleEl) titleEl.textContent = 'Start Training';
                if (messageEl) {
                    messageEl.textContent = 'Enter a name for this trained voice model:';
                    messageEl.style.display = 'block';
                    messageEl.classList.remove('error');
                    delete messageEl.dataset.originalMessage;
                }
                if (inputField) {
                    inputField.placeholder = 'e.g., MyVoice, Female-Narrator, John-Doe';
                    inputField.value = '';
                }
                if (submitBtn) {
                    submitBtn.textContent = 'Start Training';
                    submitBtn.setAttribute('data-context', 'train_model_');
                }
                if (cancelBtn) {
                    cancelBtn.setAttribute('data-context', 'train_model_');
                }
                if (errorEl) {
                    errorEl.classList.remove('show');
                    errorEl.textContent = '';
                }

                // Set up validation with current model list
                window.inputModalValidation = (value) => {
                    console.log('[VALIDATION] Called with value:', value);
                    console.log('[VALIDATION] existingModels:', existingModels);
                    console.log('[VALIDATION] Is array?', Array.isArray(existingModels));

                    if (!value || value.trim().length === 0) {
                        return 'Please enter a model name';
                    }

                    const trimmedValue = value.trim();
                    console.log('[VALIDATION] Trimmed value:', trimmedValue);
                    console.log('[VALIDATION] Checking if includes...');

                    if (existingModels && Array.isArray(existingModels)) {
                        console.log('[VALIDATION] Array contents:', existingModels);
                        const exists = existingModels.includes(trimmedValue);
                        console.log('[VALIDATION] Exists?', exists);

                        if (exists) {
                            return 'Model "' + trimmedValue + '" already exists!';
                        }
                    } else {
                        console.log('[VALIDATION] existingModels is not an array or is null');
                    }

                    return null;
                };

                overlay.classList.add('show');

                // Focus the input field after a brief delay
                setTimeout(() => {
                    if (inputField) {
                        inputField.focus();
                        inputField.select();
                    }
                }, 100);
            }
            """
        )

        # Handler for training modal submission
        def handle_train_model_input(input_value, folder, ref_audio, batch_size, lr, epochs, save_interval):
            """Process input modal submission for training."""
            # Context filtering: only process if this is our context
            if not input_value or not input_value.startswith("train_model_"):
                return gr.update()

            # Extract speaker name from context prefix
            parts = input_value.split("_")
            if len(parts) >= 3:
                speaker_name = "_".join(parts[2:-1])
                return train_model(folder, speaker_name, ref_audio, batch_size, lr, epochs, save_interval)

            return gr.update()

        input_trigger.change(
            handle_train_model_input,
            inputs=[input_trigger, components['train_folder_dropdown'], components['ref_audio_dropdown'], components['batch_size_slider'],
                    components['learning_rate_slider'], components['num_epochs_slider'], components['save_interval_slider']],
            outputs=[components['training_status']]
        )


# Export for tab registry
get_tool_class = lambda: TrainModelTool

if __name__ == "__main__":
    """Standalone testing of Train Model tool."""
    from modules.core_components.tools import run_tool_standalone
    run_tool_standalone(TrainModelTool, port=7863, title="Train Model - Standalone")