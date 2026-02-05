"""
Finetune Dataset Tool

Manage and prepare finetuning datasets.
"""

import gradio as gr
from textwrap import dedent
from modules.core_components.tool_base import Tool, ToolConfig
# format_help_html comes from shared_state


class FinetuneDatasetTool(Tool):
    """Finetune Dataset Tool implementation."""

    config = ToolConfig(
        name="Finetune Dataset",
        module_name="tool_finetune_dataset",
        description="Manage training datasets",
        enabled=True,
        category="preparation"
    )

    @classmethod
    def create_tool(cls, shared_state):
        """Create Finetune Dataset tool UI."""
        components = {}

        # Get helper functions and config
        format_help_html = shared_state['format_help_html']
        get_dataset_folders = shared_state['get_dataset_folders']
        get_dataset_files = shared_state['get_dataset_files']
        load_dataset_item = shared_state['load_dataset_item']
        save_dataset_transcript = shared_state['save_dataset_transcript']
        delete_dataset_item = shared_state['delete_dataset_item']
        auto_transcribe_finetune = shared_state['auto_transcribe_finetune']
        batch_transcribe_folder = shared_state['batch_transcribe_folder']
        save_trimmed_audio = shared_state['save_trimmed_audio']
        normalize_audio = shared_state['normalize_audio']
        convert_to_mono = shared_state['convert_to_mono']
        clean_audio = shared_state['clean_audio']
        _user_config = shared_state['_user_config']
        WHISPER_AVAILABLE = shared_state['WHISPER_AVAILABLE']
        DEEPFILTER_AVAILABLE = shared_state['DEEPFILTER_AVAILABLE']
        DATASETS_DIR = shared_state['DATASETS_DIR']
        show_confirmation_modal_js = shared_state['show_confirmation_modal_js']
        save_preference = shared_state['save_preference']
        confirm_trigger = shared_state['confirm_trigger']

        with gr.TabItem("Finetune Dataset"):
            gr.Markdown("Manage and prepare your finetuning dataset")
            with gr.Row():
                # Left - File list and management
                with gr.Column(scale=1):
                    gr.Markdown("### Dataset Files")

                    components['finetune_folder_dropdown'] = gr.Dropdown(
                        choices=["(Select Dataset)"] + get_dataset_folders(),
                        value="(Select Dataset)",
                        label="Dataset Folder",
                        info="Subfolders in datasets",
                        interactive=True,
                    )

                    components['refresh_folder_btn'] = gr.Button("Refresh Folders", size="sm")

                    with gr.Column(elem_id="finetune-files-container"):
                        components['finetune_dropdown'] = gr.Radio(
                            choices=[],
                            show_label=False,
                            interactive=True,
                            elem_id="finetune-files-group"
                        )

                    with gr.Row():
                        components['refresh_finetune_btn'] = gr.Button("Refresh", size="sm", scale=1)
                        components['delete_finetune_btn'] = gr.Button("Delete", size="sm", scale=1)

                    components['finetune_audio_preview'] = gr.Audio(
                        label="Audio Preview & Trim",
                        type="filepath",
                        interactive=True
                    )

                    with gr.Row():
                        components['finetune_clean_btn'] = gr.Button("AI Denoise", size="sm", visible=DEEPFILTER_AVAILABLE)
                        components['finetune_normalize_btn'] = gr.Button("Normalize Volume", size="sm")
                        components['finetune_mono_btn'] = gr.Button("Convert to Mono", size="sm")

                    components['save_trimmed_btn'] = gr.Button("Save Audio", size="sm", variant="primary")

                    gr.Markdown("### Transcription Settings")

                    available_models = ['VibeVoice ASR']
                    if WHISPER_AVAILABLE:
                        available_models.insert(0, 'Whisper')

                    default_model = _user_config.get("transcribe_model", "Whisper")
                    if default_model not in available_models:
                        default_model = available_models[0]

                    components['finetune_transcribe_model'] = gr.Radio(
                        choices=available_models,
                        value=default_model,
                        label="Transcription Model",
                        info="Choose transcription engine"
                    )

                    components['finetune_transcribe_lang'] = gr.Dropdown(
                        choices=["Auto-detect", "English", "Chinese", "Japanese", "Korean",
                                 "French", "German", "Spanish", "Russian"],
                        value=_user_config.get("whisper_language", "Auto-detect"),
                        label="Language (Whisper only)",
                        visible=(_user_config.get("transcribe_model", "Whisper") == "Whisper")
                    )

                # Right - Transcript editor
                with gr.Column(scale=2):
                    gr.Markdown("### Edit Transcript")

                    components['finetune_transcript'] = gr.Textbox(
                        label="Transcript",
                        placeholder="Load an audio file or auto-transcribe to edit the transcript...",
                        lines=10,
                        info="Edit the transcript to match the audio exactly"
                    )

                    with gr.Row():
                        components['auto_transcribe_btn'] = gr.Button("Auto-Transcribe", variant="primary", scale=1)
                        components['save_transcript_btn'] = gr.Button("Save Transcript", variant="primary", scale=1)

                    with gr.Column(scale=1):
                        gr.Markdown("#### Batch Transcript\n_Transcribes entire dataset_", container=True)
                        components['batch_transcribe_btn'] = gr.Button("Batch Transcribe", variant="primary", size="lg")
                        with gr.Row():
                            components['batch_replace_existing'] = gr.Checkbox(
                                label="Replace existing transcripts",
                                info="If unchecked, only files without transcripts will be processed",
                                value=False
                            )

                        components['finetune_status'] = gr.Textbox(
                            label="Status",
                            interactive=False,
                            lines=5,
                            max_lines=15
                        )

                        finetune_quick_guide = dedent("""\
                        **Quick Guide:**
                        - Create subfolders in /datasets to organize training sets
                        - Use **Batch Transcribe** to Transcribe all files at once
                        - Or edit individual files, trimming track and adjusting transcripts as needed.

                        *See Help Guide tab → Finetune Dataset for detailed instructions*
                        """)
                        gr.HTML(
                            value=format_help_html(finetune_quick_guide),
                            container=True,
                            padding=True
                        )

        return components

    @classmethod
    def setup_events(cls, components, shared_state):
        """Wire up Finetune Dataset tab events."""

        # Get helper functions
        get_dataset_folders = shared_state['get_dataset_folders']
        get_dataset_files = shared_state['get_dataset_files']
        load_dataset_item = shared_state['load_dataset_item']
        save_dataset_transcript = shared_state['save_dataset_transcript']
        delete_dataset_item = shared_state['delete_dataset_item']
        auto_transcribe_finetune = shared_state['auto_transcribe_finetune']
        batch_transcribe_folder = shared_state['batch_transcribe_folder']
        save_trimmed_audio = shared_state['save_trimmed_audio']
        normalize_audio = shared_state['normalize_audio']
        convert_to_mono = shared_state['convert_to_mono']
        clean_audio = shared_state['clean_audio']
        show_confirmation_modal_js = shared_state['show_confirmation_modal_js']
        save_preference = shared_state['save_preference']
        confirm_trigger = shared_state['confirm_trigger']
        DATASETS_DIR = shared_state['DATASETS_DIR']

        def refresh_folder_list():
            """Refresh folder list."""
            folders = get_dataset_folders()
            return gr.update(choices=["(Select Dataset)"] + folders, value="(Select Dataset)")

        def refresh_finetune_list(folder):
            """Refresh file list for the current folder."""
            files = get_dataset_files(folder)
            return gr.update(choices=files, value=None)

        def update_file_list(folder):
            """Update file list when folder changes."""
            files = get_dataset_files(folder)
            return gr.update(choices=files, value=None)

        # When folder changes, update file list
        components['finetune_folder_dropdown'].change(
            update_file_list,
            inputs=[components['finetune_folder_dropdown']],
            outputs=[components['finetune_dropdown']]
        )

        components['refresh_folder_btn'].click(
            refresh_folder_list,
            outputs=[components['finetune_folder_dropdown']]
        )

        components['refresh_finetune_btn'].click(
            refresh_finetune_list,
            inputs=[components['finetune_folder_dropdown']],
            outputs=[components['finetune_dropdown']]
        )

        components['finetune_dropdown'].change(
            load_dataset_item,
            inputs=[components['finetune_folder_dropdown'], components['finetune_dropdown']],
            outputs=[components['finetune_audio_preview'], components['finetune_transcript']]
        )

        components['save_transcript_btn'].click(
            save_dataset_transcript,
            inputs=[components['finetune_folder_dropdown'], components['finetune_dropdown'], components['finetune_transcript']],
            outputs=[components['finetune_status']]
        )

        # Show modal on delete button click
        components['delete_finetune_btn'].click(
            fn=None,
            inputs=None,
            outputs=None,
            js=show_confirmation_modal_js(
                title="Delete Dataset Item?",
                message="This will permanently delete the audio file and its transcript. This action cannot be undone.",
                confirm_button_text="Delete",
                context="finetune_"
            )
        )

        # Process confirmation
        confirm_trigger.change(
            delete_dataset_item,
            inputs=[confirm_trigger, components['finetune_folder_dropdown'], components['finetune_dropdown']],
            outputs=[components['finetune_status'], components['finetune_dropdown']]
        )

        components['auto_transcribe_btn'].click(
            auto_transcribe_finetune,
            inputs=[components['finetune_folder_dropdown'], components['finetune_dropdown'], components['finetune_transcribe_model'], components['finetune_transcribe_lang']],
            outputs=[components['finetune_transcript'], components['finetune_status']]
        )

        components['batch_transcribe_btn'].click(
            batch_transcribe_folder,
            inputs=[components['finetune_folder_dropdown'], components['batch_replace_existing'], components['finetune_transcribe_lang'], components['finetune_transcribe_model']],
            outputs=[components['finetune_status']]
        )

        def save_and_reload(folder, filename, audio):
            """Save trimmed audio, then return values to refresh and reload."""
            if folder and folder != "(No folders)":
                base_dir = DATASETS_DIR / folder
            else:
                base_dir = DATASETS_DIR

            saved_audio, status = save_trimmed_audio(str(base_dir / filename) if filename else None, audio)
            return None, status, filename

        # Normalize
        components['finetune_normalize_btn'].click(
            normalize_audio,
            inputs=[components['finetune_audio_preview']],
            outputs=[components['finetune_audio_preview']]
        )

        # Convert to mono
        components['finetune_mono_btn'].click(
            convert_to_mono,
            inputs=[components['finetune_audio_preview']],
            outputs=[components['finetune_audio_preview']]
        )

        # Clean audio
        components['finetune_clean_btn'].click(
            clean_audio,
            inputs=[components['finetune_audio_preview']],
            outputs=[components['finetune_audio_preview']]
        )

        save_trimmed_event = components['save_trimmed_btn'].click(
            save_and_reload,
            inputs=[components['finetune_folder_dropdown'], components['finetune_dropdown'], components['finetune_audio_preview']],
            outputs=[components['finetune_audio_preview'], components['finetune_status'], components['finetune_dropdown']]
        )

        # After saving, reload the same file
        save_trimmed_event.then(
            load_dataset_item,
            inputs=[components['finetune_folder_dropdown'], components['finetune_dropdown']],
            outputs=[components['finetune_audio_preview'], components['finetune_transcript']]
        )

        # Toggle language dropdown based on transcribe model
        def toggle_finetune_transcribe_settings(model):
            return gr.update(visible=(model == "Whisper"))

        components['finetune_transcribe_model'].change(
            toggle_finetune_transcribe_settings,
            inputs=[components['finetune_transcribe_model']],
            outputs=[components['finetune_transcribe_lang']]
        )

        # Save finetune transcription preferences
        components['finetune_transcribe_model'].change(
            lambda x: save_preference("transcribe_model", x),
            inputs=[components['finetune_transcribe_model']],
            outputs=[]
        )

        components['finetune_transcribe_lang'].change(
            lambda x: save_preference("whisper_language", x),
            inputs=[components['finetune_transcribe_lang']],
            outputs=[]
        )


# Export for tab registry
get_tool_class = lambda: FinetuneDatasetTool

if __name__ == "__main__":
    """Standalone testing of Finetune Dataset tool."""
    from modules.core_components.tools import run_tool_standalone
    run_tool_standalone(FinetuneDatasetTool, port=7864, title="Finetune Dataset - Standalone")
