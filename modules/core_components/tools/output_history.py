"""
Output History Tab

Browse and manage previously generated audio files.

Standalone testing:
    python -m modules.core_components.tools.output_history
"""
# Setup path for standalone testing BEFORE imports
if __name__ == "__main__":
    import sys
    from pathlib import Path as PathLib
    project_root = PathLib(__file__).parent.parent.parent.parent
    sys.path.insert(0, str(project_root))

import gradio as gr
from pathlib import Path
from modules.core_components.tool_base import Tab, TabConfig


class OutputHistoryTab(Tab):
    """Output History tab implementation."""

    config = TabConfig(
        name="Output History",
        module_name="tab_output_history",
        description="Browse and manage generated audio files",
        enabled=True,
        category="utility"
    )

    @classmethod
    def create_tab(cls, shared_state):
        """Create Output History tab UI."""
        components = {}

        # Get OUTPUT_DIR from shared_state
        OUTPUT_DIR = shared_state.get('OUTPUT_DIR')

        # Helper function to get output files
        def get_output_files():
            """Get list of generated output files.

            Returns:
                List of filenames
            """
            if not OUTPUT_DIR or not OUTPUT_DIR.exists():
                return []
            files = sorted(OUTPUT_DIR.glob("*.wav"), key=lambda x: x.stat().st_mtime, reverse=True)
            return [f.name for f in files]

        with gr.TabItem("Output History"):
            gr.Markdown("Browse and manage previously generated audio files")
            with gr.Row():
                with gr.Column(scale=1):
                    with gr.Column(scale=1, elem_id="output-files-container"):
                        components['output_dropdown'] = gr.Radio(
                            choices=get_output_files(),
                            show_label=False,
                            interactive=True,
                            elem_id="output-files-group"
                        )
                    components['refresh_outputs_btn'] = gr.Button("Refresh", size="sm")

                with gr.Column(scale=1):
                    components['history_audio'] = gr.Audio(
                        label="Playback",
                        type="filepath"
                    )

                    components['history_metadata'] = gr.Textbox(
                        label="Generation Info",
                        interactive=False,
                        max_lines=15
                    )
                    components['delete_status'] = gr.Textbox(
                        label="Status",
                        interactive=False,
                        max_lines=1
                    )
                    # Hidden textbox to store selected filename for delete
                    components['selected_file'] = gr.Textbox(visible=False)
                    components['delete_output_btn'] = gr.Button("Delete", size="sm")

        return components

    @classmethod
    def setup_events(cls, components, shared_state):
        """Wire up Output History events."""

        # Get required items from shared_state
        OUTPUT_DIR = shared_state.get('OUTPUT_DIR')
        show_confirmation_modal_js = shared_state.get('show_confirmation_modal_js')
        confirm_trigger = shared_state.get('confirm_trigger')

        # Local helper functions
        # Local helper functions
        def get_output_files():
            """Get list of generated output files.

            Returns:
                List of filenames
            """
            if not OUTPUT_DIR or not OUTPUT_DIR.exists():
                return []
            files = sorted(OUTPUT_DIR.glob("*.wav"), key=lambda x: x.stat().st_mtime, reverse=True)
            return [f.name for f in files]

        def refresh_outputs():
            """Refresh the output file list."""
            return gr.update(choices=get_output_files(), value=None)

        def load_output_audio(selected_file):
            """Load a selected output file for playback and show metadata."""
            if not selected_file:
                return None, "", ""

            file_path = OUTPUT_DIR / selected_file

            if file_path.exists():
                metadata_file = file_path.with_suffix(".txt")
                if metadata_file.exists():
                    try:
                        metadata = metadata_file.read_text(encoding="utf-8")
                        return str(file_path), metadata, selected_file
                    except:
                        pass
                return str(file_path), "No metadata available", selected_file
            return None, "", ""

        def delete_output_file(action, selected_file):
            """Delete output file and metadata."""
            # Ignore empty calls or wrong context
            if not action or not action.strip() or not action.startswith("output_"):
                return gr.update(), gr.update(), gr.update(), ""

            # Handle cancel
            if "cancel" in action:
                return "Deletion cancelled", gr.update(), gr.update(), ""

            # Only process confirm
            if "confirm" not in action:
                return gr.update(), gr.update(), gr.update(), ""

            if not selected_file:
                return "[ERROR] No file selected", gr.update(), None, ""

            try:
                audio_path = OUTPUT_DIR / selected_file
                txt_path = audio_path.with_suffix(".txt")

                deleted = []
                if audio_path.exists():
                    audio_path.unlink()
                    deleted.append("audio")
                if txt_path.exists():
                    txt_path.unlink()
                    deleted.append("text")

                updated_list = get_output_files()
                msg = f"Deleted: {audio_path.name} ({', '.join(deleted)})" if deleted else "[ERROR] Files not found"
                return msg, gr.update(choices=updated_list, value=None), None, ""
            except Exception as e:
                return f"[ERROR] Error: {str(e)}", gr.update(), None, ""

        # Show modal on delete button choices=updated_list, value=Noneal available)
        if show_confirmation_modal_js and confirm_trigger:
            components['delete_output_btn'].click(
                fn=None,
                js=show_confirmation_modal_js(
                    title="Delete Output File?",
                    message="This will permanently delete the generated audio and its metadata. This action cannot be undone.",
                    confirm_button_text="Delete",
                    context="output_"
                )
            )

            # Process confirmation
            confirm_trigger.change(
                delete_output_file,
                inputs=[confirm_trigger, components['selected_file']],
                outputs=[components['delete_status'], components['output_dropdown'], components['history_audio'], components['selected_file']]
            )

        # Refresh button
        components['refresh_outputs_btn'].click(
            refresh_outputs,
            outputs=[components['output_dropdown']]
        )

        # Load on dropdown change
        components['output_dropdown'].change(
            load_output_audio,
            inputs=[components['output_dropdown']],
            outputs=[components['history_audio'], components['history_metadata'], components['selected_file']]
        )


# Export for registry
get_tab_class = lambda: OutputHistoryTab


# Standalone testing
if __name__ == "__main__":
    from modules.core_components.tools import run_tool_standalone
    run_tool_standalone(OutputHistoryTab, port=7868, title="Output History - Standalone")
