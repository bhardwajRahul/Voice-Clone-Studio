"""
Prep Samples Tab

Prepare audio samples for voice cloning with robust audio handling.
"""
# Setup path for standalone testing BEFORE imports
if __name__ == "__main__":
    import sys
    from pathlib import Path
    project_root = Path(__file__).parent.parent.parent.parent
    sys.path.insert(0, str(project_root))
    # Also add modules directory for vibevoice_asr imports
    sys.path.insert(0, str(project_root / "modules"))

import gradio as gr
import re
from pathlib import Path
from modules.core_components.tool_base import Tool, ToolConfig


class PrepSamplesTool(Tool):
    """Prep Samples tab implementation with robust audio handling."""

    config = ToolConfig(
        name="Prep Samples",
        module_name="tool_prep_samples",
        description="Prepare and manage voice samples",
        enabled=True,
        category="preparation"
    )

    @classmethod
    def create_tool(cls, shared_state):
        """Create Prep Samples tool UI."""
        components = {}

        # Get helper functions and config
        get_sample_choices = shared_state['get_sample_choices']
        _user_config = shared_state['_user_config']
        LANGUAGES = shared_state['LANGUAGES']
        WHISPER_AVAILABLE = shared_state['WHISPER_AVAILABLE']
        DEEPFILTER_AVAILABLE = shared_state['DEEPFILTER_AVAILABLE']

        with gr.TabItem("Prep Samples"):
            gr.Markdown("Prepare audio samples for voice cloning")
            with gr.Row():
                # Left column - Existing samples browser
                with gr.Column(scale=1):
                    gr.Markdown("### Existing Samples")

                    existing_sample_choices = get_sample_choices()
                    components['existing_sample_dropdown'] = gr.Dropdown(
                        choices=existing_sample_choices,
                        value=existing_sample_choices[0] if existing_sample_choices else None,
                        label="Browse Samples",
                        info="Select a sample to preview or edit"
                    )

                    with gr.Row():
                        components['preview_sample_btn'] = gr.Button("Preview Sample", size="sm")
                        components['refresh_preview_btn'] = gr.Button("Refresh", size="sm")
                        components['load_sample_btn'] = gr.Button("Load to Editor", size="sm")

                    with gr.Row():
                        components['clear_cache_btn'] = gr.Button("Clear Cache", size="sm")
                        components['delete_sample_btn'] = gr.Button("Delete", size="sm", variant="stop")

                    components['existing_sample_audio'] = gr.Audio(
                        label="Sample Preview",
                        type="filepath",
                        interactive=False
                    )

                    components['existing_sample_text'] = gr.Textbox(
                        label="Sample Text",
                        max_lines=10,
                        interactive=False
                    )

                    components['existing_sample_info'] = gr.Textbox(
                        label="Info",
                        interactive=False,
                        lines=3
                    )

                    # Transcription settings
                    gr.Markdown("### Transcription Settings")

                    with gr.Row():
                        components['whisper_language'] = gr.Dropdown(
                            choices=["Auto-detect"] + LANGUAGES[1:],
                            value=_user_config.get("whisper_language", "Auto-detect"),
                            label="Language",
                        )

                        available_models = ['VibeVoice ASR']
                        if WHISPER_AVAILABLE:
                            available_models.insert(0, 'Whisper')

                        default_model = _user_config.get("transcribe_model", "Whisper")
                        if default_model not in available_models:
                            default_model = available_models[0]

                        components['transcribe_model'] = gr.Dropdown(
                            choices=available_models,
                            value=default_model,
                            label="Model",
                        )

                # Right column - Audio/Video editing
                with gr.Column(scale=2):
                    gr.Markdown("### Edit Audio/Video")

                    components['prep_file_input'] = gr.File(
                        label="Audio or Video File",
                        type="filepath",
                        file_types=["audio", "video"],
                        interactive=True
                    )

                    components['prep_audio_editor'] = gr.Audio(
                        label="Audio Editor (Use Trim icon ✂️ to edit)",
                        type="filepath",
                        interactive=True,
                        visible=False
                    )

                    with gr.Row():
                        components['clear_btn'] = gr.Button("Clear", scale=1, size="sm")
                        components['clean_btn'] = gr.Button("AI Denoise", scale=2, size="sm",
                                                            variant="secondary", visible=DEEPFILTER_AVAILABLE)
                        components['normalize_btn'] = gr.Button("Normalize Volume", scale=2, size="sm")
                        components['mono_btn'] = gr.Button("Convert to Mono", scale=2, size="sm")

                    components['prep_audio_info'] = gr.Textbox(
                        label="Audio Info",
                        interactive=False,
                        lines=2
                    )

                    gr.Markdown("### Transcription / Reference Text")
                    components['transcription_output'] = gr.Textbox(
                        label="Text",
                        lines=4,
                        max_lines=10,
                        interactive=True,
                        placeholder="Transcription will appear here, or enter/edit text manually..."
                    )

                    with gr.Row():
                        components['transcribe_btn'] = gr.Button("Transcribe Audio", variant="primary")
                        components['save_sample_btn'] = gr.Button("Save Sample", variant="primary")

                    components['save_status'] = gr.Textbox(label="Status", interactive=False, lines=2)

            return components

    @classmethod
    def setup_events(cls, components, shared_state):
        """Wire up Prep Samples tab events."""

        # Get helper functions
        get_sample_choices = shared_state['get_sample_choices']
        get_available_samples = shared_state['get_available_samples']
        load_sample_details = shared_state['load_sample_details']
        get_prompt_cache_path = shared_state['get_prompt_cache_path']
        play_completion_beep = shared_state.get('play_completion_beep')
        save_preference = shared_state['save_preference']
        show_confirmation_modal_js = shared_state['show_confirmation_modal_js']
        show_input_modal_js = shared_state['show_input_modal_js']
        confirm_trigger = shared_state['confirm_trigger']
        input_trigger = shared_state['input_trigger']
        TEMP_DIR = shared_state['TEMP_DIR']
        SAMPLES_DIR = shared_state['SAMPLES_DIR']
        WHISPER_AVAILABLE = shared_state['WHISPER_AVAILABLE']

        # Audio utility functions from shared_state
        is_audio_file = shared_state['is_audio_file']
        is_video_file = shared_state['is_video_file']
        extract_audio_from_video = shared_state['extract_audio_from_video']
        get_audio_duration = shared_state['get_audio_duration']
        format_time = shared_state['format_time']
        normalize_audio = shared_state['normalize_audio']
        convert_to_mono = shared_state['convert_to_mono']
        clean_audio = shared_state['clean_audio']
        save_as_sample = shared_state['save_as_sample']

        # ASR/model getters
        get_whisper_model = shared_state.get('get_whisper_model')
        get_vibe_voice_model = shared_state.get('get_vibe_voice_model')
        get_deepfilter_model = shared_state.get('get_deepfilter_model')

        def on_prep_audio_load_handler(audio_file):
            """When audio/video is loaded, extract info and convert if needed."""
            if audio_file is None:
                return None, "No file loaded"

            try:
                # Check if video
                if is_video_file(audio_file):
                    print(f"Video file detected: {Path(audio_file).name}")
                    audio_path = extract_audio_from_video(audio_file)

                    if audio_path:
                        duration = get_audio_duration(audio_path)
                        info = f"[VIDEO] Audio extracted\nDuration: {format_time(duration)} ({duration:.2f}s)"
                        return audio_path, info
                    else:
                        return None, "[ERROR] Failed to extract audio. Ensure file has audio track."

                # Audio file
                elif is_audio_file(audio_file):
                    duration = get_audio_duration(audio_file)
                    info = f"Duration: {format_time(duration)} ({duration:.2f}s)"
                    return audio_file, info
                else:
                    return None, "[ERROR] Unsupported file type. Upload audio (.wav, .mp3) or video (.mp4, .mov)"

            except Exception as e:
                return None, f"Error: {str(e)}"

        def load_sample_to_editor(sample_name):
            """Load sample into the working audio editor."""
            if not sample_name:
                return None, None, "", "No sample selected", gr.update(visible=False)

            audio_path, ref_text, info = load_sample_details(sample_name)

            if audio_path:
                # Format simple info for editor
                duration = get_audio_duration(audio_path)
                editor_info = f"Duration: {format_time(duration)} ({duration:.2f}s)"
                return audio_path, audio_path, ref_text, editor_info, gr.update(visible=True)

            return None, None, "", "Sample not found", gr.update(visible=False)

        def load_existing_sample_handler(sample_name):
            """Load an existing sample for preview."""
            if not sample_name:
                return None, "", "No sample selected"

            audio_path, ref_text, info_text = load_sample_details(sample_name)
            return audio_path, ref_text, info_text

        def refresh_samples_handler():
            """Refresh samples dropdown."""
            return gr.update(choices=get_sample_choices())

        def delete_sample_handler(action, sample_name):
            """Delete a sample (wav, json, and cache files)."""
            # Ignore empty calls or wrong context
            if not action or not action.strip() or not action.startswith("sample_"):
                return gr.update(), gr.update(), gr.update()

            # Handle cancel
            if "cancel" in action:
                return "Deletion cancelled", gr.update(), gr.update()

            # Only process confirm
            if "confirm" not in action:
                return gr.update(), gr.update(), gr.update()

            if not sample_name:
                return "[ERROR] No sample selected", gr.update(), gr.update()

            try:
                import os

                # Delete wav file
                wav_path = SAMPLES_DIR / f"{sample_name}.wav"
                if wav_path.exists():
                    os.remove(wav_path)

                # Delete json file
                json_path = SAMPLES_DIR / f"{sample_name}.json"
                if json_path.exists():
                    os.remove(json_path)

                # Delete cache files
                for model_size in ["0.6B", "1.7B"]:
                    cache_path = get_prompt_cache_path(sample_name, model_size)
                    if cache_path.exists():
                        os.remove(cache_path)

                # Refresh choices
                choices = get_sample_choices()

                return (
                    f"Sample '{sample_name}' deleted",
                    gr.update(choices=choices),
                    gr.update(choices=choices)
                )

            except Exception as e:
                return f"[ERROR] Error deleting sample: {str(e)}", gr.update(), gr.update()

        def clear_sample_cache_handler(sample_name):
            """Clear voice prompt cache for a sample."""
            if not sample_name:
                return "[ERROR] No sample selected", gr.update()

            try:
                import os
                deleted = []

                for model_size in ["0.6B", "1.7B"]:
                    cache_path = get_prompt_cache_path(sample_name, model_size)
                    if cache_path.exists():
                        os.remove(cache_path)
                        deleted.append(model_size)

                if deleted:
                    # Reload info
                    _, _, new_info = load_sample_details(sample_name)
                    return f"Cache cleared for: {', '.join(deleted)}", new_info
                else:
                    return "No cache files found", gr.update()

            except Exception as e:
                return f"[ERROR] Error clearing cache: {str(e)}", gr.update()

        def normalize_audio_handler(audio_file):
            """Normalize audio volume."""
            return normalize_audio(audio_file)

        def convert_to_mono_handler(audio_file):
            """Convert stereo to mono."""
            return convert_to_mono(audio_file)

        def clean_audio_handler(audio_file, progress=gr.Progress()):
            """Clean audio using DeepFilterNet."""
            return clean_audio(audio_file, progress)

        def transcribe_audio_handler(audio_file, whisper_language, transcribe_model, progress=gr.Progress()):
            """Transcribe audio using Whisper or VibeVoice ASR."""
            if audio_file is None:
                return "[ERROR] Please load an audio file first."

            try:
                if transcribe_model == "VibeVoice ASR":
                    progress(0.2, desc="Loading VibeVoice ASR...")

                    if not get_vibe_voice_model:
                        return "[ERROR] VibeVoice ASR not available"

                    model = get_vibe_voice_model()
                    progress(0.4, desc="Transcribing...")
                    result = model.transcribe(audio_file)

                else:  # Whisper
                    if not WHISPER_AVAILABLE:
                        return "[ERROR] Whisper not available. Use VibeVoice ASR instead."

                    progress(0.2, desc="Loading Whisper...")

                    if not get_whisper_model:
                        return "[ERROR] Whisper not available"

                    model = get_whisper_model()
                    progress(0.4, desc="Transcribing...")

                    # Language options
                    options = {}
                    if whisper_language and whisper_language != "Auto-detect":
                        lang_code = {
                            "English": "en", "Chinese": "zh", "Japanese": "ja",
                            "Korean": "ko", "German": "de", "French": "fr",
                            "Russian": "ru", "Portuguese": "pt", "Spanish": "es",
                            "Italian": "it"
                        }.get(whisper_language, None)
                        if lang_code:
                            options["language"] = lang_code

                    result = model.transcribe(audio_file, **options)

                progress(1.0, desc="Done!")
                transcription = result["text"].strip()

                # Clean VibeVoice output (remove [Speaker N]:)
                if transcribe_model == "VibeVoice ASR":
                    transcription = re.sub(r'\[.*?\]\s*:', '', transcription)
                    transcription = re.sub(r'\[.*?\]', '', transcription)
                    transcription = ' '.join(transcription.split())

                if play_completion_beep:
                    play_completion_beep()

                return transcription

            except Exception as e:
                import traceback
                print(f"Error in transcribe:\n{traceback.format_exc()}")
                return f"[ERROR] Error transcribing: {str(e)}"

        def handle_save_sample_input(input_value, audio, transcription):
            """Process save sample modal input."""
            if not input_value or not input_value.startswith("save_sample_"):
                return gr.update(), gr.update(), gr.update()

            parts = input_value.split("_")
            if len(parts) >= 3:
                if parts[2] == "cancel":
                    return gr.update(), gr.update(), gr.update()

                sample_name = "_".join(parts[2:-1])
                status, dropdown1, dropdown2, _ = save_as_sample(audio, transcription, sample_name)
                return status, dropdown1, dropdown2

            return gr.update(), gr.update(), gr.update()

        # Wire up events

        # Load sample to editor
        components['load_sample_btn'].click(
            load_sample_to_editor,
            inputs=[components['existing_sample_dropdown']],
            outputs=[components['prep_file_input'], components['prep_audio_editor'],
                     components['transcription_output'], components['prep_audio_info'],
                     components['prep_audio_editor']]
        )

        # Preview on dropdown change
        components['existing_sample_dropdown'].change(
            load_existing_sample_handler,
            inputs=[components['existing_sample_dropdown']],
            outputs=[components['existing_sample_audio'], components['existing_sample_text'],
                     components['existing_sample_info']]
        )

        # Preview button
        components['preview_sample_btn'].click(
            load_existing_sample_handler,
            inputs=[components['existing_sample_dropdown']],
            outputs=[components['existing_sample_audio'], components['existing_sample_text'],
                     components['existing_sample_info']]
        )

        # Refresh button
        components['refresh_preview_btn'].click(
            refresh_samples_handler,
            outputs=[components['existing_sample_dropdown']]
        )

        # Delete sample modal
        components['delete_sample_btn'].click(
            fn=None,
            js=show_confirmation_modal_js(
                title="Delete Sample?",
                message="This will permanently delete the sample audio, metadata, and cached files. This action cannot be undone.",
                confirm_button_text="Delete",
                context="sample_"
            )
        )

        # Delete confirmation handler
        # Update both prep samples dropdown and main app dropdown (if exists)
        main_sample_dropdown = shared_state.get('sample_dropdown', components['existing_sample_dropdown'])
        confirm_trigger.change(
            delete_sample_handler,
            inputs=[confirm_trigger, components['existing_sample_dropdown']],
            outputs=[components['save_status'], components['existing_sample_dropdown'], main_sample_dropdown]
        )

        # Clear cache
        components['clear_cache_btn'].click(
            clear_sample_cache_handler,
            inputs=[components['existing_sample_dropdown']],
            outputs=[components['save_status'], components['existing_sample_info']]
        )

        # File input change - load and show editor
        components['prep_file_input'].change(
            on_prep_audio_load_handler,
            inputs=[components['prep_file_input']],
            outputs=[components['prep_audio_editor'], components['prep_audio_info']]
        ).then(
            lambda audio: (
                gr.update(visible=audio is not None),
                gr.update(visible=audio is None)
            ),
            inputs=[components['prep_audio_editor']],
            outputs=[components['prep_audio_editor'], components['prep_file_input']]
        )

        # Clear button
        components['clear_btn'].click(
            lambda: (None, None, ""),
            outputs=[components['prep_file_input'], components['prep_audio_editor'],
                     components['prep_audio_info']]
        )

        # Normalize
        components['normalize_btn'].click(
            normalize_audio_handler,
            inputs=[components['prep_audio_editor']],
            outputs=[components['prep_audio_editor']]
        )

        # Convert to mono
        components['mono_btn'].click(
            convert_to_mono_handler,
            inputs=[components['prep_audio_editor']],
            outputs=[components['prep_audio_editor']]
        )

        # Clean audio
        components['clean_btn'].click(
            clean_audio_handler,
            inputs=[components['prep_audio_editor']],
            outputs=[components['prep_audio_editor']]
        )

        # Transcribe
        components['transcribe_btn'].click(
            transcribe_audio_handler,
            inputs=[components['prep_audio_editor'], components['whisper_language'],
                    components['transcribe_model']],
            outputs=[components['transcription_output']]
        )

        # Save sample modal
        components['save_sample_btn'].click(
            fn=None,
            js=show_input_modal_js(
                title="Save Voice Sample",
                message="Enter a name for this voice sample:",
                placeholder="e.g., MyVoice, Female-Accent, John-Doe",
                context="save_sample_"
            )
        )

        # Save sample input handler
        # Update both prep samples dropdown and main app dropdown (if exists)
        main_sample_dropdown = shared_state.get('sample_dropdown', components['existing_sample_dropdown'])
        input_trigger.change(
            handle_save_sample_input,
            inputs=[input_trigger, components['prep_audio_editor'], components['transcription_output']],
            outputs=[components['save_status'], components['existing_sample_dropdown'], main_sample_dropdown]
        )

        # Save preferences
        components['transcribe_model'].change(
            lambda x: save_preference("transcribe_model", x),
            inputs=[components['transcribe_model']],
            outputs=[]
        )

        components['whisper_language'].change(
            lambda x: save_preference("whisper_language", x),
            inputs=[components['whisper_language']],
            outputs=[]
        )


# Export for tab registry
get_tool_class = lambda: PrepSamplesTool


if __name__ == "__main__":
    """Standalone testing of Prep Samples tool."""
    from modules.core_components.tools import run_tool_standalone
    run_tool_standalone(PrepSamplesTool, port=7865, title="Prep Samples - Standalone")
