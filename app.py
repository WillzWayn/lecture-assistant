import gradio as gr
from enum import StrEnum
from pathlib import Path
import hashlib
import openai
from gradio.themes import Soft
import logging
import time
from abc import ABC, abstractmethod


logging.basicConfig(level=logging.INFO)

try:
    import mlx_whisper
    MLX = True
except ImportError:
    raise ImportError("Please install the 'mlx-whisper' package to use this module.")

# Attempt to import Hugging Face transformers
try:
    import torch
    from transformers import AutoModelForCausalLM, AutoTokenizer
    from transformers import pipeline
    HF_AVAILABLE = True
except ImportError:
    HF_AVAILABLE = False

# -------------------------
# Abstract Interfaces
# -------------------------
class TranscriptionService(ABC):
    @abstractmethod
    def transcribe(self, audio_file: str) -> str:
        pass

class SummarizationService(ABC):
    @abstractmethod
    def summarize(self, prompt: str) -> str:
        pass

# -------------------------
# Concrete Adapters
# -------------------------
class MLXWhisperAdapter(TranscriptionService):
    def __init__(self, model_name: str):
        self.model_name = model_name

    def transcribe(self, audio_file: str) -> str:
        logging.debug("Transcribing with MLX Whisper: %s", self.model_name)
        result = mlx_whisper.transcribe(audio_file, path_or_hf_repo=self.model_name)
        text = result.get("text", "")
        if isinstance(text, list):
            text = " ".join(str(chunk) for chunk in text)
        return str(text)

class HuggingFaceWhisperAdapter(TranscriptionService):
    def __init__(self, model_name: str):
        if not HF_AVAILABLE:
            raise ImportError("Hugging Face transformers not available.")
        self.model_name = model_name
        self.pipe = pipeline("automatic-speech-recognition", model=model_name, device=0 if torch.cuda.is_available() else -1)

    def transcribe(self, audio_file: str) -> str:
        logging.debug("Transcribing with Hugging Face Whisper: %s", self.model_name)
        result = self.pipe(audio_file)
        return result["text"]

class CloudWhisperAdapter(TranscriptionService):
    def __init__(self, api_key: str, model_name: str = "whisper-1"):
        self.api_key = api_key
        self.model_name = model_name
        self.client = openai.OpenAI(api_key=self.api_key)

    def transcribe(self, audio_file: str) -> str:
        logging.debug("Transcribing with Cloud Whisper: %s", self.model_name)
        with open(audio_file, "rb") as audio:
            transcript = self.client.audio.transcriptions.create(
                model=self.model_name,
                file=audio
            )
        return transcript.text

class LocalOpenAIAdapter(SummarizationService):
    def __init__(self, base_url: str, api_key: str, model_name: str):
        self.client = openai.OpenAI(base_url=base_url, api_key=api_key)
        self.model_name = model_name

    def summarize(self, prompt: str) -> str:
        logging.debug("Summarizing with Local OpenAI: %s", self.model_name)
        response = self.client.chat.completions.create(
            model=self.model_name,
            messages=[{"role": "user", "content": prompt}]
        )
        return response.choices[0].message.content or "No summary generated."

class HuggingFaceLLMAdapter(SummarizationService):
    def __init__(self, model_name: str):
        if not HF_AVAILABLE:
            raise ImportError("Hugging Face transformers not available.")
        
        self.model_name = model_name
        device = 0 if torch.cuda.is_available() else -1

        # Create the pipeline once — faster and cleaner
        self.pipe = pipeline(
            "text-generation",
            model=model_name,
            device=device,
            torch_dtype=torch.float16 if torch.cuda.is_available() else torch.float32,
        )

    def summarize(self, prompt: str) -> str:
        logging.debug("Summarizing with Hugging Face pipeline: %s", self.model_name)
        
        # Generate with basic parameters; tune max_new_tokens or temperature as needed
        output = self.pipe(
            prompt,
            max_new_tokens=1500,
            do_sample=True,
            temperature=0.7,
            top_p=0.9,
            truncation=True,
        )

        return output[0]["generated_text"]

class GROQAdapter(SummarizationService):
    def __init__(self, api_key: str, model_name: str):
        self.client = openai.OpenAI(
            base_url="https://api.groq.com/openai/v1",
            api_key=api_key
        )
        self.model_name = model_name

    def summarize(self, prompt: str) -> str:
        logging.debug("Summarizing with GROQ: %s", self.model_name)
        response = self.client.chat.completions.create(
            model=self.model_name,
            messages=[{"role": "user", "content": prompt}]
        )
        return response.choices[0].message.content or "No summary generated."

class LocalLLMAdapter(SummarizationService):
    def __init__(self, base_url: str, api_key: str, model_name: str):
        self.client = openai.OpenAI(base_url=base_url, api_key=api_key)
        self.model_name = model_name

    def summarize(self, prompt: str) -> str:
        logging.debug("Summarizing with Local LLM: %s", self.model_name)
        response = self.client.chat.completions.create(
            model=self.model_name,
            messages=[{"role": "user", "content": prompt}]
        )
        return response.choices[0].message.content or "No summary generated."

# -------------------------
# Logging
# -------------------------
logging.basicConfig(level=logging.DEBUG, format="%(asctime)s [%(levelname)s] %(message)s")
logging.info("Initializing Audio Transcription & Summarization application")

# -------------------------
# OpenAI client (local)
# -------------------------
client = openai.OpenAI(
    base_url="http://localhost:1234/v1",
    api_key="dummy"
)

# -------------------------
# Cache directory
# -------------------------
CACHE_DIR = Path(__file__).resolve().parent / "cache"
CACHE_DIR.mkdir(parents=True, exist_ok=True)

# -------------------------
# Utility Functions
# -------------------------
def sha256_of_file(file_path: str, block_size: int = 65536) -> str:
    """Compute SHA256 hash of a file."""
    logging.debug("Computing SHA256 for file: %s", file_path)
    p = Path(file_path)
    if not p.is_file():
        raise FileNotFoundError(f"File not found: {file_path}")
    h = hashlib.sha256()
    with p.open("rb") as f:
        for chunk in iter(lambda: f.read(block_size), b""):
            h.update(chunk)
    return h.hexdigest()

def get_cached_transcription(audio_file: str, force_refresh=False) -> str | None:
    """Return cached transcription if available and not forcing refresh."""
    logging.debug("Checking cache for file: %s", audio_file)
    digest = sha256_of_file(audio_file)
    cache_path = CACHE_DIR / f"{digest}.txt"
    logging.debug("Cache path: %s", cache_path)
    if cache_path.exists() and not force_refresh:
        logging.debug("Cache exists, attempting to read")
        try:
            content = cache_path.read_text(encoding="utf-8")
            logging.info(f"Loaded transcription {audio_file} from cache {digest}")
            return content
        except Exception as e:
            logging.warning(f"Unable to read cache: {e}")
    logging.debug("No valid cache found")
    return None

# -------------------------
# Enums
# -------------------------
class WhisperModelChoice(StrEnum):
    WHISPER_MLX_SMALL = "mlx-community/whisper-tiny"
    WHISPER_MLX_LARGE = "mlx-community/whisper-large-v3-mlx"
    WHISPER_HF_BASE = "openai/whisper-base"
    WHISPER_CLOUD = "cloud/whisper-1"

class HuggingFaceLLMModelDefaultChoices(StrEnum):
    GRANITE_4_H_TINY = "ibm/granite-4-h-tiny"
    QWEN3_4B_2507 = "qwen/qwen3-4b-2507"
    GEMMA_3N_E4B = "google/gemma-3n-e4b"
    BAGUETTOTRON = 'PleIAs/Baguettotron'

class BackendChoice(StrEnum):
    LOCAL_OPENAI = "Local OpenAI"
    HUGGING_FACE = "Hugging Face"
    GROQ = "GROQ"
    LOCAL_LLM = "Local LLM"



class PromptChoice(StrEnum):
    SUMMARY = "Summary"
    SUMMARY_BULLETS = "Summary with bullet points"
    FLASHCARDS = "Flashcards"

PROMPT_MAPPING: dict[str, str] = {
    PromptChoice.SUMMARY: """
You are an expert teaching assistant.

Here is a class transcription:
<START TRANSCRIPTION>
{transcription}
<END TRANSCRIPTION>

Step 1: Summarize the key points clearly and concisely.  
Step 2: Review your summary and improve it by making it more structured and easy to understand, without adding new information.  
Step 3: Output the refined summary. 

### Refined Summary

**1. Topic Overview**  
Concise statement of the main theme.

**2. Key Concepts Covered**  
- Bullet point summary of the main ideas and subtopics.

**3. Instructor’s Emphasis / Takeaways**  
- Highlight any repeated or stressed points.

**4. Contextual Notes (if relevant)**  
- Any dependencies, references, or frameworks mentioned.
""",
    PromptChoice.SUMMARY_BULLETS: "Provide a summary with bullet points for this info: {transcription}",
    PromptChoice.FLASHCARDS: """You are an expert teaching assistant. Here is a class transcription:

{transcription}

Step 1: Create flashcards with clear questions and answers based on this class.  
Step 2: Review each flashcard and improve the wording for clarity and accuracy, without adding new content.  
Step 3: Output the refined set of flashcards.
""",
    "Custom": "{transcription}"
}

# -------------------------
# Transcription
# -------------------------
def get_transcription_service(model: str) -> TranscriptionService:
    if model.startswith("mlx-community/"):
        return MLXWhisperAdapter(model)
    elif model.startswith("openai/"):
        return HuggingFaceWhisperAdapter(model)
    elif model.startswith("cloud/"):
        # Assume API key is set in environment or config
        import os
        api_key = os.getenv("OPENAI_API_KEY", "")
        return CloudWhisperAdapter(api_key, model.split("/")[-1])
    else:
        raise ValueError(f"Unsupported transcription model: {model}")

def transcribe_file(audio_file: str, model: str) -> str:
    """Transcribe audio using selected Whisper model."""
    logging.debug("Starting transcription with model: %s for file: %s", model, audio_file)
    start = time.time()
    try:
        service = get_transcription_service(model)
        text = service.transcribe(audio_file)
        logging.info(f"Transcription of file {audio_file} completed in {time.time() - start:.2f}s")
        return str(text)
    except Exception as e:
        logging.error("Error during transcription: %s", e)
        return f"Error during transcription: {e}"
# -------------------------
# Prompt / Metadata
# -------------------------
def format_file_metadata(audio_file: str, digest: str | None = None) -> str:
    logging.debug("Formatting metadata for file: %s", audio_file)
    p = Path(audio_file)
    if not p.exists():
        return ""
    size_kb = p.stat().st_size / 1024
    if digest is None:
        try:
            digest = sha256_of_file(audio_file)
        except FileNotFoundError:
            digest = "Unavailable"
    return (
        f"**File:** {p.name}  \n"
        f"**Size:** {size_kb:.1f} KB  \n"
        f"**SHA256:** `{digest}`"
    )

def fetch_models(base_url: str, api_key: str = "") -> list[str]:
    """Fetch available models from OpenAI-compatible server."""
    try:
        import requests
        headers = {}
        if api_key:
            headers["Authorization"] = f"Bearer {api_key}"
        response = requests.get(f"{base_url}/models", headers=headers, timeout=5)
        if response.status_code == 200:
            data = response.json()
            models = [model["id"] for model in data.get("data", [])]
            return models
        else:
            logging.warning(f"Failed to fetch models: {response.status_code}")
            return []
    except Exception as e:
        logging.error(f"Error fetching models: {e}")
        return []

def update_backend_settings(backend):
    if backend == BackendChoice.GROQ:
        return gr.update(visible=True, label="GROQ API Key"), gr.update(visible=False, value="https://api.groq.com/openai/v1"), gr.update(visible=True), gr.update(visible=True, choices=[])
    elif backend in [BackendChoice.LOCAL_OPENAI, BackendChoice.LOCAL_LLM]:
        default_url = "http://localhost:1234/v1" if backend == BackendChoice.LOCAL_OPENAI else "http://localhost:8000/v1"
        return gr.update(visible=False), gr.update(visible=True, value=default_url), gr.update(visible=True), gr.update(visible=True, choices=[])
    else:  # Hugging Face
        return gr.update(visible=False), gr.update(visible=False), gr.update(visible=False), gr.update(visible=True, choices=[HuggingFaceLLMModelDefaultChoices.GRANITE_4_H_TINY.value, HuggingFaceLLMModelDefaultChoices.QWEN3_4B_2507.value, HuggingFaceLLMModelDefaultChoices.GEMMA_3N_E4B.value])

def fetch_and_update_models(base_url, api_key):
    models = fetch_models(base_url, api_key)
    if models:
        return gr.update(choices=models, value=models[0] if models else None)
    else:
        return gr.update(choices=[], value=None)

def update_prompt(prompt_type: str):
    logging.debug("Updating prompt for type: %s", prompt_type)
    return gr.update(
        value=PROMPT_MAPPING.get(prompt_type, PROMPT_MAPPING[PromptChoice.SUMMARY.value]),
        interactive=(prompt_type == "Custom")
    )

def get_summarization_service(backend: str, llm_model_choice: str, base_url: str = "", api_key: str = "") -> SummarizationService:
    if backend == BackendChoice.HUGGING_FACE:
        if not HF_AVAILABLE:
            raise ImportError("Hugging Face not available. Please install transformers and torch.")
        model_mapping = {
            "ibm/granite-4-h-tiny": "ibm-granite/granite-4.0-h-tiny",
            "qwen/qwen3-4b-2507": "Qwen/Qwen3-4B",
            "google/gemma-3n-e4b": "google/gemma-3-4B",
            "PleIAs/Baguettotron": "PleIAs/Baguettotron"
        }
        model_path = model_mapping.get(llm_model_choice)
        if not model_path:
            raise ValueError(f"Model {llm_model_choice} not supported for Hugging Face.")
        return HuggingFaceLLMAdapter(model_path)
    elif backend == BackendChoice.GROQ:
        if not api_key:
            import os
            api_key = os.getenv("GROQ_API_KEY", "")
        return GROQAdapter(api_key, llm_model_choice)
    elif backend == BackendChoice.LOCAL_LLM:
        if not base_url:
            base_url = "http://localhost:8000/v1"
        return LocalLLMAdapter(base_url, api_key or "dummy", llm_model_choice)
    else:  # Local OpenAI
        if not base_url:
            base_url = "http://localhost:1234/v1"
        return LocalOpenAIAdapter(base_url, api_key or "dummy", llm_model_choice)

def safe_generate_summary(prompt: str, llm_model_choice: str, backend: str, base_url: str = "", api_key: str = "") -> str:
    """Generate summary using selected backend with safe error handling."""
    logging.debug("Generating summary with backend: %s, model: %s", backend, llm_model_choice)
    try:
        service = get_summarization_service(backend, llm_model_choice, base_url, api_key)
        return service.summarize(prompt)
    except Exception as e:
        logging.error("Error generating summary: %s", e)
        return f"Error generating summary: {e}"

# -------------------------
# Main workflow
# -------------------------
def transcribe_with_custom(audio, llm_model_choice, speech_to_text_model, prompt_template, backend, base_url, api_key):
    logging.info("Starting transcription and summarization process with backend: %s", backend)
    if not audio:
        logging.warning("No audio file provided")
        return "No audio file provided.", "", ""
    try:
        digest = sha256_of_file(audio)
        logging.debug("Computed digest: %s", digest)
    except FileNotFoundError:
        logging.error("Audio file could not be found for hashing: %s", audio)
        return "Audio file could not be found for hashing.", "", ""

    logging.debug("Attempting to get cached transcription")
    transcription = get_cached_transcription(audio) or transcribe_file(audio, speech_to_text_model)

    if not transcription.startswith("Error:") and transcription.strip():
        logging.debug("Writing transcription to cache")
        try:
            cache_path = CACHE_DIR / f"{digest}.txt"
            cache_path.write_text(transcription, encoding="utf-8")
        except Exception as e:
            logging.warning(f"Failed to write cache: {e}")

    file_meta = format_file_metadata(audio, digest)

    if transcription.startswith("Error:") or not transcription.strip():
        logging.warning("Transcription failed or empty, returning without summary")
        return transcription, "", file_meta

    logging.info("Preparing prompt for summarization")
    if not prompt_template.strip():
        prompt_template = PROMPT_MAPPING[PromptChoice.SUMMARY.value]
    elif "{transcription}" not in prompt_template:
        prompt_template += " {transcription}"

    prompt = prompt_template.replace("{transcription}", transcription)
    logging.debug("Prompt prepared")
    summary = safe_generate_summary(prompt, llm_model_choice, backend, base_url, api_key)

    logging.info("Transcription and summarization process completed")
    return summary, transcription, file_meta

# -------------------------
# Gradio UI
# -------------------------
CUSTOM_CSS = """
/* ---------- File Metadata Box ---------- */



/* ---------- Text Areas ---------- */
#summary-output textarea,
#transcription-output textarea {
    font-size: 1rem;
    line-height: 1.6;
    background-color: #fefefe;
    color: #111827;
    border: 1px solid #d1d5db;
    border-radius: 8px;
    padding: 10px;
    resize: vertical;
    transition: border-color 0.2s ease, box-shadow 0.2s ease;
}

#summary-output textarea:focus,
#transcription-output textarea:focus {
    border-color: #2563eb;
    box-shadow: 0 0 0 3px rgba(37, 99, 235, 0.2);
    outline: none;
}

/* ---------- Buttons ---------- */
.gr-button-primary {
    background: linear-gradient(135deg, #2563eb, #1d4ed8);
    border: none;
    color: #ffffff;
    font-weight: 600;
    box-shadow: 0 4px 12px rgba(37, 99, 235, 0.25);
    border-radius: 8px;
    padding: 10px 18px;
    transition: transform 0.12s ease, box-shadow 0.12s ease, background 0.2s ease;
}

.gr-button-primary:hover {
    transform: translateY(-2px);
    box-shadow: 0 6px 16px rgba(37, 99, 235, 0.35);
    background: linear-gradient(135deg, #1e40af, #1d4ed8);
}

/* ---------- Body & Labels ---------- */
body {
    background-color: #f3f4f6;
    color: #1f2937;
    font-family: "Inter", sans-serif;
}

label {
    color: #111827 !important;
    font-weight: 500;
}

/* ---------- Accordions ---------- */
.gr-accordion {
    border-radius: 8px;
    border: 1px solid #d1d5db;
    background: #ffffff;
    padding: 10px 14px;
}

.gr-accordion .gr-accordion-title {
    font-weight: 600;
    color: #1d4ed8;
}

/* ---------- Misc ---------- */
.gr-row, .gr-column {
    gap: 12px;
}

"""

with gr.Blocks(theme=Soft(), css=CUSTOM_CSS) as demo:
    gr.Markdown("# Audio Transcription & Summarization")
    gr.Markdown("Upload an audio file, choose a Whisper model, and generate a tailored summary using the local Granite model.")

    with gr.Row():
        with gr.Column(scale=3):
            audio_input = gr.Audio(type="filepath", label="Upload Audio File")
            file_details = gr.Markdown("", elem_id="file-meta")
            with gr.Accordion("Full Transcription", open=False):
                transcription_output = gr.Textbox(label="Transcription", lines=10, show_copy_button=True, elem_id="transcription-output")

        with gr.Column(scale=2):
            backend_dropdown = gr.Dropdown(
                choices=[b.value for b in BackendChoice],
                label="LLM Backend",
                value=BackendChoice.LOCAL_OPENAI
            )
            api_key_input = gr.Textbox(label="API Key", visible=False, type="password")
            base_url_input = gr.Textbox(label="Base URL", visible=True, placeholder="http://localhost:1234/v1")
            fetch_models_btn = gr.Button("Fetch Available Models", visible=False)
            llm_model_dropdown = gr.Dropdown(
                choices=[],
                label="Language Model",
                value=None
            )
            speech_to_text_dropdown = gr.Dropdown(
                choices=[m.value for m in WhisperModelChoice],
                label="Speech-To-Text Model",
                value=WhisperModelChoice.WHISPER_MLX_SMALL
            )
            prompt_dropdown = gr.Dropdown(
                choices=[p.value for p in PromptChoice] + ["Custom"],
                label="Prompt Type",
                value=PromptChoice.SUMMARY
            )
            with gr.Accordion("Prompt Template", open=False):
                prompt_textbox = gr.Textbox(
                    label="Template",
                    value=PROMPT_MAPPING[PromptChoice.SUMMARY],
                    interactive=False,
                    lines=6,
                    show_copy_button=True
                )
            with gr.Row():
                submit_btn = gr.Button("Transcribe & Summarize", variant="primary")
                clear_btn = gr.ClearButton(
                    [audio_input, transcription_output, prompt_textbox, llm_model_dropdown, speech_to_text_dropdown, prompt_dropdown, backend_dropdown, api_key_input, base_url_input, fetch_models_btn, file_details],
                    value="Clear"
                )

    summary_output = gr.Textbox(label="Summary", lines=8, show_copy_button=True, elem_id="summary-output")

    prompt_dropdown.change(update_prompt, inputs=prompt_dropdown, outputs=prompt_textbox)
    backend_dropdown.change(update_backend_settings, inputs=backend_dropdown, outputs=[api_key_input, base_url_input, fetch_models_btn, llm_model_dropdown])
    fetch_models_btn.click(fetch_and_update_models, inputs=[base_url_input, api_key_input], outputs=llm_model_dropdown)
    submit_btn.click(
        transcribe_with_custom,
        inputs=[audio_input, llm_model_dropdown, speech_to_text_dropdown, prompt_textbox, backend_dropdown, base_url_input, api_key_input],
        outputs=[summary_output, transcription_output, file_details],
        queue=True
    )

logging.info("Launching Gradio interface")
demo.launch()
