
# Lecture Assistant

A Python application for transcribing audio lectures, generating summaries, and creating study aids such as flashcards. Supports multiple transcription backends (MLX Whisper, Hugging Face, OpenAI) and LLM summarization backends (Local OpenAI, Hugging Face, GROQ).

---

## Features

- Transcribe audio files using:
  - MLX Whisper (local models)
  - Hugging Face Whisper models
  - OpenAI Whisper API
- Generate summaries with different prompt styles:
  - Concise summaries
  - Bullet-point summaries
  - Flashcards
- Supports multiple LLM backends:
  - Local OpenAI
  - Hugging Face
  - GROQ
  - Local LLM instances
- Caching of transcriptions for faster processing
- Clean and responsive UI built with Gradio

---

## Installation

```bash
git clone https://github.com/your-username/lecture-assistant.git
cd lecture-assistant
python -m venv venv
source venv/bin/activate   # On Windows: venv\Scripts\activate
pip install -r requirements.txt
````

**Dependencies** (in `requirements.txt`):

```
gradio
openai
torch
transformers
mlx-whisper
```

Optional for Hugging Face LLMs:

```
accelerate
```

---

## Usage

1. Run the app:

```bash
python app.py
```

2. Open the Gradio interface in your browser.

3. Upload an audio file of a lecture or class.

4. Choose:

   * Speech-to-text model (Whisper)
   * LLM backend
   * Language model
   * Prompt type (summary, bullets, flashcards, custom)

5. Click **Transcribe & Summarize** to get your output.

6. Cached transcriptions are automatically stored in the `cache/` folder for faster reprocessing.

---

## Project Structure

```
lecture-assistant/
│
├─ app.py                # Main application
├─ cache/                # Cached transcriptions
├─ requirements.txt      # Python dependencies
├─ README.md             # This file
```

---

## Notes

* For local LLMs, ensure the server is running at the specified `base_url`.
* Set environment variables for API keys if using Cloud Whisper or GROQ.
* Supports both GPU and CPU modes for faster processing when available.

---

## License

MIT License
