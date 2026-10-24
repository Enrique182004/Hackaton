# SYSTEM DIRECTIVE: SAR Whiteboard Assistant (Hackathon Build)

## 1. Project Overview & Operational Context
You are tasked with building a complete Spatial Augmented Reality (SAR) Whiteboard Assistant. The system combines a physical whiteboard, a webcam, and a digital projector to create an embodied, animated 2D AI avatar that visually tracks, evaluates, and physically interacts with a user's dry-erase marker drawings in real-time. 

This is a local, self-hosted deployment. The core objective is to execute an asynchronous Multimodal Loop (Sense -> Think -> Act) driven by a local Retrieval-Augmented Generation (RAG) knowledge base.

## 2. Target Directory Structure
Maintain strict modularity. Generate code according to this architecture:
```text
/biobench_sar
│── app.py                  # Main Flask server and WebSocket socketio loop
│── /core
│   ├── homography.py       # OpenCV calibration and coordinate mapping
│   ├── vision_loop.py      # Background thread for webcam frame buffering
│   └── audio_trigger.py    # PyAudio/Whisper listening thread
│── /rag
│   ├── ingestor.py         # Multi-format file parsing (PDF, CSV, etc.)
│   ├── vector_store.py     # ChromaDB and SentenceTransformers logic
│── /llm
│   ├── vlm_client.py       # API/Local routing for VLM evaluation
│   └── tts_engine.py       # Text-to-Speech generation
│── /static
│   ├── css/style.css       # Pure black background, avatar CSS sprites
│   ├── js/main.js          # WebSocket client, DOM updates, avatar movement
│   └── assets/sprites/     # 2D character sprite sheets
└── /templates
    └── index.html          # Projector display UI
```

## 3. Technology Stack & Dependencies
*   **Backend Server:** Python 3.10+, Flask, Flask-SocketIO (for WebSockets).
*   **Computer Vision:** `opencv-python` (cv2), `numpy`.
*   **Audio & TTS:** `SpeechRecognition`, `pyaudio`, `edge-tts` (or pyttsx3), `openai-whisper` (local).
*   **RAG & Embeddings:** `chromadb`, `sentence-transformers` (`all-MiniLM-L6-v2`).
*   **Document Parsers:** `PyMuPDF` (PDF, TXT), `pandas` (CSV), `python-docx` (DOCX), `python-pptx` (PPTX), `pytesseract` (PNG, JPG).
*   **Frontend:** Vanilla HTML5, CSS3, JavaScript.

## 4. Hardware Calibration: The Homography Matrix
The webcam and projector have different optical lenses and physical offsets. 
*   **Requirement:** You must implement a `cv2.findHomography` pipeline.
*   **Workflow:** The projector displays a known checkerboard calibration grid. The webcam captures it. OpenCV calculates the transformation matrix. 
*   **Mapping:** Every physical coordinate (X, Y of the user's dry-erase ink) captured by the webcam must be passed through `cv2.warpPerspective` to calculate the exact display coordinates for the frontend projector UI.

## 5. Multiformat RAG Ingestion Pipeline
The Flask app must feature a `/precharge` dashboard for uploading study materials. 
*   **Text/PDF (`PyMuPDF`):** Extract raw text and chunk by paragraphs.
*   **CSV (`pandas`):** Convert rows into natural language chunks (e.g., "Row 1: Column A is X, Column B is Y").
*   **DOCX (`python-docx`):** Extract paragraph text sequentially.
*   **PPTX (`python-pptx`):** Iterate through slides and extract text shapes.
*   **Images (`pytesseract`):** Run Optical Character Recognition on PNG/JPG files to extract text. Note: Provide instructions for installing the Tesseract system binary.
*   **Vectorization:** Embed all chunks using `SentenceTransformers` and store them in a persistent local `ChromaDB` collection. 

## 6. The Multimodal Interaction Loop
The system must run non-blocking background threads to prevent the camera feed from freezing while the AI thinks.
1.  **SENSE:** A background thread listens for a hotkey or voice wake-word via PyAudio. When triggered, Whisper transcribes the voice query. Simultaneously, the vision thread grabs the latest OpenCV frame of the whiteboard and applies the homography matrix to isolate the writable area.
2.  **THINK:** 
    *   Query ChromaDB using the Whisper transcript to retrieve the top-k relevant document chunks.
    *   Package the text transcript, the whiteboard image, and the RAG context into a single prompt for the Vision-Language Model (Local LLaVA or API).
    *   **CRITICAL:** The VLM system prompt must strictly enforce this JSON output schema:
        ```json
        {
          "spoken_text": "The explanation or feedback for the user.",
          "animation_state": "idle | walking | pointing | celebrating | error",
          "target_x": 850,
          "target_y": 420
        }
        ```
3.  **ACT:** The backend parses the JSON. It passes `spoken_text` to the TTS engine for asynchronous audio playback. Simultaneously, it emits a WebSocket event containing `animation_state`, `target_x`, and `target_y` to the frontend.

## 7. Holographic Frontend UI & Avatar CSS
*   **The Canvas:** The `body` background MUST be `#000000` (pure black) with `overflow: hidden`. Projectors cannot emit black light; this renders the background invisible, making the UI elements look like holograms on the physical whiteboard.
*   **The Avatar:** Implemented as a CSS sprite (`div` with a background image). 
*   **Movement:** Use `position: absolute`. When WebSockets receive new coordinates, update the `left` and `top` style properties via JavaScript. Use CSS `transition: all 1.5s ease-in-out` to make the avatar glide smoothly to the target ink. Swap the background-position/class based on `animation_state`.

## 8. Hackathon Target Scenario: Environmental Biology
Configure the default prompts and test cases to support a thesis on identifying *Acanthamoeba* in Laguna de Bustillos, integrating the hydrological dynamics and soil runoff of Leptosol and Feozem edaphological classifications in Cuauhtémoc, Chihuahua. The avatar must act as an expert research assistant capable of verifying drawn flowcharts and sterile lab procedures based on these specific parameters.

## 9. Strict AI Code Generation Directives
1.  **No Monoliths:** Never attempt to generate the entire application in a single script. Wait for the user to specify which module (e.g., `vision_loop.py` or `ingestor.py`) to build.
2.  **Concurrency First:** Ensure `asyncio`, `threading`, or `multiprocessing` are used appropriately. The camera loop must run at 30+ FPS regardless of what the LLM is doing.
3.  **Setup Scripts:** Provide explicit Windows PowerShell commands to install all pip dependencies and system binaries before writing Python code. 
4.  **Error Handling:** Include robust `try/except` blocks for the JSON parsing step, as VLMs can occasionally hallucinate markdown formatting around JSON outputs.