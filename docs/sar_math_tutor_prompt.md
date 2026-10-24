# SYSTEM DIRECTIVE: SAR Whiteboard Math Assistant (Hackathon Build)

> Source of truth for the **math** subject profile (`SAR_PROFILE=math`, the default). The original
> Environmental Biology directive, [sar_whiteboard_assistant_prompt.md](sar_whiteboard_assistant_prompt.md),
> is kept as the `biology` profile and still describes the module layout and RAG ingestion formats.
> How the spec maps onto code: [BUILD_PLAN.md](BUILD_PLAN.md) (contracts G–I, the math semantics of each
> mode and the math scenario test cases).

## 1. Project Overview & Operational Context

You are tasked with building a Spatial Augmented Reality (SAR) Whiteboard Assistant designed to tutor middle
and high school students in mathematics (e.g., fractions, equations, algebra). The system combines a physical
whiteboard, a webcam, and a digital projector to create an embodied, animated 2D AI avatar that visually
tracks, evaluates, and physically interacts with a user's dry-erase marker drawings in real-time.

This is a local, self-hosted deployment. The core objective is to execute an asynchronous Multimodal Loop
(Sense -> Think -> Act) driven by voice interactions and computer vision.

## 2. Hardware Setup & Calibration

- **Camera:** Standard webcam mounted on a tripod facing the whiteboard.
- **Display:** Digital projector mounted adjacent to the webcam, pointing at the exact same surface.
- **Homography Calibration:** The webcam and projector have different physical offsets. You must implement a
  `cv2.findHomography` pipeline. A projected calibration grid must be captured by the webcam to map the
  camera's spatial coordinates exactly to the projector's display plane.

## 3. Technology Stack

- **Backend:** Python 3, Flask, Flask-SocketIO (for WebSockets), `opencv-python`, `numpy`.
- **Audio & TTS:** `pyaudio`, `edge-tts` (or local fallback), `openai-whisper` (local).
- **RAG (Knowledge Base):** `chromadb`, `sentence-transformers`, `PyMuPDF`, `pandas`, `python-docx` for
  ingesting math worksheets and quiz banks.
- **Frontend:** Vanilla HTML5, CSS3, JavaScript.

## 4. The Multimodal Interaction Loop

The system runs non-blocking background threads to ensure the camera feed never freezes.

1. **SENSE:** A background thread listens via PyAudio. Whisper transcribes the student's voice command.
   OpenCV captures the current high-resolution frame of the physical whiteboard.
2. **THINK:** The text transcript and the whiteboard image are fed into a Vision-Language Model (VLM). The
   model evaluates the physical math equations and generates a structured JSON response.
3. **ACT:** Flask routes the JSON's text output to the TTS engine for audio playback, while pushing X/Y
   coordinates and animation states via WebSockets to the CSS sprite frontend.

## 5. Math Tutor Logic & Strict TTS Formatting

When generating the VLM prompt and backend logic, enforce these core tutoring behaviors:

### A. Phonetic TTS Formatting (CRITICAL)

- The VLM must write all math phonetically in its output so the voice engine reads it naturally.
- NEVER use symbols like `^`, `/`, `*`, or `=` in the spoken output.
- Write "two thirds" (not 2/3), "x squared" (not x^2), and "three x plus five equals twelve" (not 3x+5=12).

### B. Pedagogical Modes

- **Step-by-Step Solving:** If the user asks for help, the AI must NEVER give the final answer. It must
  verbally explain only the next operation using simple vocabulary, project a highlight on the relevant part
  of the equation, and wait for the student to write the next step.
- **Generate Similar Problems:** If asked for more practice, the AI must read the current equation on the
  board, swap variables/constants to maintain the same difficulty, and verbally dictate the new equation for
  the student to write.
- **Quiz Mode:** The AI queries the local ChromaDB for progressive math problems from uploaded worksheets,
  dictates the problem, and evaluates the student's physical ink answer.

  > **How this is built:** the problems are not searched in ChromaDB. When a CSV quiz bank or a worksheet is
  > uploaded, its problems and answers are parsed into the quiz bank (`rag/quiz_bank.py`, JSON at
  > `SAR_QUIZ_BANK_FILE`); a quiz takes them from the ticked files, easiest level first, at most
  > `SAR_QUIZ_LENGTH` (default 10) per quiz, and when no worksheet problems are ticked the code makes up
  > progressive problems (the model only reads them out). ChromaDB holds the files' text for answering
  > questions; answer keys are removed from every file when it is added, so the model never sees them.
  > Answers are **graded in code** (`grading.py`, with sympy): the model only reads what the student wrote,
  > and the code compares it with the answer key (equivalent forms, several answers and systems count).
  > Asking for help during a quiz ("I'm stuck", "hint") gives one hint on the same problem without ending
  > the quiz, and saying "stop the quiz" asks for a yes before it ends. Details:
  > [BUILD_PLAN.md](BUILD_PLAN.md) contracts C, G and H.

### C. Learning a Topic (works with an empty board)

A student does not need anything on the board to start. They can say, for example, "I want to learn how to
solve a fraction":

1. The AI explains the idea of the topic briefly, in plain words for a middle or high school student.
2. It asks whether the student has an exercise in mind or whether it should make one up.
3. It follows up on the answer: it dictates an easy starter problem and invites the student to try the first
   step, or it asks the student to write their own exercise on the board and then ask for help.

The AI remembers the last few turns of the conversation, so follow-ups such as "make one up", "I have my
own" or "I did it" make sense without repeating the question.

## 6. Required JSON Output Schema

The VLM must be rigidly prompted to return ONLY this JSON structure to prevent application crashes:

```json
{
  "spoken_text": "Phonetically spelled, conversational response (e.g., 'Let us move the two x to the other side').",
  "animation_state": "idle | walking | pointing | celebrating | error",
  "target_x": 850,
  "target_y": 420
}
```

The schema is the same in every mode and for both profiles: a mode only changes what goes into `spoken_text`
and which `animation_state` is chosen (for example `pointing` at the next step in help mode, `celebrating`
for a correct quiz answer, `error` with one hint for a wrong one). `target_x` / `target_y` are projector
pixels.
