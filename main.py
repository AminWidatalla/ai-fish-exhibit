import cv2
import os
import numpy as np
import time
import socket
import threading
import qrcode
import base64
import sys
from pathlib import Path
from flask import Flask, render_template_string
from flask_socketio import SocketIO

import pyttsx3
from groq import Groq
from dotenv import load_dotenv
load_dotenv()

# ----------------------------
# 1. SETTINGS & CONFIGURATION
# ----------------------------
W, H = 1280, 720
VIDEO_H = H

BASE_DIR = Path(getattr(sys, 'frozen', False) and getattr(sys, '_MEIPASS', Path.cwd()) or Path(__file__).resolve().parent)
IMAGE_FOLDER = BASE_DIR / "images"
STREAM_FPS = 15


GROQ_API_KEY = os.environ.get("GROQ_API_KEY")
# CONVERSATION STATES
STATE_IDLE     = "idle"
STATE_THINKING = "thinking"
STATE_TALKING  = "talking"

current_state        = STATE_IDLE
last_interaction_time = 0       # Used to decide when to show/hide the QR code
interaction_text     = ""
conversation_history   = []       # List of (question, answer) tuples for context
MAX_HISTORY_MESSAGES = 6       # Keep the last 3 Q&A pairs

# ---------------------------------------------------------------------------
# VOICE / VIDEO SYNC — core idea
# ---------------------------------------------------------------------------
# `last_word_time` is updated every time pyttsx3 fires a 'started-word' event.
# The display loop uses it to decide whether speech is "actively happening"
# right now.  A grace window of SPEECH_PAUSE_GRACE seconds is applied so that
# brief natural pauses between sentences (commas, full-stops) do NOT freeze the
# video — only real gaps longer than the grace window pause the video frame.
# The video NEVER goes back to the static image while STATE_TALKING is active.
# It only returns to the static image once ask_ai_and_speak() sets STATE_IDLE.
# ---------------------------------------------------------------------------
SPEECH_PAUSE_GRACE = 0.6        # seconds — tune this to your TTS rhythm

last_word_time      = 0.0       # timestamp of the most recent 'started-word' event
speech_lock         = threading.Lock()

# The display loop saves the last rendered video frame here so it can freeze on
# it during a pause instead of calling get_video_frame() (which would advance).
current_video_frame: np.ndarray | None = None


def is_speech_active() -> bool:
    """Return True if a word was spoken recently enough to consider TTS active."""
    with speech_lock:
        return (time.time() - last_word_time) < SPEECH_PAUSE_GRACE


# ----------------------------
# 2. LOAD VIDEO ASSETS
# ----------------------------
TATTOO_IMAGE_PATH = IMAGE_FOLDER / "Tattoo.png"
IDLE_VIDEO_PATH   = IMAGE_FOLDER / "fish iddle.mov"

cap_idle = cv2.VideoCapture(str(IDLE_VIDEO_PATH))


def _make_fallback_frame(text: str) -> np.ndarray:
    """Create a dark fallback frame with centred text."""
    frame = np.zeros((H, W, 3), dtype=np.uint8)
    frame[:] = (20, 20, 20)
    cv2.putText(frame, text, (W // 2 - 260, H // 2),
                cv2.FONT_HERSHEY_SIMPLEX, 1.2, (255, 255, 255), 3)
    return frame


# Load the static image shown when the fish is idle / thinking
if TATTOO_IMAGE_PATH.exists():
    tattoo_image = cv2.imread(str(TATTOO_IMAGE_PATH))
    if tattoo_image is not None:
        tattoo_image = cv2.resize(tattoo_image, (W, H))
    else:
        tattoo_image = _make_fallback_frame("TATTOO IMAGE NOT FOUND")
else:
    tattoo_image = _make_fallback_frame("TATTOO IMAGE NOT FOUND")


def get_video_frame(cap: cv2.VideoCapture) -> np.ndarray:
    """
    Read the next frame from *cap*, looping back to the start on exhaustion.
    Always returns a frame sized to (W, H).
    """
    ret, frame = cap.read()
    if not ret:
        cap.set(cv2.CAP_PROP_POS_FRAMES, 0)
        ret, frame = cap.read()

    if ret:
        return cv2.resize(frame, (W, H))

    # Video file missing / unreadable
    fallback = np.zeros((H, W, 3), dtype=np.uint8)
    fallback[:] = (50, 20, 20)
    cv2.putText(fallback, "VIDEO NOT FOUND", (W // 2 - 200, H // 2),
                cv2.FONT_HERSHEY_SIMPLEX, 1.5, (255, 255, 255), 3)
    return fallback


# ----------------------------
# 3. AI & VOICE THREAD ENGINE
# ----------------------------

def ask_ai_and_speak(question: str) -> None:
    """
    1. Call Groq for an answer.
    2. Speak the answer with pyttsx3.
    3. Keep STATE_TALKING alive for at least the estimated speech duration so
       the display loop never flips back to the static image mid-sentence.
    4. Only set STATE_IDLE once everything is truly finished.
    """
    global current_state, interaction_text, last_word_time

    current_state    = STATE_THINKING
    interaction_text = "..."

    # ---- 3a. Fetch AI answer ------------------------------------------------
    try:
        client = Groq(api_key=os.environ.get("GROQ_API_KEY"))

        messages = [
            {
                "role": "system",
                "content": (
                    "You are a magical, ancient fish in a bowl. "
                    "A human woman is standing beside you. "
                    "The user talking to you will expect the human to answer, "
                    "but YOU (the fish) are the one talking. You must surprise them. "
                    "Keep answers under 3 short sentences. "
                    "NEVER say you are an AI, a language model, or from OpenAI. "
                    "If asked 'who are you', say you are the ancient fish of this museum."
                ),
            },
        ]
        messages.extend(conversation_history)
        messages.append({"role": "user", "content": question})

        response = client.chat.completions.create(
            model="llama-3.3-70b-versatile",
            messages=messages,
        )
        answer = response.choices[0].message.content

        conversation_history.append({"role": "user", "content": question})
        conversation_history.append({"role": "assistant", "content": answer})

        if len(conversation_history) > MAX_HISTORY_MESSAGES:
            conversation_history[:] = conversation_history[-MAX_HISTORY_MESSAGES:]

    except Exception as exc:
        print(f"OpenAI Error: {exc}")
        answer = "Bloop! My connection to the ocean is interrupted."

    interaction_text = answer

    # ---- 3b. Estimate total speech duration ---------------------------------
    word_count       = len(answer.split())
    words_per_minute = 160.0
    # Add a small buffer so the fish mouth-video doesn't cut off before the
    # final word has fully played out.
    expected_duration = (word_count / words_per_minute) * 60.0 + 0.5

    # ---- 3c. Start video immediately, before the first word plays ----------
    # Seed last_word_time NOW so the display loop sees speech as "active"
    # from the very first frame.
    with speech_lock:
        last_word_time = time.time()

    current_state = STATE_TALKING

    # ---- 3d. Initialise TTS engine -----------------------------------------
    engine = pyttsx3.init()
    engine.setProperty("rate", int(words_per_minute))
    engine.setProperty("voice", "HKEY_LOCAL_MACHINE\\SOFTWARE\\Microsoft\\Speech\\Voices\\Tokens\\TTS_MS_EN-US_DAVID_11.0")
    print("DEBUG - Current voice:", engine.getProperty("voice"))
    def on_word_start(name, location, length):
        """
        Called by pyttsx3 each time a new word begins.
        Refreshing last_word_time keeps is_speech_active() returning True
        for the duration of real speech.  The display loop will freeze the
        video frame only when this timestamp goes stale beyond SPEECH_PAUSE_GRACE.
        """
        global last_word_time
        with speech_lock:
            last_word_time = time.time()

    engine.connect("started-word", on_word_start)

    # ---- 3e. Speak ----------------------------------------------------------
    speech_start = time.time()
    engine.say(answer)
    engine.runAndWait()

    # ---- 3f. Hold STATE_TALKING for the full estimated duration -------------
    # Even after runAndWait() returns, we might be slightly under the expected
    # duration due to timing jitter.  Keep the state alive so the video doesn't
    # snap back to the static image before the audio has truly finished.
    elapsed = time.time() - speech_start
    if elapsed < expected_duration:
        time.sleep(expected_duration - elapsed)

    # ---- 3g. Return to idle -------------------------------------------------
    current_state    = STATE_IDLE
    interaction_text = ""
    # Zero out last_word_time so is_speech_active() returns False immediately.
    with speech_lock:
        last_word_time = 0.0


# ----------------------------
# 4. WEB SERVER & SOCKETS
# ----------------------------
app      = Flask(__name__)
socketio = SocketIO(app, cors_allowed_origins="*", async_mode="threading")

HTML_TEMPLATE = """
<!DOCTYPE html>
<html>
<head>
    <meta name="viewport" content="width=device-width, initial-scale=1.0, maximum-scale=1.0, user-scalable=no">
    <title>Interactive Exhibit</title>
    <style>
        body, html { margin: 0; padding: 0; width: 100%; height: 100%; background: #000; overflow: hidden; font-family: sans-serif; }
        
        #video-container { position: absolute; top: 0; left: 0; width: 100%; height: 100%; display: flex; justify-content: center; align-items: center; pointer-events: none; z-index: 1; }
        img { width: 100%; height: 100%; object-fit: cover; }

        #chat-container {
            position: absolute; bottom: 30px; left: 5%; width: 90%; box-sizing: border-box;
            background: rgba(15, 15, 15, 0.75); padding: 15px; display: flex; gap: 10px; flex-direction: column;
            border: 1px solid #c9a050; border-radius: 12px;
            box-shadow: 0 10px 30px rgba(0,0,0,0.8); z-index: 9999; backdrop-filter: blur(8px);
        }
        
        .input-row { display: flex; gap: 10px; width: 100%; box-sizing: border-box; }

        #question-input {
            flex: 1; padding: 15px; border-radius: 8px; border: 1px solid rgba(255,255,255,0.2);
            font-size: 16px; background: rgba(0,0,0,0.6); color: #fff;
            resize: none; overflow-y: hidden;
            height: 20px; 
            transition: height 0.4s cubic-bezier(0.25, 0.8, 0.25, 1);
        }
        #question-input:focus { 
            height: 100px; 
            outline: none; border-color: #c9a050; 
        }
        #question-input::placeholder { color: #aaa; }

        .btn { padding: 10px; border-radius: 8px; border: none; font-weight: bold; cursor: pointer; color: #fff; text-align: center; }
        #mic-btn { background: #aa3333; font-size: 20px; flex: 0 0 60px; }
        #send-btn { background: #c9a050; font-size: 16px; flex: 1; }

        #rotate-message { display: none; position: absolute; top: 0; left: 0; width: 100%; height: 100%; background: #000; color: white; z-index: 99999; flex-direction: column; justify-content: center; align-items: center; text-align: center; font-size: 26px; font-weight: bold; }
        @media screen and (orientation: portrait) { #rotate-message { display: flex; } }
    </style>
    <script src="https://cdnjs.cloudflare.com/ajax/libs/socket.io/4.0.1/socket.io.js"></script>
</head>
<body>
    <div id="rotate-message">Please rotate your phone sideways!</div>
    <div id="video-container"><img id="stream-image" src="" alt="Connecting..."></div>
    
    <div id="chat-container">
        <textarea id="question-input" placeholder="Tap to type your question..."></textarea>
        <div class="input-row">
            <button class="btn" id="mic-btn" title="Tap to speak">🎤</button>
            <button class="btn" id="send-btn">Ask Question</button>
        </div>
    </div>

    <script>
        const socket = io();
        const img    = document.getElementById('stream-image');
        const input  = document.getElementById('question-input');
        const btn    = document.getElementById('send-btn');
        const mic    = document.getElementById('mic-btn');

        socket.on('update_image', function(data) { img.src = data.image; });

        function sendQuestion(text) {
            const question = (typeof text === 'string') ? text.trim() : input.value.trim();
            if (question !== "") {
                socket.emit('ask_question', { question: question });
                input.value = "";
                input.blur();
            }
        }
        btn.addEventListener('click', () => sendQuestion());
        input.addEventListener('keypress', (e) => {
            if (e.key === 'Enter' && !e.shiftKey) {
                e.preventDefault();
                sendQuestion();
            }
        });

        let recognizing = false;
        const SpeechRecognition = window.SpeechRecognition || window.webkitSpeechRecognition;
        if (SpeechRecognition) {
            const recognition = new SpeechRecognition();
            recognition.lang            = 'en-US';
            recognition.interimResults  = false;
            recognition.onstart  = () => { recognizing = true;  mic.textContent = '🔴'; };
            recognition.onend    = () => { recognizing = false; mic.textContent = '🎤'; };
            recognition.onresult = (event) => {
                const transcript = event.results[0][0].transcript || '';
                if (transcript) sendQuestion(transcript);
            };
            mic.addEventListener('click', () => {
                if (recognizing) { recognition.stop(); } else { recognition.start(); }
            });
        }
    </script>
</body>
</html>
"""


@app.route('/')
def index():
    return render_template_string(HTML_TEMPLATE)


@socketio.on('connect')
def handle_connect():
    global last_interaction_time
    last_interaction_time = time.time()


@socketio.on('ask_question')
def handle_question(data):
    global current_state, last_interaction_time
    last_interaction_time = time.time()

    if current_state == STATE_IDLE:
        question = data['question']
        threading.Thread(target=ask_ai_and_speak, args=(question,), daemon=True).start()


threading.Thread(
    target=lambda: socketio.run(
        app, host='0.0.0.0', port=8080,
        ssl_context='adhoc', allow_unsafe_werkzeug=True
    ),
    daemon=True
).start()


# ----------------------------
# 5. MAIN DISPLAY LOOP
# ----------------------------

def get_local_ip() -> str:
    s = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
    try:
        s.connect(('10.255.255.255', 1))
        return s.getsockname()[0]
    except Exception:
        return '127.0.0.1'
    finally:
        s.close()


LOCAL_IP   = get_local_ip()
SERVER_URL = f"https://{LOCAL_IP}:8080"

qr_pil   = qrcode.make(SERVER_URL).convert('RGB')
qr_image = cv2.cvtColor(np.array(qr_pil), cv2.COLOR_RGB2BGR)
qr_image = cv2.resize(qr_image, (150, 150))

cv2.namedWindow("Interactive Exhibit", cv2.WINDOW_NORMAL)
cv2.setWindowProperty("Interactive Exhibit", cv2.WND_PROP_FULLSCREEN, cv2.WINDOW_FULLSCREEN)

last_emit_time = time.time()
print(f"System Ready. Connect phone to: {SERVER_URL}")

while True:
    current_time = time.time()

    # ------------------------------------------------------------------
    # VISUAL STATE MACHINE
    # ------------------------------------------------------------------
    # • IDLE / THINKING  → always show static tattoo image
    # • TALKING          → show fish video
    #     - is_speech_active() == True  → advance video (mouth moving)
    #     - is_speech_active() == False → freeze on last frame (mouth paused)
    #   The video NEVER goes back to the static image while STATE_TALKING.
    #   It only returns to the static image when STATE_IDLE is set by the
    #   voice thread after the full speech duration has elapsed.
    # ------------------------------------------------------------------

    if current_state in (STATE_IDLE, STATE_THINKING):
        display_frame       = tattoo_image.copy()
        current_video_frame = None          # forget the frozen frame

    else:  # STATE_TALKING
        if is_speech_active():
            # Mouth moving — advance the video by reading the next frame
            new_frame           = get_video_frame(cap_idle)
            current_video_frame = new_frame.copy()
            display_frame       = new_frame
        else:
            # Natural pause in speech — freeze on the last frame we had.
            # If for some reason we don't have one yet, grab one now.
            if current_video_frame is None:
                current_video_frame = get_video_frame(cap_idle)
            display_frame = current_video_frame.copy()

    local_frame = display_frame.copy()

    # ------------------------------------------------------------------
    # QR CODE — show when idle for 20+ seconds
    # ------------------------------------------------------------------
    if (current_time - last_interaction_time) > 20:
        display_frame[20:170, 20:170] = qr_image
        local_frame[20:170, 20:170]   = qr_image

    # ------------------------------------------------------------------
    # STREAM TO MOBILE APP
    # ------------------------------------------------------------------
    if current_time - last_emit_time > (1.0 / STREAM_FPS):
        small_display  = cv2.resize(display_frame, (640, 360))
        _, img_encoded = cv2.imencode('.jpg', small_display, [cv2.IMWRITE_JPEG_QUALITY, 55])
        img_base64     = base64.b64encode(img_encoded).decode('utf-8')
        socketio.emit('update_image', {'image': f'data:image/jpeg;base64,{img_base64}'})
        last_emit_time = current_time

    # ------------------------------------------------------------------
    # LOCAL DISPLAY (monitor attached to the exhibit machine)
    # ------------------------------------------------------------------
    cv2.imshow("Interactive Exhibit", local_frame)
    if cv2.waitKey(1) & 0xFF == 27:   # ESC to quit
        break

# Cleanup
cap_idle.release()
cv2.destroyAllWindows()