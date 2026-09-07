import cv2
import numpy as np
import time
import socket
import threading
import qrcode
import base64
import sys
import os
import platform
import subprocess
from pathlib import Path
from dotenv import load_dotenv
from flask import Flask, render_template_string
from flask_socketio import SocketIO

import pyttsx3
from openai import OpenAI

# ----------------------------
# 1. SETTINGS & CONFIGURATION
# ----------------------------
load_dotenv()

OPENAI_API_KEY = os.getenv("OPENAI_API_KEY")

if not OPENAI_API_KEY:
    raise ValueError("OPENAI_API_KEY not found. Please create a .env file with OPENAI_API_KEY=your_key")

W, H = 1280, 720
VIDEO_H = H

BASE_DIR = Path(getattr(sys, 'frozen', False) and getattr(sys, '_MEIPASS', Path.cwd()) or Path(__file__).resolve().parent)
IMAGE_FOLDER = BASE_DIR / "images"
STREAM_FPS = 15

# ---------------------------------------------------------------------------
# CONVERSATION STATES & HISTORY
# ---------------------------------------------------------------------------
STATE_IDLE     = "idle"
STATE_THINKING = "thinking"
STATE_TALKING  = "talking"

current_state         = STATE_IDLE
last_interaction_time = 0
interaction_text      = ""

conversation_history = []
MAX_HISTORY = 6

# ---------------------------------------------------------------------------
# VOICE ENGINE & VIDEO SYNC (CROSS-PLATFORM)
# ---------------------------------------------------------------------------
SPEECH_PAUSE_GRACE = 0.6
last_word_time      = 0.0
speech_lock         = threading.Lock()

current_video_frame: np.ndarray | None = None
system_os = platform.system()


def is_speech_active() -> bool:
    """Return True if a word was spoken recently enough to consider TTS active."""
    with speech_lock:
        return (time.time() - last_word_time) < SPEECH_PAUSE_GRACE


def _speak_windows_threaded(text: str) -> None:
    """Run TTS on Windows in an isolated thread to prevent driver lockups."""
    try:
        temp_engine = pyttsx3.init('sapi5')
        temp_engine.setProperty("rate", 160)
        temp_engine.say(text)
        temp_engine.runAndWait()
    except Exception as e:
        print(f"Windows TTS Thread Error: {e}")


# ----------------------------
# 2. LOAD VIDEO ASSETS
# ----------------------------
TATTOO_IMAGE_PATH = IMAGE_FOLDER / "Tattoo.png"
IDLE_VIDEO_PATH   = IMAGE_FOLDER / "fish iddle.mov"

cap_idle = cv2.VideoCapture(str(IDLE_VIDEO_PATH))

# Adjustable Video FPS Speed Control (Increase if too slow, decrease if too fast)
TARGET_VIDEO_FPS = 45.0  
FRAME_DELAY      = 1.0 / TARGET_VIDEO_FPS
last_frame_time  = 0.0


def _make_fallback_frame(text: str) -> np.ndarray:
    frame = np.zeros((H, W, 3), dtype=np.uint8)
    frame[:] = (20, 20, 20)
    cv2.putText(frame, text, (W // 2 - 260, H // 2),
                cv2.FONT_HERSHEY_SIMPLEX, 1.2, (255, 255, 255), 3)
    return frame


if TATTOO_IMAGE_PATH.exists():
    tattoo_image = cv2.imread(str(TATTOO_IMAGE_PATH))
    if tattoo_image is not None:
        tattoo_image = cv2.resize(tattoo_image, (W, H))
    else:
        tattoo_image = _make_fallback_frame("TATTOO IMAGE NOT FOUND")
else:
    tattoo_image = _make_fallback_frame("TATTOO IMAGE NOT FOUND")


def get_video_frame(cap: cv2.VideoCapture) -> np.ndarray:
    """Read next frame from cap, keeping timing locked to TARGET_VIDEO_FPS."""
    global last_frame_time, current_video_frame
    
    current_time = time.time()
    
    if (current_time - last_frame_time) >= FRAME_DELAY:
        ret, frame = cap.read()
        if not ret:
            cap.set(cv2.CAP_PROP_POS_FRAMES, 0)
            ret, frame = cap.read()

        if ret:
            current_video_frame = cv2.resize(frame, (W, H))
            last_frame_time     = current_time

    if current_video_frame is not None:
        return current_video_frame

    fallback = np.zeros((H, W, 3), dtype=np.uint8)
    fallback[:] = (50, 20, 20)
    cv2.putText(fallback, "VIDEO NOT FOUND", (W // 2 - 200, H // 2),
                cv2.FONT_HERSHEY_SIMPLEX, 1.5, (255, 255, 255), 3)
    return fallback


# ----------------------------
# 3. AI & VOICE THREAD ENGINE
# ----------------------------

def ask_ai_and_speak(question: str, language_name: str = "English") -> None:
    global current_state, interaction_text, last_word_time, conversation_history

    current_state    = STATE_THINKING
    interaction_text = "..."

    system_prompt = (
        "You are a magical, ancient fish in a bowl. "
        "A human woman is standing beside you. "
        "The user talking to you will expect the human to answer, "
        "but YOU (the fish) are the one talking. You must surprise them. "
        "Keep answers under 3 short sentences. "
        "NEVER say you are an AI, a language model, or from OpenAI. "
        "If asked 'who are you', say you are the ancient fish of this museum. "
        f"CRITICAL: Respond exclusively in {language_name}."
    )

    messages = [{"role": "system", "content": system_prompt}]
    messages.extend(conversation_history)
    messages.append({"role": "user", "content": question})

    try:
        client   = OpenAI(api_key=OPENAI_API_KEY, timeout=10.0)
        response = client.chat.completions.create(
            model="gpt-4o-mini",
            messages=messages,
        )
        answer = response.choices[0].message.content
    except Exception as exc:
        print(f"OpenAI Error: {exc}")
        answer = "Bloop! My connection to the ocean is interrupted."

    interaction_text = answer

    conversation_history.append({"role": "user", "content": question})
    conversation_history.append({"role": "assistant", "content": answer})
    if len(conversation_history) > MAX_HISTORY:
        conversation_history = conversation_history[-MAX_HISTORY:]

    current_state = STATE_TALKING
    safe_answer   = answer.replace('"', '').replace("'", "")

    word_count        = len(answer.split())
    expected_duration = (word_count / 160.0) * 60.0 + 0.5

    with speech_lock:
        last_word_time = time.time()

    if system_os == "Linux":
        proc = subprocess.Popen(["espeak-ng", "-s", "160", safe_answer])
        while proc.poll() is None:
            with speech_lock:
                last_word_time = time.time()
            time.sleep(0.1)
    else:
        tts_thread = threading.Thread(target=_speak_windows_threaded, args=(safe_answer,), daemon=True)
        tts_thread.start()

        start_time = time.time()
        while tts_thread.is_alive() or (time.time() - start_time) < expected_duration:
            with speech_lock:
                last_word_time = time.time()
            time.sleep(0.1)

    current_state    = STATE_IDLE
    interaction_text = ""
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
        #language-select {
            padding: 8px; border-radius: 6px; background: rgba(0,0,0,0.8); color: #c9a050;
            border: 1px solid #c9a050; font-size: 14px; margin-bottom: 5px;
        }
        #question-input {
            flex: 1; padding: 15px; border-radius: 8px; border: 1px solid rgba(255,255,255,0.2);
            font-size: 16px; background: rgba(0,0,0,0.6); color: #fff;
            resize: none; overflow-y: hidden; height: 20px;
            transition: height 0.4s cubic-bezier(0.25, 0.8, 0.25, 1);
        }
        #question-input:focus { height: 100px; outline: none; border-color: #c9a050; }
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
        <select id="language-select">
            <option value="en-US|English">English</option>
            <option value="es-ES|Spanish">Español</option>
            <option value="fr-FR|French">Français</option>
            <option value="de-DE|German">Deutsch</option>
            <option value="tr-TR|Turkish">Türkçe</option>
            <option value="ar-SA|Arabic">العربية</option>
            <option value="zh-CN|Mandarin">中文</option>
        </select>
        <textarea id="question-input" placeholder="Tap to type your question..."></textarea>
        <div class="input-row">
            <button class="btn" id="mic-btn" title="Tap to speak">🎤</button>
            <button class="btn" id="send-btn">Ask Question</button>
        </div>
    </div>
    <script>
        const socket   = io();
        const img      = document.getElementById('stream-image');
        const input    = document.getElementById('question-input');
        const btn      = document.getElementById('send-btn');
        const mic      = document.getElementById('mic-btn');
        const langSel  = document.getElementById('language-select');

        socket.on('update_image', function(data) { img.src = data.image; });

        function getSelectedLang() {
            const parts = langSel.value.split('|');
            return { code: parts[0], name: parts[1] };
        }

        function sendQuestion(text) {
            const question = (typeof text === 'string') ? text.trim() : input.value.trim();
            if (question !== "") {
                const lang = getSelectedLang();
                socket.emit('ask_question', { question: question, language_name: lang.name });
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
            recognition.interimResults = false;

            recognition.onstart = () => { recognizing = true; mic.textContent = '🔴'; };
            recognition.onend   = () => { recognizing = false; mic.textContent = '🎤'; };
            recognition.onresult = (event) => {
                const transcript = event.results[0][0].transcript || '';
                if (transcript) sendQuestion(transcript);
            };

            mic.addEventListener('click', () => {
                if (recognizing) {
                    recognition.stop();
                } else {
                    recognition.lang = getSelectedLang().code;
                    recognition.start();
                }
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
        question = data.get('question', '')
        language_name = data.get('language_name', 'English')
        threading.Thread(
            target=ask_ai_and_speak, 
            args=(question, language_name), 
            daemon=True
        ).start()


threading.Thread(
    target=lambda: socketio.run(
        app, host='0.0.0.0', port=8080,
        allow_unsafe_werkzeug=True
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
SERVER_URL = f"http://{LOCAL_IP}:8080"

qr_pil   = qrcode.make(SERVER_URL).convert('RGB')
qr_image = cv2.cvtColor(np.array(qr_pil), cv2.COLOR_RGB2BGR)
qr_image = cv2.resize(qr_image, (150, 150))

cv2.namedWindow("Interactive Exhibit", cv2.WINDOW_NORMAL)
cv2.setWindowProperty("Interactive Exhibit", cv2.WND_PROP_FULLSCREEN, cv2.WINDOW_FULLSCREEN)

last_emit_time = time.time()
print(f"System Ready. Connect phone to: {SERVER_URL}")

while True:
    current_time = time.time()

    if current_state in (STATE_IDLE, STATE_THINKING):
        display_frame       = tattoo_image.copy()
        current_video_frame = None

    else:  # STATE_TALKING
        if is_speech_active():
            display_frame = get_video_frame(cap_idle)
        else:
            if current_video_frame is None:
                current_video_frame = get_video_frame(cap_idle)
            display_frame = current_video_frame.copy()

    local_frame = display_frame.copy()

    if (current_time - last_interaction_time) > 20:
        display_frame[20:170, 20:170] = qr_image
        local_frame[20:170, 20:170]   = qr_image

    if current_time - last_emit_time > (1.0 / STREAM_FPS):
        small_display  = cv2.resize(display_frame, (640, 360))
        _, img_encoded = cv2.imencode('.jpg', small_display, [cv2.IMWRITE_JPEG_QUALITY, 55])
        img_base64     = base64.b64encode(img_encoded).decode('utf-8')
        socketio.emit('update_image', {'image': f'data:image/jpeg;base64,{img_base64}'})
        last_emit_time = current_time

    cv2.imshow("Interactive Exhibit", local_frame)
    if cv2.waitKey(1) & 0xFF == 27:
        break

cap_idle.release()
cv2.destroyAllWindows()