# AI Fish Exhibit 🐟

An interactive, voice-driven AI exhibit built for a museum-style installation. Visitors scan a QR code with their phone, ask a question, and a magical animated fish answers out loud — with synced video, real-time streaming, and short-term conversational memory.

## How it works

1. A computer running the exhibit displays a static image (idle) or an animated fish video (talking) on a screen.
2. Visitors scan a QR code with their phone, which opens a simple web page with a text box and a microphone button.
3. The question is sent over a local WebSocket connection to the exhibit computer.
4. The backend sends the question to an LLM (Groq's Llama 3.3 70B), gets a response in-character as an "ancient fish," and speaks it aloud using text-to-speech.
5. The fish video is synced to the speech in real time — advancing frames while speaking, freezing on natural pauses, and returning to the idle image once the answer finishes.
6. A live low-latency video feed of the exhibit is also streamed to the visitor's phone via Socket.IO.

## My contribution

This was originally built as part of a team project. **My specific contribution was adding short-term conversational memory** — before this feature, every question was sent to the AI in total isolation with no awareness of anything asked earlier. I implemented:

- A trimmed message-history list (`conversation_history`) that stores recent question/answer pairs
- Logic to inject that history into every new API call so the AI can reference earlier parts of the conversation
- A cap (`MAX_HISTORY_MESSAGES`) that keeps only the most recent exchanges, balancing conversational context against API cost and response latency

I also migrated the project's API integration from OpenAI to Groq (Llama 3.3 70B) for faster inference and to remove a paid-billing dependency, and fixed a hardcoded API key security issue by moving credentials into environment variables.

## Tech stack

- **Python** — core application logic
- **OpenCV** — video frame processing and local display
- **Flask + Flask-SocketIO** — local web server and real-time communication with the visitor's phone
- **Groq API (Llama 3.3 70B)** — conversational AI responses
- **pyttsx3** — offline text-to-speech
- **qrcode** — generates the pairing QR code shown on the exhibit screen
- **python-dotenv** — environment variable management for API keys

## Setup

1. Clone the repo:
   ```bash
   git clone https://github.com/AminWidatalla/ai-fish-exhibit.git
   cd ai-fish-exhibit
   ```

2. Install dependencies:
   ```bash
   pip install -r requirements.txt
   ```

3. Create a `.env` file in the project root with your own Groq API key:
   ```
   GROQ_API_KEY=your-groq-key-here
   ```
   (Get a free key at [console.groq.com](https://console.groq.com))

4. Add your own media files to the `images/` folder:
   - `Tattoo.png` — the static idle image
   - `fish iddle.mov` — the animated talking video

5. Run it:
   ```bash
   python main.py
   ```

6. Make sure your phone is on the **same network** as the computer running the script, then scan the QR code shown on screen.

## Notes

- Requires an internet connection on the network used, since the AI response call needs to reach Groq's API.
- The exhibit's phone-facing web page uses a self-signed SSL certificate for local HTTPS — browsers will show a security warning on first connection, which is expected for local testing.

## License

MIT
