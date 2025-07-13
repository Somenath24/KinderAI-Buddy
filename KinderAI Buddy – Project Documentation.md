# KinderAI Buddy – Project Documentation

---

## Project Overview

KinderAI Buddy is an interactive, child-friendly **Agentic AI** web application that engages kids in safe, adaptive, and educational conversations.  
It is designed as an **Agentic AI system**—capable of self-reflection, goal adaptation, and continuous personalization based on each child’s unique journey.  
KinderAI Buddy features personalized chat, adaptive difficulty, memory extraction, skill tracking, face/hand detection, playful multimedia responses, and an agentic loop for self-reflection and goal adaptation.

---

## 🌟 Features

- **Personalized AI chat** for children with memory and skill tracking
- **Adaptive difficulty** for math/language questions
- **Face and hand detection**, face recognition (MediaPipe, OpenCV, face_recognition)
- **Name recognition** for personalized greetings and session continuity
- **Text-to-speech** and **speech-to-text** (voice chat, OpenAI Whisper, ElevenLabs, browser TTS)
- **Story and image generation** (OpenAI, Unsplash)
- **Agentic self-reflection & goal adaptation**: At the end of each session, the AI reflects on the conversation and sets a personalized goal for the next session, adapting its behavior accordingly.
- **Session and chat history management** (SQLite, Flask-Session)
- **Progress and plan tracking** (session plan, completed activities, skill level)
- **Safe, child-friendly prompt** (age-appropriate, positive, supportive)
- **Web dashboard** with live video, chat, and settings

---

## 🧠 Agentic Capabilities

- **Self-reflection:** At the end of each session, the AI uses OpenAI to reflect on the child's engagement and learning, saving a summary and a suggested goal for the next session.
- **Goal adaptation:** The next session automatically uses the last session's goal, making the experience adaptive and personalized.
- **Skill adaptation:** The AI tracks correct/incorrect answers and adjusts question difficulty to match the child's skill level.
- **Memory extraction:** Periodically extracts and updates the child's preferences, mood, and interests using OpenAI.

---

## 🔄 Agentic Loop: Reflection, Goal Adaptation, and Personalization

### What is the Agentic Loop?

The **agentic loop** is a process where the AI system not only interacts with the user, but also reflects on its own performance, adapts its future behavior, and personalizes its approach based on past sessions. This loop enables the system to become more effective and supportive over time.

### How are Self-Reflection and Goal Adaptation Implemented?

- **Self-Reflection:**  
  At the end of each session (when the planned activities are completed), the AI uses OpenAI's language model to analyze the entire conversation. It generates a reflection summarizing the child's engagement, interests, and learning progress.
- **Goal Adaptation:**  
  Along with the reflection, the AI suggests a personalized goal for the next session (e.g., "Encourage the child to talk more about their favorite animals"). This goal is saved and automatically used to guide the next session's conversation and activities.

### How Does the System Adapt to the Child Over Time?

- At the start of each new session, the last session's goal is loaded and incorporated into the AI's system prompt, ensuring continuity and personalization.
- The AI tracks the child's skill level (e.g., math/language question difficulty) and adapts questions accordingly.
- The system periodically extracts and updates the child's preferences, mood, and interests, further personalizing the experience.
- All reflections and goals are stored in the database, enabling longitudinal adaptation and analysis.

**This agentic loop allows KinderAI Buddy to become more responsive, supportive, and tailored to each child's unique needs and growth over multiple sessions.**

---

## 🗺️ System Architecture & Data Flow

```
+-------------------+         +-------------------+         +-------------------+
|                   |         |                   |         |                   |
|   User (Child)    +-------->+   Web Dashboard   +-------->+   Flask Backend   |
| (Text/Voice/Video)|         | (HTML/JS/CSS)     |         |                   |
+-------------------+         +-------------------+         +-------------------+
         |                            |                               |
         |                            v                               |
         |                +-------------------+                      |
         |                |  Live Video Feed  |                      |
         |                +-------------------+                      |
         |                            |                               |
         |                            v                               |
         |                +-------------------+                      |
         |                | Face/Hand Detect  |<---------------------+
         |                | (MediaPipe, OpenCV|                      |
         |                |  face_recognition)|                      |
         |                +-------------------+                      |
         |                            |                               |
         |                            v                               |
         |                +-------------------+                      |
         |                | Name Recognition  |                      |
         |                +-------------------+                      |
         |                            |                               |
         |                            v                               |
         |                +-------------------+                      |
         |                |  Chat Interface   |<---------------------+
         |                +-------------------+                      |
         |                            |                               |
         |                            v                               |
         |                +-------------------+                      |
         |                |  Voice Service    |<---------------------+
         |                | (TTS/STT:         |                      |
         |                |  ElevenLabs,      |                      |
         |                |  OpenAI Whisper,  |                      |
         |                |  browser TTS)     |                      |
         |                +-------------------+                      |
         |                            |                               |
         |                            v                               |
         |                +-------------------+                      |
         |                |  Story/Image Gen  |<---------------------+
         |                | (OpenAI, Unsplash)|                      |
         |                +-------------------+                      |
         |                            |                               |
         |                            v                               |
         |                +-------------------+                      |
         |                |  Skill Tracking   |<---------------------+
         |                +-------------------+                      |
         |                            |                               |
         |                            v                               |
         |                +-------------------+                      |
         |                |  Memory Extraction|<---------------------+
         |                |  (OpenAI)         |                      |
         |                +-------------------+                      |
         |                            |                               |
         |                            v                               |
         |                +-------------------+                      |
         |                |  Agentic Loop     |<---------------------+
         |                |  (Reflection &    |                      |
         |                |   Goal Adaptation)|                      |
         |                +-------------------+                      |
         |                            |                               |
         |                            v                               |
         |                +-------------------+                      |
         |                |  Database         |<---------------------+
         |                | (SQLite: chat,    |                      |
         |                |  profiles,        |                      |
         |                |  reflections)     |                      |
         |                +-------------------+                      |
         |                                                        |
         +--------------------------------------------------------+
```

**Key modules:**
- **Live video is analyzed for face and hand detection and name recognition to personalize the experience.**
- **Recognized faces trigger personalized greetings and session continuity.**
- **Voice services handle both TTS and STT using ElevenLabs, OpenAI Whisper, and browser TTS.**
- **Chat interface supports text and voice, with story/image generation and skill/memory tracking.**
- **Agentic loop enables self-reflection and goal adaptation, with all data stored in SQLite.**

---

## 🛠️ Technologies Used

| Category         | Technology/Library                |
|------------------|----------------------------------|
| Backend          | Flask, Flask-Session             |
| AI/LLM           | OpenAI API (ChatGPT, Whisper)    |
| Voice            | ElevenLabs API (TTS), browser TTS|
| Vision           | MediaPipe, OpenCV, face_recognition |
| Storage          | SQLite                           |
| Images           | Unsplash API                     |
| Frontend         | HTML, CSS, JavaScript            |

---

## 🚀 Getting Started

### 1. Clone the repository

```sh
git clone https://github.com/yourusername/kinderai-buddy.git
cd kinderai-buddy
```

### 2. Install dependencies

```sh
pip install -r requirements.txt
```

### 3. Configure API keys

Create a `.env` file in the project root with your keys:

```
API_KEY=your_openai_api_key
ELEVENLABS_API_KEY=your_elevenlabs_api_key
UNSPLASH_ACCESS_KEY=your_unsplash_access_key
SECRET_KEY=your_flask_secret_key
```

### 4. Run the app

```sh
python flask_dashboard_app.py
```

Visit [http://localhost:5000](http://localhost:5000) in your browser.

---

## 📚 How It Works

- **Session Management:** Each child session is tracked with a unique session ID and profile. All chat history is stored server-side (no cookie size issues).
- **Agentic Loop:** At the end of a session plan, the AI reflects on the session and saves a summary and next goal. The next session adapts to this goal.
- **Multimodal:** Supports text, voice, and video (face/hand detection and recognition).
- **Personalization:** The AI adapts to the child's age, mood, preferences, and skill level.
- **Face/Hand Detection & Name Recognition:** Live video is analyzed for face and hand detection and name recognition to personalize the experience. Recognized faces trigger personalized greetings and session continuity.
- **Safety:** All prompts and responses are filtered for age-appropriateness and positivity.

---

## Key Code Snippets

### Flask App Configuration (Session, API Keys)
```python
from flask import Flask
from flask_session import Session
import os

app = Flask(__name__)
app.config["SESSION_TYPE"] = "filesystem"
app.config["SESSION_FILE_DIR"] = "./flask_session_data"
app.config["SESSION_PERMANENT"] = False
Session(app)

API_KEY = os.environ.get("API_KEY")
ELEVENLABS_API_KEY = os.environ.get("ELEVENLABS_API_KEY")
UNSPLASH_ACCESS_KEY = os.environ.get("UNSPLASH_ACCESS_KEY")
app.secret_key = os.environ.get("SECRET_KEY", "default_secret")
```

### Self-Reflection and Goal Adaptation Logic
```python
def generate_session_reflection(chat_history, api_key=API_KEY, model="gpt-3.5-turbo"):
    """
    Uses OpenAI to generate a session reflection and suggest a next goal based on chat history.
    Returns: tuple (reflection, next_goal)
    """
    import openai, json
    chat_text = ""
    for msg in chat_history:
        if msg["role"] == "user":
            chat_text += f'Child: {msg["content"]}\n'
        elif msg["role"] == "assistant":
            chat_text += f'Assistant: {msg["content"]}\n'
    prompt = (
        "You are a child psychologist AI. Given the following chat, reflect on the session: "
        "Did the child talk more? What did they enjoy? Suggest a goal for the next session. "
        "Respond in JSON with keys 'reflection' and 'next_goal'.\n\n"
        f"Chat:\n{chat_text}\n"
    )
    client = openai.OpenAI(api_key=api_key)
    try:
        response = client.chat.completions.create(
            model=model,
            messages=[{"role": "user", "content": prompt}],
            temperature=0,
            max_tokens=256,
        )
        result = json.loads(response.choices[0].message.content.strip())
        return result.get("reflection", ""), result.get("next_goal", "")
    except Exception as e:
        return f"Error: {e}", ""
```

### Face/Hand Detection Example
```python
def detect_faces(image_bytes):
    """
    Detects faces in an image using OpenCV's Haar Cascade.
    Returns: (base64-encoded image string, number of faces detected)
    """
    nparr = np.frombuffer(image_bytes, np.uint8)
    img = cv2.imdecode(nparr, cv2.IMREAD_COLOR)
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    faces = cv2.CascadeClassifier(cv2.data.haarcascades + "haarcascade_frontalface_default.xml") \
        .detectMultiScale(gray, 1.1, 4)
    for (x, y, w, h) in faces:
        cv2.rectangle(img, (x, y), (x + w, y + h), (0, 255, 0), 2)
    _, buffer = cv2.imencode('.jpg', img)
    img_b64 = base64.b64encode(buffer).decode('utf-8')
    return img_b64, len(faces)
```

---

## ❓ FAQ

**Q: What is KinderAI Buddy?**  
A: KinderAI Buddy is an interactive, child-friendly AI web application that engages kids in safe, adaptive, and educational conversations using text, voice, and video. It features personalized chat, skill tracking, face/hand detection, and an agentic loop for self-reflection and goal adaptation.

---

**Q: How does the agentic loop work?**  
A: At the end of each session, the AI reflects on the conversation using OpenAI, saves a summary and a suggested goal for the next session, and adapts its behavior accordingly. The next session starts with this new goal for continuity and personalization.

---

**Q: What technologies are used for face and hand detection?**  
A: The system uses MediaPipe and OpenCV for hand and face detection, and the `face_recognition` library for recognizing known faces.

---

**Q: How is voice handled?**  
A: Voice input is transcribed using OpenAI Whisper, and voice output is generated using ElevenLabs or browser-based TTS.

---

**Q: Is my child's data safe?**  
A: All chat history, profiles, and reflections are stored securely on your server using SQLite. No data is shared with third parties except for AI processing (OpenAI, ElevenLabs, Unsplash).

---

**Q: Can I run this on my own computer?**  
A: Yes! Just follow the "Getting Started" instructions above to install dependencies, configure API keys, and run the app locally.

---

**Q: How can I contribute or suggest features?**  
A: Pull requests and suggestions are welcome! Please open an issue or submit a PR on GitHub.

---

**Q: Who do I contact for support or collaboration?**  
A: See the Contact section above or open an issue on GitHub.

---

## 📝 Academic/Research Use

KinderAI Buddy implements an **agentic loop** (reflection, adaptation, and memory) and is suitable for research on adaptive, personalized, and multimodal AI companions for children.  
If you use this project in research, please cite appropriately and respect the [license](LICENSE).

---

## 📄 License

Creative Commons Attribution-NonCommercial 4.0 International (CC BY-NC 4.0)  
See LICENSE file for details.

---

## 🤝 Contributing

Pull requests and suggestions are welcome! Please open an issue or submit a PR.

---

## 📧 Contact

https://github.com/somenath24

---

## 🛟 Support & Issues

For support or to report bugs, please open an issue on [GitHub](https://github.com/somenath24).