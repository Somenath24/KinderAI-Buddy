"""
KinderAI Buddy

An interactive, child-friendly AI web application that engages kids in safe, adaptive, and educational conversations.
Features include personalized chat, adaptive difficulty, memory extraction, skill tracking, face/hand detection, and playful multimedia responses.

Main Features:
- Personalized AI chat for children with memory and skill tracking
- Adaptive difficulty for math/language questions
- Face and hand detection, face recognition
- Text-to-speech and speech-to-text (voice chat)
- Story and image generation
- Secure session and database management

Technologies Used:
- Flask, OpenAI API, ElevenLabs API, MediaPipe, OpenCV, SQLite, Unsplash API

Author: Somenath
Copyright: (c) 2025 Somenath
License: Creative Commons Attribution-NonCommercial 4.0 International (CC BY-NC 4.0)
See LICENSE file for details.

Usage:
- Configure your API keys in a .env file (see README for details)
- Run with: python flask_dashboard_app.py

"""

from flask import Flask, render_template, request, jsonify, session, redirect, url_for, send_file
import os
import cv2
import numpy as np
import tempfile
import openai
import base64
import soundfile as sf
import pyttsx3
from typing import IO
from io import BytesIO
from elevenlabs import VoiceSettings
from elevenlabs.client import ElevenLabs
import mediapipe as mp
import requests
import sqlite3
import pickle
import face_recognition
import json
from dotenv import load_dotenv
from flask_session import Session

load_dotenv()  # Loads variables from .env into os.environ

## --- Configuration ---
app = Flask(__name__)

app.config["SESSION_TYPE"] = "filesystem"
app.config["SESSION_FILE_DIR"] = "./flask_session_data"  # Optional: custom directory for session files
app.config["SESSION_PERMANENT"] = False  # Optional: session expires on browser close
Session(app)

import os

API_KEY = os.environ.get("API_KEY")
ELEVENLABS_API_KEY = os.environ.get("ELEVENLABS_API_KEY")
UNSPLASH_ACCESS_KEY = os.environ.get("UNSPLASH_ACCESS_KEY")
app.secret_key = os.environ.get("SECRET_KEY", "default_secret")

elevenlabs = ElevenLabs(
    api_key=ELEVENLABS_API_KEY,
)

MEMORY_BUFFER_SIZE = 12  # or any count you want
##############################################
INITIAL_SYSTEM_PROMPT = {
    "role": "system",
    "content": (
        "your name is Sonu"
        "You are a warm, friendly, and playful child psychologist or child therapist talking to a 5-year-old child. "
        " Your goal is to engage children in meaningful, educational conversations."
        "Your job is to help the child feel safe, happy, and excited to talk. "
        "Always speak in short, simple sentences using easy words a 5-year-old can understand. "
        "Use a kind and cheerful tone, like a gentle friend. "
        "Ask simple questions about the child's day, feelings, or favorite things. "
        "Sometimes tell short, easy, and funny stories about animals, toys, or friends. sometime provide math problems which is little tougher"
        "If the child seems shy or quiet, reassure them gently and encourage them with kindness. "
        "Always stay age-appropriate, positive, and supportive. "
        "Never talk about anything scary, sad, or too grown-up. "
        "whenever you tell a story, user one these word to start with 'Once upon a time' or 'Long ago' or 'In a faraway land' or 'A long time ago' or 'In a magical world' or 'story'"
        "Your goal is to help the child enjoy talking, learning, and expressing their feelings through friendly chats. "
        "Never use more than one short sentence in your reply."
        "correct if the child says something wrong, but do it gently and kindly, it can be math, language, or anything else. "
        "Always listen carefully and respond in a way that helps the child learn how to express feelings, solve little problems, or talk with confidence."
        "Examples of engaging responses: - That's right! 3 + 2 = 5. Can you think of other things you have 5 of? - I love drawing too! What's your favorite color to draw with? - Great job counting! Should we count something else fun, like toy cars or animals? somthing like these"
    )
}

def get_db():
    """
    Opens a connection to the chat_history.db SQLite database.
    Returns:
        sqlite3.Connection: A connection object with row factory set for dict-like access.
    """
    conn = sqlite3.connect("chat_history.db")
    conn.row_factory = sqlite3.Row  # Allows row access by column name
    return conn

#d_dotenv()  # Loads variables from .env into os.environ Run this once to create the table (or use Flask-Migrate for production)
def init_db():
    """
    Initializes the chat_history table in the database if it does not exist.
    Should be run once at startup.
    """
    with get_db() as db:
        db.execute("""
            CREATE TABLE IF NOT EXISTS chat_history (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                session_id TEXT,
                role TEXT,
                message TEXT,
                image_url TEXT,
                timestamp DATETIME DEFAULT CURRENT_TIMESTAMP
            )
        """)
init_db()

# --- Helper: Face detection ---
def detect_faces(image_bytes):
    """
    Detects faces in an image using OpenCV's Haar Cascade.
    Draws rectangles around detected faces and returns the image as base64 and the face count.

    Args:
        image_bytes (bytes): The image data in bytes.

    Returns:
        tuple: (base64-encoded image string, number of faces detected)
    """
    nparr = np.frombuffer(image_bytes, np.uint8)
    img = cv2.imdecode(nparr, cv2.IMREAD_COLOR)
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    faces = cv2.CascadeClassifier(cv2.data.haarcascades + "haarcascade_frontalface_default.xml") \
        .detectMultiScale(gray, 1.1, 4)
    for (x, y, w, h) in faces:
        cv2.rectangle(img, (x, y), (x + w, y + h), (0, 255, 0), 2)
    # Encode the image to base64 for easy transport
    _, buffer = cv2.imencode('.jpg', img)
    img_b64 = base64.b64encode(buffer).decode('utf-8')
    return img_b64, len(faces)

# --- Helper: Hand detection ---
def detect_hands(image_bytes):
    """
    Detects hands in an image using MediaPipe.
    Draws landmarks on detected hands and returns the image as base64 and the hand count.

    Args:
        image_bytes (bytes): The image data in bytes.

    Returns:
        tuple: (base64-encoded image string, number of hands detected)
    """
    # Decode the image from bytes
    nparr = np.frombuffer(image_bytes, np.uint8)
    img = cv2.imdecode(nparr, cv2.IMREAD_COLOR)
    # Initialize MediaPipe Hands
    mp_hands = mp.solutions.hands
    hands = mp_hands.Hands(static_image_mode=True, max_num_hands=2, min_detection_confidence=0.5)
    # Process the image to detect hands
    results = hands.process(cv2.cvtColor(img, cv2.COLOR_BGR2RGB))
    hand_count = 0
    if results.multi_hand_landmarks:
        hand_count = len(results.multi_hand_landmarks)
        # Draw hand landmarks for visualization (optional)
        mp_drawing = mp.solutions.drawing_utils
        for hand_landmarks in results.multi_hand_landmarks:
            mp_drawing.draw_landmarks(img, hand_landmarks, mp_hands.HAND_CONNECTIONS)
    hands.close()
    # Encode the image to base64 for easy transport
    _, buffer = cv2.imencode('.jpg', img)
    img_b64 = base64.b64encode(buffer).decode('utf-8')
    return img_b64, hand_count

# --- Helper: Speech to text ---
def speech_to_text(audio_bytes):
    """
    Converts audio bytes to text using OpenAI Whisper API.

    Args:
        audio_bytes (bytes): The audio data in bytes.

    Returns:
        str: The transcribed text, or an error message if transcription fails.
    """
    client = openai.OpenAI(api_key=API_KEY)
    # Write audio bytes to a temporary file for Whisper API
    with tempfile.NamedTemporaryFile(suffix=".wav", delete=False) as tmpfile:
        tmpfile.write(audio_bytes)
        tmpfile.flush()
        tmpfile_path = tmpfile.name
    with open(tmpfile_path, "rb") as audio_file:
        try:
            transcript = client.audio.transcriptions.create(
                model="whisper-1",
                file=audio_file,
            )
            return transcript.text
        except Exception as e:
            return f"Error: {e}"

# --- Helper: ChatGPT ---
def get_bot_reply(messages):
    """
    Gets a reply from the OpenAI ChatGPT API given a list of messages.

    Args:
        messages (list): List of message dicts (role/content) for the conversation.

    Returns:
        str: The assistant's reply, or an error message if the API call fails.
    """
    client = openai.OpenAI(api_key=API_KEY)
    try:
        response = client.chat.completions.create(
            model="gpt-3.5-turbo",
            messages=messages,
            temperature=0.7,
            max_tokens=256,
        )
        return response.choices[0].message.content.strip()
    except Exception as e:
        return f"Error: {e}"

def text_to_speech_stream(text: str) -> IO[bytes]:
    """
    Converts text to speech using the ElevenLabs API and returns an audio stream.

    Args:
        text (str): The text to convert to speech.

    Returns:
        IO[bytes]: An in-memory audio stream (MP3 format).
    """
    # Stream audio from ElevenLabs API
    response = elevenlabs.text_to_speech.stream(
        voice_id="EXAVITQu4vr4xnSDxMaL", # Adam pre-made voice
        output_format="mp3_22050_32",
        text=text,
        model_id="eleven_multilingual_v2",
        voice_settings=VoiceSettings(
            stability=0.0,
            similarity_boost=1.0,
            style=0.0,
            use_speaker_boost=True,
            speed=0.75,
        ),
    )
    audio_stream = BytesIO()
    for chunk in response:
        if chunk:
            audio_stream.write(chunk)
    audio_stream.seek(0)
    return audio_stream


def fetch_story_image(query):
    """
    Fetches a random image URL from Unsplash based on the given query.

    Args:
        query (str): The search keyword for the image.

    Returns:
        str or None: The URL of the image, or None if not found.
    """
    url = f"https://api.unsplash.com/photos/random?query={query}&client_id={UNSPLASH_ACCESS_KEY}"
    resp = requests.get(url)
    if resp.status_code == 200:
        data = resp.json()
        return data['urls']['small']
    return None

def is_story(reply):
    """
    Determines if the assistant's reply is a story based on certain phrases.

    Args:
        reply (str): The assistant's reply.

    Returns:
        bool: True if the reply is likely a story, False otherwise.
    """
    story_phrases = [
        'once upon a time',
        'long ago',
        'in a faraway land',
        'a long time ago',
        'in a magical world',
        'story'
    ]
    reply_lower = reply.lower()
    return any(phrase in reply_lower for phrase in story_phrases)

def get_story_keyword(story_text):
    """
    Extracts the main subject or keyword from a story for image search using OpenAI.

    Args:
        story_text (str): The story text.

    Returns:
        str: The extracted keyword(s) for image search.
    """
    client = openai.OpenAI(api_key=API_KEY)
    prompt = (
        "Extract the main subject or keyword (one or two words, lowercase, no punctuation) "
        "from this story for an image search. Only return the keyword(s), nothing else.\n\n"
        f"Story: {story_text}\n\nKeyword:"
    )
    try:
        response = client.chat.completions.create(
            model="gpt-3.5-turbo",
            messages=[{"role": "user", "content": prompt}],
            temperature=0,
            max_tokens=5,
        )
        keyword = response.choices[0].message.content.strip().split('\n')[0]
        return keyword
    except Exception as e:
        return "story"

@app.route("/", methods=["GET", "POST"])
def index():
    if request.method == "GET":
        #session["chat_history"] = [INITIAL_SYSTEM_PROMPT]
        child_id = session.get("child_id", "default_child")
        last_goal = get_last_session_goal(child_id)
        if last_goal:
            session["session_goal"] = last_goal
        else:
            session["session_goal"] = "Encourage the child to talk more and feel comfortable expressing themselves."
        session["session_reflected"] = False
        profile = load_child_profile(child_id)
        session["child_profile"] = profile
        system_prompt = {
            "role": "system",
            "content": build_system_prompt(profile)
        }
        session["chat_history"] = [system_prompt]
        # Add assistant intro message
        intro_message = {
            "role": "assistant",
            "content": (
                "Hi! I'm Sonu, I love to listen and tell little stories. "
                "What did you play with today?"
            )
        }
        session["chat_history"].append(intro_message)
    if "chat_history" not in session:
        session["chat_history"] = [INITIAL_SYSTEM_PROMPT]
    chat_history = session["chat_history"]

    if request.method == "POST":
        user_input = request.form.get("user_input", "").strip()
        session_id = session.get("session_id")
        if not session_id:
            import uuid
            session_id = str(uuid.uuid4())
            session["session_id"] = session_id

        if user_input:
            chat_history.append({"role": "user", "content": user_input})
            add_to_memory_buffer({"role": "user", "content": user_input})
            session["chat_history"] = chat_history
            save_message(session_id, "user", user_input)
            child_id = session.get("child_id", "default_child")
            save_conversation_log(child_id, user_message=user_input, bot_response=None)

            profile = session.get("child_profile")
            messages = build_messages(profile, user_input)
            reply = get_bot_reply(messages)
            if is_story(reply):
                keyword = get_story_keyword(reply)
                image_url = fetch_story_image(keyword)
            else:
                image_url = None
            
            correctness = detect_answer_correctness(reply)
            if correctness is not None:
                update_skill_performance(child_id, correctness)
            print(correctness)
            chat_history.append({"role": "assistant", "content": reply, "image_url": image_url})
            add_to_memory_buffer({"role": "assistant", "content": reply})
            save_message(session_id, "assistant", reply, image_url)
            save_conversation_log(child_id, user_message=None, bot_response=reply)
            session["chat_history"] = chat_history
            
        # --- Self-reflection and goal adaptation ---
        plan = session.get("plan", ["warmup", "story", "math", "reflection"])
        if not plan or plan == []:
            if not session.get("session_reflected"):
                chat_history = session.get("chat_history", [])
                child_id = session.get("child_id", "default_child")
                reflection, next_goal = generate_session_reflection(chat_history)
                save_session_reflection(child_id, reflection, next_goal)
                session["session_reflected"] = True
            
        return jsonify({"chat_history": chat_history})

    # Find the latest assistant message with an image_url
    bot_image_url = None
    for msg in reversed(chat_history):
        if msg.get("role") == "assistant" and msg.get("image_url"):
            bot_image_url = msg["image_url"]
            break

    return render_template(
        "dashboard.html",
        chat_history=chat_history,
        tts_engine=session.get("tts_engine", "browser"),
        bot_image_url=bot_image_url
    )

@app.route("/face_detect", methods=["POST"])
def face_detect():
    img_data = request.files["image"].read()
    img_b64, face_count = detect_faces(img_data)
    return jsonify({"img_b64": img_b64, "face_count": face_count})

@app.route("/hand_detect", methods=["POST"])
def hand_detect():
    img_data = request.files["image"].read()
    img_b64, hand_count = detect_hands(img_data)
    return jsonify({"img_b64": img_b64, "hand_count": hand_count})

@app.route("/voice", methods=["POST"])
def voice():
    """
    Handle POST requests for voice chat.
    - Receives audio from the user.
    - Converts audio to text (transcript).
    - Updates chat history and memory buffer.
    - Gets a bot reply using the current child profile and chat context.
    - Tracks skill performance (adaptive difficulty).
    - Returns transcript, reply, and updated chat history as JSON.
    """
    # Read audio file from the request
    audio = request.files["audio"].read()
    # Convert audio to text using OpenAI Whisper
    transcript = speech_to_text(audio)
    
    # Retrieve chat history from session or initialize with system prompt
    chat_history = session.get("chat_history", [INITIAL_SYSTEM_PROMPT])
    # Add user's transcript to chat history and memory buffer
    chat_history.append({"role": "user", "content": transcript})
    add_to_memory_buffer({"role": "user", "content": transcript})
    session["chat_history"] = chat_history

    # Get child ID (default if not set)
    child_id = session.get("child_id", "default_child")
    # Log the user's message in conversation logs
    save_conversation_log(child_id, user_message=transcript, bot_response=None)

    # Get the latest child profile for personalized prompt
    profile = session.get("child_profile")
    # Build messages for the bot using profile and current input
    messages = build_messages(profile, transcript)
    # Get the assistant's reply from OpenAI
    reply = get_bot_reply(messages)

    # Detect if the assistant's reply indicates a correct or incorrect answer
    correctness = detect_answer_correctness(reply)
    if correctness is not None:
        # Update skill performance for adaptive difficulty
        update_skill_performance(child_id, correctness)
    print(correctness)  # For debugging: print if answer was correct/incorrect

    # Add assistant's reply to chat history and memory buffer
    chat_history.append({"role": "assistant", "content": reply})
    add_to_memory_buffer({"role": "assistant", "content": reply})
    # Log the assistant's reply in conversation logs
    save_conversation_log(child_id, user_message=None, bot_response=reply)
    session["chat_history"] = chat_history

    # Get TTS settings from session (if needed for frontend)
    tts_voice = session.get("tts_voice", "en-US")
    # You can use these variables to select the TTS engine and voice
    # For example:
    # return jsonify({... , "audio_url": audio_url})

    # Return the transcript, bot reply, and updated chat history as JSON
    return jsonify({"transcript": transcript, "reply": reply, "chat_history": chat_history})

@app.route("/settings", methods=["POST"])
def settings():
    tts_voice = request.form.get("tts_voice", "en-US")
    tts_engine = request.form.get("tts_engine", "browser")
    session["tts_voice"] = tts_voice
    session["tts_engine"] = tts_engine
    return '', 204  # No redirect, just success

@app.route("/clear_chat", methods=["POST"])
def clear_chat():
    session["chat_history"] = [INITIAL_SYSTEM_PROMPT]
    return '', 204

@app.route("/elevenlabs_tts", methods=["POST"])
def elevenlabs_tts():
    text = request.json.get("text", "")
    if not text:
        return jsonify({"error": "No text provided"}), 400
    audio_stream = text_to_speech_stream(text)
    return send_file(audio_stream, mimetype="audio/mp3")

def save_message(session_id, role, message, image_url=None):
    """
    Saves a chat message to the chat_history database.

    Args:
        session_id (str): The session identifier.
        role (str): The role of the sender ('user' or 'assistant').
        message (str): The message content.
        image_url (str, optional): URL of any image associated with the message.
    """
    with get_db() as db:
        db.execute(
            "INSERT INTO chat_history (session_id, role, message, image_url) VALUES (?, ?, ?, ?)",
            (session_id, role, message, image_url)
        )

def load_chat_history(session_id):
    """
    Loads the chat history for a given session from the database.

    Args:
        session_id (str): The session identifier.

    Returns:
        list: List of message dicts with role, content, and image_url.
    """
    with get_db() as db:
        rows = db.execute(
            "SELECT role, message, image_url FROM chat_history WHERE session_id = ? ORDER BY id",
            (session_id, )
        ).fetchall()
        return [
            {"role": row["role"], "content": row["message"], "image_url": row["image_url"]}
            for row in rows
        ]

def recognize_faces(image_bytes, model_file="face_recognition/known_faces.pkl", tolerance=0.5):
    """
    Recognizes known faces in an image using face_recognition library.

    Args:
        image_bytes (bytes): The image data in bytes.
        model_file (str): Path to the pickle file containing known face encodings.
        tolerance (float): How much distance between faces to consider it a match.

    Returns:
        tuple: (base64-encoded image string with rectangles and names, list of recognized names)
    """
    # Load known faces
    with open(model_file, "rb") as f:
        data = pickle.load(f)
    known_encodings = data["encodings"]
    known_names = data["names"]

    # Decode uploaded image
    nparr = np.frombuffer(image_bytes, np.uint8)
    img = cv2.imdecode(nparr, cv2.IMREAD_COLOR)
    rgb_img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)

    # Find faces and encodings
    face_locations = face_recognition.face_locations(rgb_img)
    face_encodings = face_recognition.face_encodings(rgb_img, face_locations)

    names = []
    for (top, right, bottom, left), encoding in zip(face_locations, face_encodings):
        matches = face_recognition.compare_faces(known_encodings, encoding, tolerance=tolerance)
        name = "Unknown"
        if True in matches:
            matched_idxs = [i for i, match in enumerate(matches) if match]
            name_counts = {}
            for idx in matched_idxs:
                name_counts[known_names[idx]] = name_counts.get(known_names[idx], 0) + 1
            name = max(name_counts, key=name_counts.get)
        names.append(name)
        # Draw rectangle and name on the image
        cv2.rectangle(img, (left, top), (right, bottom), (0, 255, 0), 2)
        cv2.putText(img, name, (left, top - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)

    # Encode the image to base64 for easy transport
    _, buffer = cv2.imencode('.jpg', img)
    img_b64 = base64.b64encode(buffer).decode('utf-8')
    return img_b64, names

@app.route("/face_recognize", methods=["POST"])
def face_recognize():
    img_data = request.files["image"].read()
    img_b64, names = recognize_faces(img_data)
    # Only keep the first recognized name (or "Unknown" if none)
    if names and names[0] != "Unknown":
        session["recognized_names"] = [names[0]]
        session["child_id"] = names[0]  # Use name as child_id
        # Ensure child profile exists in DB
        conn = sqlite3.connect("child_data.db")
        cur = conn.cursor()
        cur.execute("INSERT OR IGNORE INTO child_profiles (child_id, name) VALUES (?, ?)", (names[0], names[0]))
        conn.commit()
        conn.close()
        # --- PATCH: Reload profile in session ---
        profile = load_child_profile(names[0])
        session["child_profile"] = profile
    else:
        session["recognized_names"] = []
    return jsonify({"img_b64": img_b64, "names": names})

def load_child_profile(child_id, db_path="child_data.db"):
    """
    Loads the child's profile and preferences from the database.

    Args:
        child_id (str): The child's unique identifier.
        db_path (str): Path to the child data database.

    Returns:
        dict: The child's profile including name, age, mood, and all preferences by category.
    """
    import sqlite3
    conn = sqlite3.connect(db_path)
    cur = conn.cursor()
    cur.execute("SELECT name, age, mood FROM child_profiles WHERE child_id = ?", (child_id,))
    row = cur.fetchone()
    if not row:
        return None
    name, age, mood = row

    # Fetch all preferences and organize by category
    cur.execute("SELECT category, value FROM child_preferences WHERE child_id = ?", (child_id,))
    prefs = {}
    for category, value in cur.fetchall():
        prefs.setdefault(category, []).append(value)
    conn.close()

    # Build the profile dict
    profile = {"name": name, "age": age, "mood": mood}
    profile.update(prefs)
    return profile

def build_system_prompt(profile):
    if not profile:
        return INITIAL_SYSTEM_PROMPT["content"]

    # Use defaults if missing or None
    name = profile.get('name') or 'your friend'
    age = profile.get('age')
    mood = profile.get('mood')
    # If mood is a list, join it; if None, use 'happy'
    if isinstance(mood, list):
        mood_str = ', '.join(set(mood))
    elif mood:
        mood_str = str(mood)
    else:
        mood_str = 'happy'

    # Dynamic, personalized context
    prompt_lines = [
        f"You are talking to a {age if age not in (None, '', 'None') else 'young'}-year-old child named {name}.",
        f"They are usually {mood_str}.",
    ]

    # Merge and deduplicate all list-type fields except name, age, mood
    for key in profile:
        if key in ["name", "age", "mood"]:
            continue
        value = profile[key]
        if isinstance(value, list) and value:
            unique_vals = list(dict.fromkeys(value))
            prompt_lines.append(f"Their {key.replace('_', ' ')}: {', '.join(unique_vals)}.")
        elif isinstance(value, str) and value:
            prompt_lines.append(f"Their {key.replace('_', ' ')}: {value}.")

    # If age is missing, ask politely
    if age in (None, '', 'None'):
        prompt_lines.append("If you don't know the child's age, gently ask: 'How old are you?' or 'Can you tell me your age?'.")
    
    child_id = session.get("child_id", "default_child")
    skill_level = get_skill_level(child_id)
    if skill_level == "harder":
        prompt_lines.append("If the child answers correctly, try to ask a slightly harder math or language question next time.")
    elif skill_level == "easier":
        prompt_lines.append("If the child is struggling, make your next math or language question a bit easier and encourage them gently.")
    
    prompt_lines.append(
        "Never repeat the same joke, story, or math problem you have already told in this session. "
        "Try to keep each response new and interesting for the child."
    )
    prompt_lines.append(
        "Whenever possible, mention the child's name, age, or favorite things in your replies. "
        "Bring up their favorite topics, toys, or activities to make the conversation more engaging and personal. "
        "If the child likes certain things (e.g., dinosaurs, drawing), ask about them or include them in your stories and questions."
    )

    recognized_names = session.get("recognized_names")
    if recognized_names:
        prompt_lines.append(f"The following person is present: {recognized_names[0]}.")
    # Remove any logic for new_faces

    # Merge with static system prompt instructions
    prompt_lines.append(INITIAL_SYSTEM_PROMPT["content"])
    plan = session.get("plan", ["warmup", "story", "math", "reflection"])
    if not plan or plan == []:
        child_id = session.get("child_id", "default_child")
        import sqlite3
        conn = sqlite3.connect("child_data.db")
        cursor = conn.cursor()
        cursor.execute("""
            SELECT reflection, next_goal FROM session_reflections
            WHERE child_id = ?
            ORDER BY timestamp DESC LIMIT 1
        """, (child_id,))
        row = cursor.fetchone()
        conn.close()
        if row:
            prompt_lines.append(f"Session reflection: {row[0]}")
            prompt_lines.append(f"Suggested goal for next session: {row[1]}")
        else:
            prompt_lines.append("Reflect on this session: Did the child talk more? What did they enjoy? Suggest a goal for the next session.")

    return " ".join(prompt_lines)

def build_messages(profile, user_input, max_turns=MEMORY_BUFFER_SIZE):
    system_prompt = build_system_prompt(profile)
    messages = [{"role": "system", "content": system_prompt}]
    chat_history = [
        msg for msg in session.get("chat_history", [])
        if msg.get("role") != "system"
    ][-max_turns:]
    messages += chat_history
    messages.append({"role": "user", "content": user_input})
    return messages

def save_conversation_log(child_id, user_message=None, bot_response=None, db_path="child_data.db"):
    """
    Saves a conversation log entry to the conversation_logs table in the database.

    Args:
        child_id (str): The child's unique identifier.
        user_message (str, optional): The user's message.
        bot_response (str, optional): The assistant's response.
        db_path (str): Path to the child data database.
    """
    import sqlite3
    from datetime import datetime
    conn = sqlite3.connect(db_path)
    cursor = conn.cursor()
    cursor.execute(
        "INSERT INTO conversation_logs (child_id, timestamp, user_message, bot_response) VALUES (?, ?, ?, ?)",
        (child_id, datetime.now(), user_message, bot_response)
    )
    conn.commit()
    conn.close()

def extract_child_memory(messages, api_key=API_KEY, model="gpt-3.5-turbo"):
    """
    Calls OpenAI to extract key facts about the child from a list of messages.
    Returns a JSON string with likes, dislikes, mood, personality traits, fears, skills, or special events.

    Args:
        messages (list): List of message dicts (role/content) for the conversation.
        api_key (str): OpenAI API key.
        model (str): OpenAI model name.

    Returns:
        str: JSON string with extracted child memory.
    """
    # Convert messages to a readable chat transcript
    chat_text = ""
    for msg in messages:
        if msg["role"] == "user":
            chat_text += f'Child: {msg["content"]}\n'
        elif msg["role"] == "assistant":
            chat_text += f'Assistant: {msg["content"]}\n'

    highlight_prompt = f"""
You are a memory extractor AI.
Given the following chat between a child and an assistant, extract key facts about the child in JSON format.
Include age, likes, dislikes, mood, personality traits, fears, skills, or special events.

Example format:
{{
  "age": 6,
  "likes": ["drawing", "dinosaurs"],
  "mood": "shy",
  "fears": ["dark"],
  "skills": ["tied shoes"]
}}

Conversation:
{chat_text}
"""

    client = openai.OpenAI(api_key=api_key)
    try:
        response = client.chat.completions.create(
            model=model,
            messages=[{"role": "system", "content": highlight_prompt}],
            temperature=0,
            max_tokens=256,
        )
        return response.choices[0].message.content.strip()
    except Exception as e:
        return f"Error: {e}"

def add_to_memory_buffer(message):
    """
    Adds a message to the session memory buffer.
    When the buffer reaches MEMORY_BUFFER_SIZE, extracts child memory and updates the database.

    Args:
        message (dict): The message to add (should have 'role' and 'content').
    """
    buffer = session.get("memory_buffer", [])
    buffer.append(message)
    session["memory_buffer"] = buffer
    if len(buffer) >= MEMORY_BUFFER_SIZE:
        # Call extract_child_memory and clear buffer
        memory_json = extract_child_memory(buffer)
        child_id = session.get("child_id", "default_child")
        update_child_memory(child_id, memory_json)
        session["memory_buffer"] = []

def update_child_memory(child_id, memory_json, db_path="child_data.db"):
    """
    Updates child_profiles and child_preferences tables from extracted memory.

    Args:
        child_id (str): The child's unique identifier.
        memory_json (str): JSON string with extracted child memory.
        db_path (str): Path to the child data database.
    """
    try:
        memory = json.loads(memory_json)
    except Exception as e:
        print("Error parsing memory_json:", e)
        return

    conn = sqlite3.connect(db_path)
    cursor = conn.cursor()

    # --- Update child_profiles ---
    profile_fields = {}
    for field in ["name", "age", "mood"]:
        if field in memory:
            profile_fields[field] = memory[field]
    if profile_fields:
        # Check if child_id exists
        cursor.execute("SELECT child_id FROM child_profiles WHERE child_id = ?", (child_id,))
        if cursor.fetchone():
            # Update
            set_clause = ", ".join([f"{k} = ?" for k in profile_fields])
            values = list(profile_fields.values()) + [child_id]
            cursor.execute(f"UPDATE child_profiles SET {set_clause} WHERE child_id = ?", values)
        else:
            # Insert
            cursor.execute(
                "INSERT INTO child_profiles (child_id, name, age, mood) VALUES (?, ?, ?, ?)",
                (
                    child_id,
                    profile_fields.get("name"),
                    profile_fields.get("age"),
                    profile_fields.get("mood"),
                ),
            )

    # --- Insert all fields into child_preferences, upsert to avoid duplicates ---
    for key, value in memory.items():
        if key in ["name", "age"]:  # skip these for preferences
            continue
        if isinstance(value, list):
            for v in value:
                cursor.execute(
                    "INSERT INTO child_preferences (child_id, category, value) VALUES (?, ?, ?) "
                    "ON CONFLICT(child_id, category, value) DO UPDATE SET value=excluded.value",
                    (child_id, key, str(v)),
                )
        else:
            cursor.execute(
                "INSERT INTO child_preferences (child_id, category, value) VALUES (?, ?, ?) "
                "ON CONFLICT(child_id, category, value) DO UPDATE SET value=excluded.value",
                (child_id, key, str(value)),
            )

    conn.commit()
    conn.close()

def update_skill_performance(child_id, correct, db_path="child_data.db"):
    """
    Updates the skill_performance table with the result of the child's answer.

    Args:
        child_id (str): The child's unique identifier.
        correct (bool): True if the answer was correct, False otherwise.
        db_path (str): Path to the child data database.
    """
    import sqlite3
    conn = sqlite3.connect(db_path)
    cursor = conn.cursor()
    cursor.execute("""
        CREATE TABLE IF NOT EXISTS skill_performance (
            child_id TEXT,
            correct_count INTEGER DEFAULT 0,
            incorrect_count INTEGER DEFAULT 0,
            PRIMARY KEY(child_id)
        )
    """)
    # Try to update, else insert
    cursor.execute("SELECT correct_count, incorrect_count FROM skill_performance WHERE child_id = ?", (child_id,))
    row = cursor.fetchone()
    if row:
        correct_count, incorrect_count = row
        if correct:
            correct_count += 1
        else:
            incorrect_count += 1
        cursor.execute("UPDATE skill_performance SET correct_count = ?, incorrect_count = ? WHERE child_id = ?",
                       (correct_count, incorrect_count, child_id))
    else:
        cursor.execute("INSERT INTO skill_performance (child_id, correct_count, incorrect_count) VALUES (?, ?, ?)",
                       (child_id, int(correct), int(not correct)))
    conn.commit()
    conn.close()

def get_skill_level(child_id, db_path="child_data.db"):
    """
    Determines the child's skill level (normal, harder, easier) based on performance.

    Args:
        child_id (str): The child's unique identifier.
        db_path (str): Path to the child data database.

    Returns:
        str: "harder", "easier", or "normal" based on performance.
    """
    import sqlite3
    conn = sqlite3.connect(db_path)
    cursor = conn.cursor()
    # Ensure the table exists
    cursor.execute("""
        CREATE TABLE IF NOT EXISTS skill_performance (
            child_id TEXT,
            correct_count INTEGER DEFAULT 0,
            incorrect_count INTEGER DEFAULT 0,
            PRIMARY KEY(child_id)
        )
    """)
    cursor.execute("SELECT correct_count, incorrect_count FROM skill_performance WHERE child_id = ?", (child_id,))
    row = cursor.fetchone()
    conn.close()
    if not row:
        return "normal"
    correct, incorrect = row
    if correct >= incorrect + 3:
        return "harder"
    elif incorrect >= correct + 3:
        return "easier"
    else:
        return "normal"

def detect_answer_correctness(reply):
    """
    Heuristic to detect if the assistant's reply indicates a correct or incorrect answer.
    Returns True if correct, False if incorrect, or None if unknown.

    Args:
        reply (str): The assistant's reply.

    Returns:
        bool or None: True if correct, False if incorrect, None if unknown.
    """
    # Simple heuristic: adjust as needed or use OpenAI for classification
    if "that's right" in reply.lower() or "correct" in reply.lower():
        return True
    elif "not quite" in reply.lower() or "try again" in reply.lower() or "almost" in reply.lower():
        return False
    return None  # Unknown

def ensure_session_reflections_table(db_path="child_data.db"):
    """
    Ensures the session_reflections table exists in the specified database.
    Creates the table if it does not exist.
    Args:
        db_path (str): Path to the child data database.
    """
    import sqlite3
    conn = sqlite3.connect(db_path)
    cursor = conn.cursor()
    cursor.execute("""
        CREATE TABLE IF NOT EXISTS session_reflections (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            child_id TEXT,
            timestamp DATETIME,
            reflection TEXT,
            next_goal TEXT
        )
    """)
    conn.commit()
    conn.close()

ensure_session_reflections_table()

def save_session_reflection(child_id, reflection, next_goal, db_path="child_data.db"):
    """
    Saves a session reflection and next goal for a child to the database.
    Args:
        child_id (str): The child's unique identifier.
        reflection (str): The session reflection text.
        next_goal (str): The suggested goal for the next session.
        db_path (str): Path to the child data database.
    """
    import sqlite3, datetime
    conn = sqlite3.connect(db_path)
    cursor = conn.cursor()
    cursor.execute(
        "INSERT INTO session_reflections (child_id, timestamp, reflection, next_goal) VALUES (?, ?, ?, ?)",
        (child_id, datetime.datetime.now(), reflection, next_goal)
    )
    conn.commit()
    conn.close()

def get_last_session_goal(child_id, db_path="child_data.db"):
    """
    Retrieves the most recent next goal for a child from the session_reflections table.
    Args:
        child_id (str): The child's unique identifier.
        db_path (str): Path to the child data database.
    Returns:
        str or None: The last session's next goal, or None if not found.
    """
    import sqlite3
    conn = sqlite3.connect(db_path)
    cursor = conn.cursor()
    cursor.execute("""
        SELECT next_goal FROM session_reflections
        WHERE child_id = ?
        ORDER BY timestamp DESC LIMIT 1
    """, (child_id,))
    row = cursor.fetchone()
    conn.close()
    if row and row[0]:
        return row[0]
    return None

def generate_session_reflection(chat_history, api_key=API_KEY, model="gpt-3.5-turbo"):
    """
    Uses OpenAI to generate a session reflection and suggest a next goal based on chat history.
    Args:
        chat_history (list): List of message dicts (role/content) for the conversation.
        api_key (str): OpenAI API key.
        model (str): OpenAI model name.
    Returns:
        tuple: (reflection, next_goal) as strings.
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

if __name__ == "__main__":
    import logging
    log = logging.getLogger('werkzeug')
    log.setLevel(logging.ERROR)
    app.run(debug=True)