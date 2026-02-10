import gradio as gr
from transformers import pipeline
import re

# --- STEP 1: LOAD MODELS ---
# Sentiment for mood detection, DialoGPT for conversation
sentiment_task = pipeline("sentiment-analysis", model="michellejieli/emotion_text_classifier")
chatbot_model = pipeline("text-generation", model="microsoft/DialoGPT-medium")

# --- LAYER 1 & 2 DATA STRUCTURES ---
conversation_history = []  # Contextual Memory
MAX_HISTORY = 5           # Keep the last 5 exchanges to save RAM

# --- LAYER 3: PRIVACY SCRUBBER ---
def anonymize_input(text):
    # Simple regex to catch common Name/Phone patterns (Basic version)
    text = re.sub(r'\b[A-Z][a-z]+\b', '[USER_NAME]', text) # Capitalized words
    text = re.sub(r'\d{10}', '[PHONE_NUMBER]', text)      # 10-digit numbers
    return text

# --- LAYER 2: CRISIS RESOURCES ---
HELP_RESOURCES = """
🚨 It sounds like you're going through a lot. Please remember you're not alone:
- Suicide & Crisis Lifeline: Call or Text 988 (USA)
- Crisis Text Line: Text HOME to 741741
- International: findahelpline.com
"""

# --- THE SYSTEMATIC CORE ---
def serenity_ai(user_input):
    global conversation_history
    
    # 1. Anonymize (Layer 3)
    clean_input = anonymize_input(user_input)
    
    # 2. Crisis Check (Layer 2)
    crisis_keywords = ["suicide", "hurt myself", "end it all", "kill myself", "depressed"]
    if any(word in clean_input.lower() for word in crisis_keywords):
        return f"I'm hearing that things are very difficult right now. {HELP_RESOURCES}"

    # 3. Sentiment Analysis
    emotion = sentiment_task(clean_input)[0]['label']
    
    # 4. Memory Integration (Layer 1)
    # Combine history into one long string for the model to "read"
    context = " ".join(conversation_history) + " " + clean_input
    
    # 5. Generate Response
    response = chatbot_model(context, max_length=100, pad_token_id=50256)[0]['generated_text']
    
    # Clean response (remove the context part)
    final_reply = response.replace(context, "").strip()
    
    # 6. Update History (Layer 1)
    conversation_history.append(f"User: {clean_input}")
    conversation_history.append(f"AI: {final_reply}")
    if len(conversation_history) > MAX_HISTORY * 2:
        conversation_history = conversation_history[-MAX_HISTORY*2:]

    return f"🌟 [Mood Detected: {emotion}]\n\n{final_reply}"

# --- GRADIO INTERFACE ---
demo = gr.Interface(
    fn=serenity_ai, 
    inputs=gr.Textbox(label="How are you feeling today?", placeholder="Type here..."),
    outputs=gr.Textbox(label="Serenity AI"),
    title="Serenity AI: Emotional Support Companion",
    description="A safe space to talk. (Note: I am an AI, not a therapist. If in crisis, use the help resources.)"
)

demo.launch(debug=True)