import streamlit as st
import speech_recognition as sr
import joblib

# Load your custom sentiment analysis model and vectorizer
model = joblib.load('logistic_regression_model.pkl')
vectorizer = joblib.load('countVectorizer.pkl')

# Function to process audio input
def recognize_speech():
    recognizer = sr.Recognizer()
    with sr.Microphone() as source:
        st.write("🎤 Listening for 1 minute...")
        recognizer.adjust_for_ambient_noise(source, duration=1)
        audio = recognizer.listen(source, timeout=60, phrase_time_limit=60)

    try:
        text = recognizer.recognize_google(audio)
        st.write(f"📝 Recognized Text: {text}")
        return text
    except sr.UnknownValueError:
        return "Sorry, I could not understand the audio."
    except sr.RequestError:
        return "Could not request results from Google Speech Recognition service."

# Function to map model predictions to sentiment labels
def map_sentiment(prediction):
    if prediction == 1:
        return "😊 Good", "#4CAF50"  # Green for good sentiment
    elif prediction == -1:
        return "😞 Bad", "#F44336"  # Red for bad sentiment
    else:
        return "😐 Neutral", "#9E9E9E"  # Grey for neutral sentiment

# Custom CSS for chat-style design
st.markdown("""
    <style>
        body {
            background-color: #2E3B4E;
            color: #ffffff;
            font-family: 'Segoe UI', Tahoma, Geneva, Verdana, sans-serif;
        }
        .title {
            text-align: center;
            font-size: 3rem;
            color: #FFD700;  /* Gold title */
            margin-bottom: 30px;
            text-shadow: 2px 2px 4px rgba(0, 0, 0, 0.5);
        }
        .chat-bubble {
            padding: 10px;
            border-radius: 10px;
            margin-bottom: 10px;
            width: fit-content;
            max-width: 80%;
        }
        .user-bubble {
            background-color: #007BFF;
            color: #ffffff;
            align-self: flex-end;
        }
        .bot-bubble {
            background-color: #2C2C2C;
            color: #ffffff;
            align-self: flex-start;
        }
    </style>
""", unsafe_allow_html=True)

# Streamlit application layout
st.markdown('<h1 class="title">💬 Chat or Record Sentiment Analysis</h1>', unsafe_allow_html=True)

# User choice: Chat or Record
option = st.selectbox("Choose your interaction mode:", ("Chat", "Record"))

if option == "Chat":
    # Initialize chat history if not present
    if 'history' not in st.session_state:
        st.session_state['history'] = []

    # Capture user input
    user_input = st.text_input("You:", help="Type your message here", key='user_input')

    # Process the input and generate a response
    if st.button('Send', help="Click to analyze the sentiment"):
        if user_input:
            # Append the user input to chat history
            st.session_state['history'].append(('user', user_input))
            
            # Preprocess the text and make a prediction
            processed_text = vectorizer.transform([user_input])
            prediction = model.predict(processed_text)[0]
            sentiment, color = map_sentiment(prediction)
            
            # Append the sentiment response to chat history
            st.session_state['history'].append(('bot', f"Sentiment: {sentiment}"))

    # Display chat history
    for sender, message in st.session_state['history']:
        if sender == 'user':
            st.chat_message("user").markdown(f"**You:** {message}")
        else:
            st.chat_message("bot").markdown(f"**Bot:** {message}")

elif option == "Record":
    # Handle recording and analysis
    if st.button('🎤 Record and Analyze', key='record_button', help="Click to start recording"):
        st.write('<div class="text-output">Please say something into your microphone.</div>', unsafe_allow_html=True)
        text = recognize_speech()
        st.write('<div class="text-output">You said: {}</div>'.format(text), unsafe_allow_html=True)
        
        if "Sorry" in text or "Could not" in text:
            st.write('<div class="text-output">{}</div>'.format(text), unsafe_allow_html=True)
        else:
            # Preprocess the text and make a prediction
            st.write('<div class="text-output">Processing text...</div>', unsafe_allow_html=True)
            processed_text = vectorizer.transform([text])
            prediction = model.predict(processed_text)[0]
            sentiment, color = map_sentiment(prediction)
            st.write(f'<div class="sentiment-output" style="color:{color}">Sentiment: {sentiment}</div>', unsafe_allow_html=True)
