import threading
from tkinter import *
import speech_recognition as sr
import requests
from datetime import datetime
import pyttsx3
import nltk
import json
from nltk.chat.util import Chat, reflections

nltk.download('punkt')
nltk.download('averaged_perceptron_tagger')
nltk.download('maxent_ne_chunker')
nltk.download('words')


# Initialize recognizer and microphone instances
recognizer = sr.Recognizer()
microphone = sr.Microphone()

# Initialize speech engine
engine = pyttsx3.init()

# Load chat pairs from corrected JSON file
with open('fixed_data1.json') as f:
    pairs_dict = json.load(f)

# Create chat pairs for nltk
pairs = [(k, v) for k, v in pairs_dict.items()]

# Create chat instance
chat = Chat(pairs, reflections)
# Function to get nltk response
from nltk import word_tokenize, pos_tag, ne_chunk
from nltk.stem import PorterStemmer


# Function to get response using NLTK processing
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity
import numpy as np


def get_response(user_input, data):
    # Initialize a Tfidf Vectorizer
    vectorizer = TfidfVectorizer(stop_words='english')

    # Extract keys from data
    keys = list(data.keys())

    # Add the user input to keys for vectorization
    keys.append(user_input)

    # Vectorize the keys
    vectors = vectorizer.fit_transform(keys)

    # Calculate cosine similarity of user input with all keys
    similarity_scores = cosine_similarity(vectors[-1], vectors[:-1])

    # Find the index of the most similar key
    most_similar_idx = np.argmax(similarity_scores)

    # Return the response associated with the most similar key
    return data[keys[most_similar_idx]][0]

def get_nltk_response(query):
    # Tokenization
    tokens = word_tokenize(query)

    # Stemming
    stemmer = PorterStemmer()
    stems = [stemmer.stem(token) for token in tokens]

    # Part-of-Speech Tagging
    pos_tags = pos_tag(tokens)

    # Named Entity Recognition
    named_entities = ne_chunk(pos_tags)
    response = chat.respond(query)

    if response is None:
        # Check for topic-related queries and return appropriate responses
        return get_response(query, pairs_dict)
    elif response == "getCurrentTime":
        return get_current_time()
    elif response == "getCurrentDate":
        return get_current_date()
    elif "getWeatherData" in response:
        location = response.split(',')[1]
        return get_weather_data(location)  # Ensure you define this function
    elif response == "getNewsData":
        return get_news_data()  # Ensure you define this function
    else:
        return response


# Function to get current time
def get_current_time():
    now = datetime.now()
    current_time = now.strftime("%H:%M:%S")
    return "Current Time = " + current_time

# Function to get current date
def get_current_date():
    now = datetime.now()
    current_date = now.strftime("%d/%m/%Y")
    return "Current Date = " + current_date

# Function to get weather data
def get_weather_data(city_name, api_key):
    base_url = "http://api.openweathermap.org/data/2.5/weather"
    complete_url = base_url + "?q=" + city_name + "&appid=" + api_key
    response = requests.get(complete_url)
    weather_data = response.json()

    if weather_data["cod"] != "404":
        main_data = weather_data["main"]
        current_temperature = main_data["temp"]
        current_pressure = main_data["pressure"]
        current_humidity = main_data["humidity"]
        weather_description = weather_data["weather"][0]["description"]

        weather_response = "Temperature: " + str(current_temperature) + \
                           "\nAtmospheric pressure: " + str(current_pressure) + \
                           "\nHumidity: " + str(current_humidity) + \
                           "\nDescription: " + str(weather_description)

        return weather_response

    else:
        return "City Not Found!"

# Function to get news data
def get_news_data(api_key):
    base_url = "https://gnews.io/api/v4/search?q=example&token=" + api_key
    response = requests.get(base_url)
    news_data = response.json()

    if news_data["totalArticles"] > 0:
        articles = news_data["articles"]
        news_response = ""

        for article in articles:
            news_response += "Title: " + article["title"] + "\nDescription: " + article["description"] + "\n\n"

        return news_response

    else:
        return "No News Found!"

# Function to get wikipedia data
import wikipediaapi


def get_wikipedia_data(topic):
    wiki_wiki = wikipediaapi.Wikipedia('en')
    page_py = wiki_wiki.page(topic)

    if not page_py.exists():
        return "No article found for " + topic

    return page_py.summary[0:60]


# Function to recognize speech from microphone
def recognize_speech_from_mic(recognizer, microphone):
    with microphone as source:
        recognizer.adjust_for_ambient_noise(source)
        audio = recognizer.listen(source)

    response = {
        "success": True,
        "error": None,
        "transcription": None
    }

    try:
        response["transcription"] = recognizer.recognize_google(audio)
    except sr.RequestError:
        response["success"] = False
        response["error"] = "API unavailable"
    except sr.UnknownValueError:
        response["error"] = "Unable to recognize speech"

    return response

# Function to start recognition
def start_recognition():
    global text_area

    while True:
        text_area.insert(END, "Assistant: Recognizing...\n")
        root.update()

        query = None

        while not query:
            response = recognize_speech_from_mic(recognizer, microphone)

            if response["error"]:
                text_area.insert(END, "Assistant: I'm sorry, I couldn't understand that. Please try again.\n")
            else:
                query = response["transcription"]

        text_area.insert(END, "You: " + query + "\n")
        bot_response = get_nltk_response(query)

        if not bot_response:
            if "weather" in query:
                bot_response = get_weather_data("Kathmandu", "4700a4dbebcf7c9ad9d1578bfcb6ca3b")
            elif "news" in query:
                bot_response = get_news_data("30314119981e8d0fe457a056978b6273")
            elif "wikipedia" in query:
                _, topic = query.split(' ', 1)
                bot_response = get_wikipedia_data(topic)
            elif "time" in query:
                bot_response = get_current_time()
            elif "date" in query:
                bot_response = get_current_date()
            else:
                bot_response = "I'm sorry, I didn't understand that."

        text_area.insert(END, "Bot: " + bot_response + "\n")
        engine.say(bot_response)
        engine.runAndWait()

        if "exit" in query:
            break

def end_application():
    root.quit()

root = Tk()
root.title("Python Project")

frame = Frame(root)
scrollbar = Scrollbar(frame)
scrollbar.pack(side=RIGHT, fill=Y)

text_area = Text(frame, wrap=WORD, yscrollcommand=scrollbar.set)
text_area.pack()
frame.pack()

scrollbar.config(command=text_area.yview)

button_frame = Frame(root)
button_frame.pack()

def start_threaded_recognition():
    recognition_thread = threading.Thread(target=start_recognition)
    recognition_thread.start()

start_button = Button(button_frame, text="Start", command=start_threaded_recognition)
start_button.pack(side=LEFT)

end_button = Button(button_frame, text="End", command=end_application)
end_button.pack(side=LEFT)

root.mainloop()
