import tensorflow as tf
import random
import requests
from nltk.corpus import stopwords
from difflib import SequenceMatcher
import numpy as np
import os
import nltk
import string
from dotenv import load_dotenv


# Chat Bot Wrapper Class
class ChatBot:

    def __init__(self, model, preprocessor, functionality):

        self.model = tf.keras.models.load_model(model)
        self.model_preprocessor = preprocessor
        self.functionality = functionality
        load_dotenv()


    def predict(self, input):

        preprocessed_input = self.model_preprocessor.nlp_preprocessing(input)

        preprocessed_input = self.model_preprocessor.embed(preprocessed_input)

        prediction = self.model.predict(np.array([preprocessed_input]))

        # Ambiguity check
        if np.sum(preprocessed_input) == 0 or np.max(prediction) < 0.30:
            return self.functionality.base_case()

        predicted_category = np.argmax(prediction, axis=1)

        match predicted_category:

            case 0:
                return self.functionality.greeting()

            case 1:
                return self.functionality.weather()

            case 2:
                return self.functionality.news()

            case 3:
                return self.functionality.motivate()

            case 4:
                return self.functionality.breakfast()


# Functionality for the Chat Bot
class ChatBotFunctionalityHelper:

    def greeting(self):
        greetings = [
            "Good morning! How can I assist you today?",
            "Hey there! What can I do for you this morning?",
            "Hello, how can I help you?",
            "Morning! Need any help getting started with your day?",
            "Hi! Ready to tackle the day? How can I assist?"
        ]
        return random.choice(greetings)

    def weather(self, location="Frankfurt"):

        API_KEY = os.getenv("API_KEY_Weather")
        if not API_KEY:
            return "Error: API key is missing."

        url = f"http://api.weatherapi.com/v1/current.json?key={API_KEY}&q={location}"

        try:
            response = requests.get(url)
            response.raise_for_status()
            data = response.json()

            weather = data['current']['condition']['text']
            wind = data['current']['wind_mph']
            temp = data['current']['feelslike_c']

            return (
                f"That is the data I found for {location}:\n"
                f"Weather: {weather}\n"
                f"Wind: {wind} mph\n"
                f"Feels Like: {temp}°C"
            )

        except requests.exceptions.RequestException as e:
            return f"Network error occurred: {e}"
        except KeyError:
            return "Weather data is not available or malformed for this location."

    def news(self):

        url = f'https://newsapi.org/v2/top-headlines?country=us&pageSize=5&apiKey={os.getenv("API_KEY_News")}'

        response = requests.get(url)
        data = response.json()

        answer = ""

        if data.get('status') == 'ok':
            articles = data.get('articles', [])
            for i, article in enumerate(articles, 1):
                answer += str(i)
                answer += " "
                answer += article['title']
                answer += "\n"
        else:
            print("Error:", data.get('message'))

        result = "Look at what I found: \n" + answer

        return result

    def motivate(self):
        motivational_quote = [
            "Success is not final, failure is not fatal: It is the courage to continue that counts. – Winston Churchill",
            "Believe you can and you're halfway there. – Theodore Roosevelt",
            "Don't watch the clock; do what it does. Keep going. – Sam Levenson",
            "Your limitation—it's only your imagination.",
            "Dream big and dare to fail. – Norman Vaughan"
            "The way to get started is to quit talking and begin doing. – Walt Disney",
            "Act as if what you do makes a difference. It does. – William James",
            "Hardships often prepare ordinary people for an extraordinary destiny. – C.S. Lewis",
            "Success usually comes to those who are too busy to be looking for it. – Henry David Thoreau",
            "The only limit to our realization of tomorrow will be our doubts of today. – Franklin D. Roosevelt"]
        introduction = ["Maybe this quote will inspire you: \n", "Think about this quote: \n",
                        "When I feel demotivated, I think about this quote: \n",
                        "When I want to feel motivated, I rememeber this quote: \n"]

        random.seed(None)

        return "".join((introduction[random.randint(0, len(introduction) - 1)],
                        motivational_quote[random.randint(0, len(motivational_quote) - 1)]))

    def recipe(self):
        recipes = []
        recipes.append({'answer': "Look at this awesome pancake recipe, I found online: \n",
                        'link': "https://www.inspiredtaste.net/24593/essential-pancake-recipe/"})
        recipes.append({'answer': "What about avocado toast? That is my favorite recipe: \n",
                        'link': "https://www.loveandlemons.com/avocado-toast-recipe/"})
        recipes.append({'answer': "What about this delicios porridge: \n",
                        'link': "https://www.allrecipes.com/recipe/73155/porridge/"})

        i = random.randint(0, 2)

        return recipes[i]['answer'] + recipes[i]['link']

    def base_case(self):
        return "I am sorry, I didn't really understand you. Could you please rephrase it?"


# Preprocessing Helper
class ChatBotHelper_Improved:

    def __init__(self, bag_of_words):
        self.bag_of_words = bag_of_words

    @staticmethod
    def nlp_preprocessing(sentence):
        stop_words = set(stopwords.words('english'))

        tokens = nltk.word_tokenize(sentence)

        lowercased_tokens = [token.lower() for token in tokens]

        filtered_tokens_stopwords = [token for token in lowercased_tokens if token not in stop_words]

        filtered_tokens = [token for token in filtered_tokens_stopwords if token not in string.punctuation]

        # Remove filler tokens
        filler_filtered_tokens = [token for token in filtered_tokens if token not in ["'", "'s", "’"]]

        return filler_filtered_tokens

    def embed(self, tokens):
        return [
            1 if any(word == token or self.lexical_similarity(word, token) > 0.75 for token in tokens)
            else 0
            for word in self.bag_of_words
        ]

    def lexical_similarity(self, word1, word2):
        # Ratio of longest common subsequence
        return SequenceMatcher(None, word1, word2).ratio()