from flask import Flask, render_template, request, jsonify
import json
import logging

from model import ChatBot, ChatBotFunctionalityHelper, ChatBotHelper_Improved

# Set up Flask app
app = Flask(__name__)

# Configure logging
logging.basicConfig(level=logging.INFO)

# Load vocab once at startup
with open("model_training_environment/vocab.json", "r") as vocab_file:
    vocab = json.load(vocab_file)

# Initialize helpers and model once
functionality = ChatBotFunctionalityHelper()
chatbothelper = ChatBotHelper_Improved(vocab)
mymodel = ChatBot('model_training_environment/chatbot_model.h5', chatbothelper, functionality)


@app.route('/')
def home():
    return render_template('main.html')


@app.route('/process_input', methods=['POST'])
def process_input():
    try:
        data = request.get_json()

        # Validate input
        user_input = data.get('input', '')
        if not isinstance(user_input, str) or not user_input.strip():
            return jsonify({'error': 'Invalid input'}), 400  # HTTP 400 Bad Request

        # Get model prediction
        processed_result = mymodel.predict(user_input)

        return jsonify({'result': processed_result}), 200  # HTTP 200 OK

    except Exception as e:
        logging.exception("Error during input processing")
        return jsonify({'error': 'An internal error occurred'}), 500  # HTTP 500 Internal Server Error


@app.route('/health')
def health_check():
    return jsonify({'status': 'ok'}), 200


#if __name__ == '__main__':
 #   app.run(debug=True)
