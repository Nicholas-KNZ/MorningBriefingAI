from flask import Flask, render_template, request, jsonify
from flask_cors import CORS
import json
import logging
from model import ChatBot, ChatBotFunctionalityHelper, ChatBotHelper_Improved
import os

app = Flask(__name__)
CORS(app)
logging.basicConfig(level=logging.INFO)

# Load vocab
with open("model_training_environment/vocab.json", "r") as vocab_file:
    vocab = json.load(vocab_file)

# Initialize model
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
        user_input = data.get('input', '')
        if not user_input.strip():
            return jsonify({'error': 'Invalid input'}), 400

        processed_result = mymodel.predict(user_input)
        return jsonify({'result': processed_result})

    except Exception as e:
        logging.exception("Error during input processing")
        return jsonify({'error': str(e)}), 500

@app.route('/health')
def health_check():
    return jsonify({'status': 'ok'})

if __name__ == '__main__':
    port = int(os.environ.get("PORT", 5000))
    app.run(host="0.0.0.0", port=port)
