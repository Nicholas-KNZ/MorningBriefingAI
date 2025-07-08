from flask import Flask, render_template, request, jsonify
from model import ChatBot, ChatBotFunctionalityHelper, ChatBotHelper_Improved
import json
import numpy as np


app = Flask(__name__)

@app.route('/')
def hello_world():
    return render_template('main.html')

@app.route('/process_input', methods=['POST'])
def process_input():

    functionality = ChatBotFunctionalityHelper()

    vocab_file = open("model_training_environment/vocab.json", "r")
    vocab = json.load(vocab_file)
    vocab_file.close()
    print(type(vocab))
    chatbothelper = ChatBotHelper_Improved(vocab)

    try:
        # Get JSON data sent by frontend
        data = request.get_json()

        # Extract input field data
        user_input = data.get('input', '')  # Default to empty string if 'input' key doesn't exist

        # Example function: convert input to uppercase
        mymodel = ChatBot('model_training_environment/chatbot_model.h5', chatbothelper, functionality)

        processed_result = mymodel.predict(user_input)

        #preprocessed_input = chatbothelper.nlp_preprocessing(user_input)
        #preprocessed_input = chatbothelper.embed(preprocessed_input)
        #processed_result = mymodel.predict(np.array([preprocessed_input]))
        #prediction = np.argmax(processed_result)


        return jsonify({'result': processed_result}), 200  # HTTP 200 OK

    except Exception as e:
            # If there is an error, return an error message
            return jsonify({'error': str(e)}), 500  # HTTP 500 Internal Server Error


if __name__ == '__main__':
    app.run(debug=True)















