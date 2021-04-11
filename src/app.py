import os

import spacy
from flask import Flask, render_template, jsonify, request

from src.components import TextProcess

app = Flask(__name__)
textProcess = TextProcess()

@app.route('/')
def index():
    return render_template('index.html')


@app.route('/answer-question', methods=['POST'])
def analyzer():
    data = request.get_json()
    question = data.get('question')
    docs = textProcess.search_text(question)
    return jsonify(docs)
   


if __name__ == '__main__':
    app.run(host='0.0.0.0', port=5000)
