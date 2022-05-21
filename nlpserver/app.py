from flask import Flask, request, jsonify
from flask_cors import CORS
import requests
import json
from google.cloud import language_v1

app = Flask(__name__)
CORS(app)

@app.route('/')
def hello():
    return "Hello World"

# java -mx4g -cp "*" edu.stanford.nlp.pipeline.StanfordCoreNLPServer -port 9000 -timeout 15000
@app.route('/corenlp', methods=["POST"])
def call_nlp_server():
    text = request.get_json()['data']

    response = requests.post(
        'http://localhost:9000/?properties={"annotators":"tokenize,ssplit,pos,sentiment","outputFormat":"json"}', 
        data = {'data': text}
    ).text

    dict_response = json.loads(response)
    sentiment = dict_response['sentences'][0]['sentiment']

    return sentiment

# export GOOGLE_APPLICATION_CREDENTIALS="/Users/vincentndokaj/Research/bold-ally-343516-60720f1bc2cc.json"
@app.route('/googlenlp', methods=["POST"])
def call_google_nlp():
    text = request.get_json()['data']

    client = language_v1.LanguageServiceClient()

    document = language_v1.Document(
        content=text, type_=language_v1.Document.Type.PLAIN_TEXT
    )

    # Detects the sentiment of the text
    sentiment = client.analyze_sentiment(
        request={"document": document}
    ).document_sentiment

    return {"magnitude" : sentiment.magnitude, "score" : sentiment.score}

if __name__ == "__main__":
    app.run(debug=True)