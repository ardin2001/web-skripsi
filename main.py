from flask import Flask, request, jsonify
import joblib
from flask import render_template

# Memuat model, vectorizer, dan transformer
model = joblib.load('sentiment_model.pkl')
count_vectorizer = joblib.load('tf_vectorizer.pkl')
tfidf_transformer = joblib.load('tfidf_vectorizer.pkl')

app = Flask(__name__)

@app.route('/')
def about():
    return jsonify(
        {'status': True, 'statusCode' : 200, 'message': 'Sentiment Analysis API'}
    )

@app.route('/home')
def hello():
    return render_template('index.html')

@app.route('/predict', methods=['POST'])
def predict():
    data = request.get_json(force=True)
    text = data['text']
    if(not text):
        return jsonify(
            {'status': False, 'statusCode' : 400, 'message': 'Text cannot be empty'}
        )
    
    text_counts = count_vectorizer.transform([text])
    text_tfidf = tfidf_transformer.transform(text_counts)
    
    prediction = model.predict(text_tfidf)[0]
    sentiment = ""
    if prediction == 0:
        sentiment = 'neutral'
    elif prediction == 1:
        sentiment = 'positive'
    elif prediction == -1:
        sentiment = 'negative'
    return jsonify(
        {'status': True, 'statusCode' : 200, 'message': 'Sentiment Analysis', 'data': {'sentiment': sentiment}}
    )

if __name__ == '__main__':
    app.run(debug=True)
