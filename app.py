from flask import Flask, render_template, request
import pickle
from tensorflow.keras.preprocessing.sequence import pad_sequences
import numpy as np

# Load the model
model = pickle.load(
    open('models/model_1.pkl', 'rb'))

with open('models/tokenizer.pkl', 'rb') as handle:
    tokenizer = pickle.load(handle)

# Initialize Flask app
app = Flask(__name__)

# Home page route


@app.route('/')
def home():
    return render_template('index.html')

# Predict route


@app.route('/predict', methods=['POST'])
def predict():
    if request.method == 'POST':
        message = request.form['message']
        seq = tokenizer.texts_to_sequences([message])
        padded = pad_sequences(seq, maxlen=120)
        prediction = model.predict(padded)
        label = "Hate Speech" if prediction[0][0] > 0.5 else "Free Speech"
        return render_template('index.html', prediction=label)


if __name__ == '__main__':
    app.run(debug=True)
