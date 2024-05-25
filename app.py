from flask import Flask, render_template, request
from test import record_to_file, extract_feature
from utils import create_model

app = Flask(__name__)

model = create_model()
model.load_weights("results/model.h5")

@app.route('/')
def index():
    return render_template('index.html')

@app.route('/predict', methods=['POST'])
def predict():
    file = "static/test.wav"  
    record_to_file(file)
    features = extract_feature(file, mel=True).reshape(1, -1)
    male_prob = model.predict(features)[0][0]
    female_prob = 1 - male_prob
    gender = "male" if male_prob > female_prob else "female"
    return render_template('result.html', gender=gender, male_prob=male_prob*100, female_prob=female_prob*100)

if __name__ == '__main__':
    app.run(debug=True)