import numpy as np
from flask import Flask, request, jsonify, render_template
import pickle

app = Flask(__name__)
model = pickle.load(open('model.pkl', 'rb'))

@app.route('/')
def home():
    return render_template('index.html')

@app.route('/predict',methods=['POST'])
def predict():
    '''
    For rendering results on HTML GUI
    '''
    f_features = [float(x) for x in request.form.values()]
    f_features[2]=1/(7.6082-f_features[2])
    final_features = [np.array(f_features)]
    prediction = model.predict(final_features)
    
    
    output = round(prediction[0], 2)

    return render_template('index.html', prediction_text='Cement 1Day Strength : {}'.format(output))



if __name__ == "__main__":
    app.run(debug=True)
    

