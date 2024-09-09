from flask import Flask, render_template, request, jsonify
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression
from sklearn.datasets import load_iris

app = Flask(__name__)
iris = load_iris()
X = iris.data
y = iris.target

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
scaler = StandardScaler()
X_train = scaler.fit_transform(X_train)
X_test = scaler.transform(X_test)

model = LogisticRegression()
model.fit(X_train, y_train)

@app.route('/')
def index():
    return render_template('index.html')

@app.route('/predict', methods=['POST'])
def predict():
    data = [float(x) for x in request.form.values()]
    final_data = np.array(data).reshape(1, -1)
    final_data = scaler.transform(final_data)
    
    prediction = model.predict(final_data)[0]
    output = iris.target_names[prediction]
    
    return render_template('index.html', prediction_text=f'Iris flower type: {output}')

if __name__ == '__main__':
    app.run(debug=True)
