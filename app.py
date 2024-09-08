import numpy as np
import pandas as pd
import pickle
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.linear_model import LinearRegression
from sklearn.impute import SimpleImputer
from flask import Flask, render_template, request, redirect, url_for, flash

df = pd.read_csv('c:\Users\srira\OneDrive\Desktop\learning\task 1\house_prices.csv')

numeric_cols = df.select_dtypes(include=[np.number]).columns
df[numeric_cols] = df[numeric_cols].fillna(df[numeric_cols].mean())

non_numeric_cols = df.select_dtypes(exclude=[np.number]).columns
for col in non_numeric_cols:
    df[col] = df[col].fillna(df[col].mode()[0])

features = ['Area', 'No. of Bedrooms', 'New/Resale', 'Lift Available', 
            'Car Parking', '24x7 Security', 'Children\'s Play Area', 
             'Landscaped Gardens', 
            'Gas Connection', 'Swimming Pool', 'Location']
X = df[features]
y = df['Price']

numeric_features = features[:-1]
categorical_features = ['Location']

preprocessor = ColumnTransformer(
    transformers=[
        ('num', SimpleImputer(strategy='mean'), numeric_features),
        ('cat', OneHotEncoder(), categorical_features)
    ])

pipeline = Pipeline([
    ('preprocessor', preprocessor),
    ('scaler', StandardScaler(with_mean=False)),
    ('model', LinearRegression())
])

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

pipeline.fit(X_train, y_train)

with open('model.pkl', 'wb') as model_file:
    pickle.dump(pipeline, model_file)

app = Flask(__name__)
app.secret_key = 'your_secret_key'

@app.route('/')
def home():
    return render_template('index.html')

@app.route('/predict', methods=['POST'])
def predict():
    try:
        form_values = [request.form.get(f) for f in features[:-1]] + [request.form.get('Location')]
        input_features = pd.DataFrame([form_values], columns=features)

        with open('model.pkl', 'rb') as model_file:
            pipeline = pickle.load(model_file)

        prediction = pipeline.predict(input_features)

        prediction_text = f'House Price: ₹{abs(prediction[0]):,.2f}'

        return render_template('index.html', prediction_text=prediction_text)

    except ValueError:
        flash('Invalid input: Please enter valid values for all features.')
        return redirect(url_for('home'))

    except Exception as e:
        flash(f'An error occurred: {str(e)}')
        return redirect(url_for('home'))

if __name__ == "__main__":
    app.run(debug=True)
