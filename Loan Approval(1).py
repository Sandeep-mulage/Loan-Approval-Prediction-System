# app.py

from flask import Flask, render_template, request
import pickle
import numpy as np

app = Flask(__name__)

# Load Model
model = pickle.load(open("model.pkl", "rb"))


@app.route('/')
def home():
    return render_template("index.html")


@app.route('/predict', methods=['POST'])
def predict():

    # Get values from form
    data = [float(x) for x in request.form.values()]

    # Use only first 10 features
    data = data[:10]

    final_data = np.array([data])

    prediction = model.predict(final_data)[0]

    # Result Logic
    if prediction == 1:
        result = "Loan Approved"
        reason = "Congratulations! You are eligible for loan."

    else:
        result = "Loan Rejected"

        if data[5] < 2500:
            reason = "Low Applicant Income"
        elif data[9] == 0:
            reason = "Poor Credit History"
        elif data[7] > 300:
            reason = "Loan Amount Too High"
        else:
            reason = "Eligibility Criteria Not Met"

    return render_template("result.html",
                           result=result,
                           reason=reason)


if __name__ == "__main__":
    app.run(debug=True))