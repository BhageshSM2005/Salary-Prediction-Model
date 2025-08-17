from flask import Flask, request, render_template
import joblib
import numpy as np

app = Flask(__name__)
model = joblib.load("Salary_model.pkl")

@app.route("/", methods=["GET", "POST"])
def home():
    if request.method == "POST":
        try:
            exp = float(request.form["experience"])
            prediction = model.predict([[exp]])
            return render_template("index.html", prediction=f"Predicted Salary: ${prediction[0]:,.2f}")
        except:
            return render_template("index.html", prediction="Invalid input. Please enter a number.")
    return render_template("index.html")

if __name__=="__main__":
    app.run(debug=True)