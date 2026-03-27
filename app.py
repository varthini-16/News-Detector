from flask import Flask, render_template, request
import joblib

app = Flask(__name__)

model = joblib.load("model.pkl")
vectorizer = joblib.load("vectorizer.pkl")

@app.route("/")
def home():
    return render_template("home.html")

@app.route("/True_news", methods=["GET", "POST"])
def True_news():
    result = ""
    reasons = []
    proof_link = ""
    confidence = ""   # ✅ ADD (HTML la use pannura)

    if request.method == "POST":
        news = request.form["news"].lower()

        transformed = vectorizer.transform([news])
        prediction = model.predict(transformed)
        prob = model.predict_proba(transformed)  # ✅ ADD

        confidence = round(max(prob[0]) * 100, 2)  # ✅ ADD

        # 👉 FAKE NEWS
        if prediction[0] == 0:
            result = "FAKE NEWS ❌"

            reasons.append("No trusted source found")
            reasons.append("No proper evidence available")

        # 👉 REAL NEWS
        else:
            result = "REAL NEWS ✅"

            # 👉 Better keyword check (IMPORTANT FIX 🔥)
            if ("usa" in news or 
                "united states" in news or 
                "israel" in news or 
                "iran" in news):
                
                proof_link = "https://youtu.be/buglfD-2Hog?si=jMiE0Rt_rHSmM1QD"

    return render_template("index.html",
                           result=result,
                           reasons=reasons,
                           proof_link=proof_link,
                           confidence=confidence)   # ✅ ADD

if __name__ == "__main__":
    app.run(debug=True)
