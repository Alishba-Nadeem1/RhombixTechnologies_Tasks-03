from flask import Flask, render_template, request
from textblob import TextBlob

app = Flask(__name__)

def scoreEssay(essayText):
    # Content score = based on length (longer = better up to a limit)
    contentScore = min(len(essayText.split()) / 10, 10)

    # Grammar score = based on spelling/grammar corrections
    blob = TextBlob(essayText)
    mistakes = sum([1 for word in blob.words if word != str(TextBlob(word).correct())])
    grammarScore = max(10 - mistakes, 0)

    # Final score (average)
    finalScore = round((contentScore + grammarScore) / 2, 2)
    return contentScore, grammarScore, finalScore

@app.route("/", methods=["GET", "POST"])
def index():
    if request.method == "POST":
        prompt = request.form["prompt"]
        essay = request.form["essay"]

        contentScore, grammarScore, finalScore = scoreEssay(essay)

        return render_template("index.html",
                               prompt=prompt,
                               essay=essay,
                               contentScore=contentScore,
                               grammarScore=grammarScore,
                               finalScore=finalScore)
    return render_template("index.html")

if __name__ == "__main__":
    app.run(debug=True)
