from flask import Flask, request, jsonify
from workflow import workflow

app = Flask(__name__)

@app.route("/generate_user_stories", methods=["POST"])
def generate_user_stories():
    data = request.json
    state = workflow.run(UserStoryState(input_type=data["type"], input_data=data["data"]))
    return jsonify({"user_stories": state.user_stories})

if __name__ == "__main__":
    app.run(debug=True)
