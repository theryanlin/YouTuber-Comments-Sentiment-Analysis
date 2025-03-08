from flask import Flask

app = Flask(__name__)

@app.route('/')
def home():
    return "Hello, Zeabur! Your Flask app is running."

if __name__ == '__main__':
    port = int(os.environ.get('PORT', 8080))  # Use PORT if set, otherwise default to 8080
    app.run(host='0.0.0.0', port=port)
