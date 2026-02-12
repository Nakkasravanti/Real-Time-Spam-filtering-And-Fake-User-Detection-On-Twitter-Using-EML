from flask import Flask, request, jsonify
from flask_cors import CORS
import os
from ml_service import MLService

# Configure static/template folders for unified deployment
# The dist folder will be at the root, so from /backend it's ../frontend/dist
app = Flask(__name__, 
            static_folder='../frontend/dist', 
            static_url_path='/')

CORS(app, resources={r"/*": {"origins": "*"}})

ml_service = MLService()

@app.errorhandler(404)
def not_found(e):
    # Catch-all route to serve index.html for React routing
    return app.send_static_file('index.html')

@app.errorhandler(500)
def handle_500(e):
    return jsonify({"error": "Internal Server Error", "details": str(e)}), 500

@app.errorhandler(Exception)
def handle_exception(e):
    return jsonify({"error": "Unexpected Error", "details": str(e)}), 500


@app.route('/')
def home():
    return app.send_static_file('index.html')

@app.route('/api/status')
def status():
    return jsonify({"message": "Spam Shield AI Backend is running", "status": "healthy"})

@app.route('/api/health')
def health():
    return jsonify({"status": "ok"})

UPLOAD_FOLDER = 'uploads'
os.makedirs(UPLOAD_FOLDER, exist_ok=True)

@app.route('/api/upload', methods=['POST'])
def upload_file():
    if 'file' not in request.files:
        return jsonify({"error": "No file part"})
    file = request.files['file']
    if file.filename == '':
        return jsonify({"error": "No selected file"})
    if file:
        filepath = os.path.join(UPLOAD_FOLDER, file.filename)
        file.save(filepath)
        return jsonify({"message": "File uploaded successfully", "path": os.path.abspath(UPLOAD_FOLDER)})
    return jsonify({"error": "Unknown error"})

@app.route('/api/predict/text', methods=['POST'])
def predict_text():
    data = request.json
    text = data.get('text', '')
    result = ml_service.predict_naive_bayes_text(text)
    return jsonify({"result": result})

@app.route('/api/analyze', methods=['POST'])
def analyze_folder():
    print("DEBUG: Received analyze request")
    folder_name = request.json.get('path', 'tweets') 
    
    # Try multiple possible locations for the folder
    possible_paths = [
        os.path.join(os.getcwd(), folder_name),           # Local /backend/tweets
        os.path.join(os.path.dirname(os.getcwd()), folder_name), # Repo root /tweets
        os.path.abspath(folder_name)                      # Absolute
    ]
    
    folder_path = None
    for p in possible_paths:
        if os.path.exists(p):
            folder_path = p
            break
            
    if not folder_path:
        # Default to first one if none exist yet (will eventually return JSON error from MLService)
        folder_path = possible_paths[1] 

    print(f"DEBUG: Selected path: {folder_path}")
    result = ml_service.analyze_folder(folder_path)
    print("DEBUG: Analysis complete")
    return jsonify(result)

@app.route('/api/train/<algorithm>', methods=['POST'])
def train_model(algorithm):
    print(f"DEBUG: Received train request for {algorithm}")
    result = ml_service.train_predict(algorithm)
    print(f"DEBUG: Training complete for {algorithm}")
    return jsonify(result)

@app.route('/api/results', methods=['GET'])
def get_results():
    return jsonify(ml_service.get_comparison())

@app.route('/api/check/user', methods=['POST'])
def check_user():
    data = request.json
    followers = data.get('followers', 0)
    following = data.get('following', 0)
    result = ml_service.check_manual_user(followers, following)
    return jsonify(result)

@app.route('/api/search/user', methods=['POST'])
def search_user():
    data = request.json
    username = data.get('username', '')
    result = ml_service.get_account_status(username)
    return jsonify(result)

if __name__ == '__main__':
    app.run(debug=True, port=5000)
