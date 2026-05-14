from flask import Flask, request, render_template, jsonify
import os
from detect import predict

app = Flask(__name__)
# Flask automatically serves files from the 'static' folder.

@app.route('/')
def index():
    return render_template('index.html')

@app.route('/detect', methods=['POST'])
def detect():
    if 'file' not in request.files:
        return jsonify({"error": "No file uploaded"}), 400
        
    file = request.files['file']
    if file.filename == '':
        return jsonify({"error": "No selected file"}), 400

    try:
        image_bytes = file.read()
        result = predict(image_bytes)
        return jsonify(result)
    except Exception as e:
        import traceback
        traceback.print_exc()
        return jsonify({"error": str(e)}), 500

if __name__ == '__main__':
    # Make sure static/heatmaps exists for saving images
    os.makedirs('static/heatmaps', exist_ok=True)
    app.run(debug=True)