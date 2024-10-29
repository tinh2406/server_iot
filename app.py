from flask import Flask, request, jsonify

import classifier_service

app = Flask(__name__)

# Định nghĩa endpoint '/process-array' nhận mảng từ request
@app.route('/predict', methods=['POST'])
def process_array():
    data = request.get_json()
    
    if not data or 'array' not in data:
        return jsonify({"error": "Request must contain an 'array' field."}), 400

    array = data['array']
    
    if not isinstance(array, list):
        return jsonify({"error": "'array' must be a list."}), 400

    response = {
        "original_array": array,
        "label":classifier_service.predict(array)
    }
    
    return jsonify(response), 200

if __name__ == '__main__':
    app.run(debug=True)