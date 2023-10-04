from flask import Flask, request, jsonify

from model.step4_run_svm_prediction import handle_request

app = Flask(__name__)


@app.route('/sentiment', methods=['POST'])
def process():
    try:
        data = request.get_json()

        if 'q' in data:
            input_string = data['q']
            result = handle_request(input_string)
            return jsonify({'sentiment': result})
        else:
            return jsonify({'error': 'Missing input_string in request data'}), 400
    except Exception as e:
        return jsonify({'error': str(e)}), 500


if __name__ == '__main__':
    app.run(debug=True, host='0.0.0.0', port=8080)
