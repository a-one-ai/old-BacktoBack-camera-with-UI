from flask import Flask, jsonify, request
import concurrent.futures
import time

app = Flask(__name__)

def background_task(data):
    # Simulate a long-running task
    for i in range(10):
        time.sleep(1)
        print(f"Task running with data '{data}'... {i+1}")

@app.route('/')
def index():
    return jsonify({'message': 'Welcome to the background task example!'})

@app.route('/start_task', methods=['POST'])
def start_task():
    data = request.json.get('data')
    if data:
        # Start the background task with data using a ThreadPoolExecutor
        with concurrent.futures.ThreadPoolExecutor() as executor:
            executor.submit(background_task, data)
        return jsonify({'message': f'Task started in background with data: {data}'}), 200
    else:
        return jsonify({'error': 'Data not provided.'}), 400

if __name__ == '__main__':
    app.run()
