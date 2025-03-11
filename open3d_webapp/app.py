from flask import Flask, render_template
import numpy as np
import os
import json
from flask_socketio import SocketIO
import time

app = Flask(__name__)
socketio = SocketIO(app, cors_allowed_origins="*")

data_path = "data/"
labels_path = os.path.join(data_path, "labels")
scans_path = os.path.join(data_path, "scans")


def load_lidar_scan(bin_file):
    scan = np.fromfile(bin_file, dtype=np.float32)
    scan = scan.reshape(-1, 5)
    return scan[:, :3].tolist()  # Return only XYZ points


def load_bounding_boxes(txt_file):
    with open(txt_file, 'r') as f:
        lines = f.readlines()
    boxes = []
    for line in lines:
        parts = line.strip().split()
        label = parts[0]
        x, y, z, w, l, h, yaw = map(float, parts[1:])
        boxes.append({"label": label, "x": x, "y": y, "z": z, "w": w, "l": l, "h": h, "yaw": yaw})
    return boxes


@app.route('/')
def index():
    return render_template('index.html')


@socketio.on('start_animation')
def start_animation(data):
    start_frame = int(data['start'])
    end_frame = int(data['end'])
    delay = float(data.get('delay', 0.1))
    
    for frame_id in range(start_frame, end_frame + 1):
        bin_file = os.path.join(scans_path, f"{frame_id:06d}.bin")
        txt_file = os.path.join(labels_path, f"{frame_id:06d}.txt")
        
        if os.path.exists(bin_file) and os.path.exists(txt_file):
            lidar_data = load_lidar_scan(bin_file)
            bounding_boxes = load_bounding_boxes(txt_file)
            
            socketio.emit('frame_data', json.dumps({
                "frame": frame_id,
                "points": lidar_data,
                "boxes": bounding_boxes
            }))
        
        time.sleep(delay)  # Control playback speed


if __name__ == '__main__':
    socketio.run(app, debug=True, port=5001)