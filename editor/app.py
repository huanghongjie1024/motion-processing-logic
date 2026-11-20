from flask import Flask, render_template, request, jsonify, send_file
import numpy as np
import os
import io
import re

app = Flask(
    __name__,
    static_folder=os.environ.get("MOTION_EDITOR_MODEL_DIR", "static"),
    static_url_path="/static",
)


def safe_npz_download_name(filename):
    """Return a safe basename while preserving the imported NPZ filename."""
    name = str(filename or '').replace('\\', '/').rsplit('/', 1)[-1]
    name = re.sub(r'[\x00-\x1f\x7f]', '', name).strip()
    if not name:
        return 'edited_motion.npz'
    if not name.lower().endswith('.npz'):
        name += '.npz'
    return name


@app.route('/')
def index():
    return render_template('index.html')

@app.route('/upload_motion', methods=['POST'])
def upload_motion():
    if 'file' not in request.files:
        return jsonify({'error': 'No file part'}), 400
    
    file = request.files['file']
    
    try:
        with np.load(file, allow_pickle=False) as data:
            joint_names = data['joint_names']
            joint_pos = data['joint_pos']
            base_pos = data['base_pos_w']
            base_quat = data['base_quat_w']
            framerate = float(data['framerate']) if 'framerate' in data else 30.0
            frame_count = len(joint_pos)
            if (joint_names.ndim != 1 or joint_pos.shape != (frame_count, len(joint_names))
                or base_pos.shape != (frame_count, 3) or base_quat.shape != (frame_count, 4)
                or not frame_count):
                return jsonify({'error': 'Invalid motion array dimensions'}), 400
            return jsonify({
                '_source_filename': safe_npz_download_name(file.filename),
                'joint_names': joint_names.astype(str).tolist(),
                'joint_pos': joint_pos.tolist(),
                'base_pos_w': base_pos.tolist(),
                'base_quat_w': base_quat.tolist(),
                'num_frames': frame_count,
                'framerate': framerate,
            })

    except (KeyError, OSError, ValueError, TypeError) as exc:
        return jsonify({'error': str(exc)}), 400
    
@app.route('/save_motion', methods=['POST'])
def save_motion():
    try:
        # 1. Get JSON data from the frontend
        data = request.json
        
        if not data:
            return jsonify({"error": "No data received"}), 400

        download_name = safe_npz_download_name(data.get('_source_filename'))

        # 2. Prepare dictionaries for np.savez
        # We assume the keys in JSON match the original npz keys
        output_dict = {}

        # Helper: Convert lists back to numpy arrays
        if 'joint_pos' in data:
            output_dict['joint_pos'] = np.array(data['joint_pos'], dtype=np.float64)
        
        if 'joint_names' in data:
            # joint_names must be numpy array of strings
            output_dict['joint_names'] = np.array(data['joint_names'])

        if 'base_pos_w' in data and data['base_pos_w']:
             output_dict['base_pos_w'] = np.array(data['base_pos_w'], dtype=np.float64)
             
        if 'base_quat_w' in data and data['base_quat_w']:
             output_dict['base_quat_w'] = np.array(data['base_quat_w'], dtype=np.float64)

        if 'framerate' in data:
            output_dict['framerate'] = np.array(data['framerate'], dtype=np.float64)

        # 3. Write to an in-memory buffer (like a virtual file)
        memory_file = io.BytesIO()
        np.savez(memory_file, **output_dict)
        memory_file.seek(0)

        # 4. Send the file back to the browser
        return send_file(
            memory_file,
            mimetype='application/octet-stream',
            as_attachment=True,
            download_name=download_name
        )

    except Exception as e:
        print(f"Error saving NPZ: {e}")
        return jsonify({"error": str(e)}), 500

if __name__ == '__main__':
    app.run(host="127.0.0.1", port=5000)
