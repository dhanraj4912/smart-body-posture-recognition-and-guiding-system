# posture_server.py
import os
import time
import threading
import sqlite3
from math import atan2, degrees
from typing import Dict, Any, List
import uuid

from flask import Flask, Response, request, jsonify, send_from_directory
from flask_socketio import SocketIO
import cv2
import numpy as np
import tensorflow as tf
import tensorflow_hub as hub
from ultralytics import YOLO
import joblib
import requests
from dotenv import load_dotenv
import smtplib
from email.mime.text import MIMEText
from flask_cors import CORS

# Local NLP for chatbot
try:
    from transformers import pipeline
    nlp_initialized = True
except ImportError:
    nlp_initialized = False

load_dotenv()

# -------- Configuration --------
DEFAULT_STREAM = os.getenv("VIDEO_DEVICE", "0")  # can be camera index or RTSP URL
STREAMS = [s.strip() for s in os.getenv("STREAMS", DEFAULT_STREAM).split(",") if s.strip()]
REFRESH_INTERVAL = float(os.getenv("REFRESH_INTERVAL", "6.0"))
ALERT_PERSIST_SECONDS = float(os.getenv("ALERT_PERSIST_SECONDS", "6.0"))
MIN_BOX_AREA = int(os.getenv("MIN_BOX_AREA", "64"))
FRAME_SKIP = int(os.getenv("FRAME_SKIP", "1"))
YOLO_WEIGHTS = os.getenv("YOLO_WEIGHTS", "yolov8n.pt")

# Botpress Configuration
BOTPRESS_PAT = os.getenv("BOTPRESS_PAT", "")
BOTPRESS_BOT_ID = os.getenv("BOTPRESS_BOT_ID", "")
BOTPRESS_WORKSPACE_ID = os.getenv("BOTPRESS_WORKSPACE_ID", "")
BOTPRESS_AVAILABLE = bool(BOTPRESS_PAT and BOTPRESS_BOT_ID and BOTPRESS_WORKSPACE_ID)

# Local NLP pipeline for chatbot
text_generator = None
if nlp_initialized:
    try:
        print("⏳ Loading local text generation model...")
        text_generator = pipeline("text2text-generation", model="google/flan-t5-base")
        print("✅ Local NLP model loaded successfully")
    except Exception as e:
        print(f"⚠️ Failed to load local NLP model: {e}")


# -------- App --------
app = Flask(__name__, static_folder='../frontend', static_url_path='')
CORS(app)  # Enable CORS for all routes
socketio = SocketIO(app, cors_allowed_origins="*", async_mode="threading")

# -------- Models --------
detector = YOLO(YOLO_WEIGHTS)
movenet = hub.load("https://tfhub.dev/google/movenet/singlepose/lightning/4")
posture_model = None
if os.path.exists('models/posture_classifier.pkl'):
    posture_model = joblib.load('models/posture_classifier.pkl')

# -------- DB --------


# -------- State --------
state_lock = threading.Lock()
feature_ema: Dict[str, Dict[int, Dict[str, float]]] = {}   # stream_id -> pid -> feature EMA
last_postures: Dict[str, Dict[int, Dict[str, Any]]] = {}   # stream_id -> pid -> info
last_eval_by_id: Dict[str, Dict[int, float]] = {}          # stream_id -> pid -> ts
user_map: Dict[int, Dict[str, Any]] = {}                   # pid -> {'user_id', 'email', 'socket_id'}

# -------- Utilities --------
def extract_keypoints(image):
    img = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
    img = tf.image.resize_with_pad(np.expand_dims(img, axis=0), 192, 192)
    img = tf.cast(img, dtype=tf.int32)
    outputs = movenet.signatures['serving_default'](img)
    return outputs['output_0'].numpy()[0, 0, :, :2]

def angle_between(a, b, c):
    ax, ay = a; bx, by = b; cx, cy = c
    v1 = (ax - bx, ay - by); v2 = (cx - bx, cy - by)
    dot = v1[0]*v2[0] + v1[1]*v2[1]
    mag1 = (v1[0]**2 + v1[1]**2)**0.5; mag2 = (v2[0]**2 + v2[1]**2)**0.5
    if mag1 == 0 or mag2 == 0: return 0.0
    from math import acos
    cosang = max(-1.0, min(1.0, dot/(mag1*mag2)))
    return degrees(acos(cosang))

def ema(prev, new, alpha=0.5):
    return alpha * new + (1 - alpha) * prev if prev is not None else new


def compute_posture_features(kpts):
    """
    Computes robust, NORMALIZED posture features from Movenet keypoints.
    Keypoints are [y, x].
    """
    def mid(p, q): 
        return ((p[0] + q[0]) / 2.0, (p[1] + q[1]) / 2.0)

    # Keypoints (y, x) - MoveNet indices
    nose = kpts[0]
    left_eye, right_eye = kpts[1], kpts[2]
    left_ear, right_ear = kpts[3], kpts[4]
    left_sh, right_sh = kpts[5], kpts[6]
    left_elbow, right_elbow = kpts[7], kpts[8]
    left_wrist, right_wrist = kpts[9], kpts[10]
    left_hip, right_hip = kpts[11], kpts[12]
    left_knee, right_knee = kpts[13], kpts[14]
    left_ankle, right_ankle = kpts[15], kpts[16]

    # Use shoulder width (x-distance) as our primary horizontal normalizer
    shoulder_width = abs(left_sh[1] - right_sh[1])
    if shoulder_width < 0.01:
        shoulder_width = 0.1
    
    # Use torso height (y-distance) as our primary vertical normalizer
    mid_sh_y = (left_sh[0] + right_sh[0]) / 2.0
    mid_hip_y = (left_hip[0] + right_hip[0]) / 2.0
    torso_height = abs(mid_hip_y - mid_sh_y)
    if torso_height < 0.01:
        torso_height = 0.1

    # --- Feature Calculations ---

    # 1. Torso Angle (Slouch Detection)
    # Measures the angle of the spine from vertical
    mid_sh = mid(left_sh, right_sh)
    mid_hip = mid(left_hip, right_hip)
    dx = mid_hip[1] - mid_sh[1]  # Delta X (horizontal)
    dy = mid_hip[0] - mid_sh[0]  # Delta Y (vertical)
    
    # Angle from vertical (0 = perfectly vertical, 90 = horizontal)
    torso_angle = abs(degrees(atan2(dx, dy)))
    if torso_angle > 90:
        torso_angle = 180 - torso_angle

    # 2. Forward Head Ratio (Head Position relative to shoulders)
    # Measures how far forward the head is
    head_x = nose[1]
    mid_sh_x = (left_sh[1] + right_sh[1]) / 2.0
    forward_head_ratio = abs(head_x - mid_sh_x) / shoulder_width if shoulder_width > 0 else 0

    # 3. Shoulder Tilt Ratio (Uneven shoulders)
    shoulder_tilt_ratio = abs(left_sh[0] - right_sh[0]) / shoulder_width if shoulder_width > 0 else 0

    # 4. Hip Tilt Ratio (Hip imbalance)
    hip_tilt_ratio = abs(left_hip[0] - right_hip[0]) / shoulder_width if shoulder_width > 0 else 0

    # 5. Spine Curve (Curvature of spine from shoulders to hips)
    spine_curve_ratio = abs(mid_sh[1] - mid_hip[1]) / torso_height if torso_height > 0 else 0

    # 6. Neck Angle (Forward neck posture)
    ear_mid = mid(left_ear, right_ear)
    shoulder_mid = mid(left_sh, right_sh)
    neck_dx = ear_mid[1] - shoulder_mid[1]
    neck_dy = ear_mid[0] - shoulder_mid[0]
    neck_angle = abs(degrees(atan2(neck_dx, neck_dy)))
    if neck_angle > 90:
        neck_angle = 180 - neck_angle

    # Convert all np.float32 values to standard Python floats
    return {
        'torso_angle': float(torso_angle),
        'forward_head_ratio': float(forward_head_ratio),
        'shoulder_tilt_ratio': float(shoulder_tilt_ratio),
        'hip_tilt_ratio': float(hip_tilt_ratio),
        'spine_curve_ratio': float(spine_curve_ratio),
        'neck_angle': float(neck_angle)
    }

def apply_calibration(user_id, feats):
    """
    Applies the user's stored baseline offsets to the current features.
    This makes the final features a "deviation from good" measurement.
    """
    if not user_id:
        return feats
        
    try:
        conn = sqlite3.connect(DB_PATH)
        c = conn.cursor()
        c.execute("""
            SELECT torso_angle_baseline, forward_head_ratio_baseline,
                   shoulder_tilt_ratio_baseline, hip_tilt_ratio_baseline
            FROM posture_calibration WHERE user_id=?
        """, (user_id,))
        row = c.fetchone()
        conn.close()
    except Exception:
        # DB might be locked, fail safe
        return feats

    if not row:
        return feats

    baselines = {
        'torso_angle': row[0] or 0.0,
        'forward_head_ratio': row[1] or 0.0,
        'shoulder_tilt_ratio': row[2] or 0.0,
        'hip_tilt_ratio': row[3] or 0.0
    }

    calibrated_feats = dict(feats)

    for key in baselines:
        if key in calibrated_feats:
            calibrated_feats[key] = max(0.0, calibrated_feats[key] - baselines[key])

    return calibrated_feats


def classify_posture_from_features(feat):
    """
    Classify posture based on extracted features with improved thresholds.
    """
    t = feat.get('torso_angle', 0.0)           # 0-90 degrees
    fhr = feat.get('forward_head_ratio', 0.0)  # 0-1 normalized
    str_ = feat.get('shoulder_tilt_ratio', 0.0) # 0-1 normalized
    htr = feat.get('hip_tilt_ratio', 0.0)      # 0-1 normalized
    scr = feat.get('spine_curve_ratio', 0.0)   # 0-1 normalized
    na = feat.get('neck_angle', 0.0)           # 0-90 degrees

    issues = []
    suggestions = []
    confs = []

    # 1. TORSO ANGLE - Slouching detection
    # Good posture: 0-12 degrees
    # Mild slouch: 12-22 degrees
    # Severe slouch: >22 degrees
    if t > 22:
        issues.append('Severe slouching detected')
        suggestions.append('✓ Sit upright with back against chair')
        suggestions.append('✓ Keep hips back in chair')
        confs.append(min(1.0, (t - 22) / 15))
    elif t > 12:
        issues.append('Mild slouching')
        suggestions.append('✓ Straighten your back')
        suggestions.append('✓ Keep ribcage lifted')
        confs.append(min(1.0, (t - 12) / 10))
    print(f"[DEBUG] torso_angle={t:.2f}")

    # 2. FORWARD HEAD POSTURE (HEAD POSITION)
    # Good: < 0.12 (head aligned with shoulders)
    # Mild: 0.12-0.20
    # Severe: > 0.20
    if fhr > 0.20:
        issues.append('Severe forward head posture')
        suggestions.append('✓ Raise monitor to eye level')
        suggestions.append('✓ Gently tuck your chin')
        confs.append(min(1.0, (fhr - 0.20) / 0.15))
    elif fhr > 0.12:
        issues.append('Mild forward head posture')
        suggestions.append('✓ Adjust screen position')
        confs.append(min(1.0, (fhr - 0.12) / 0.08))
    print(f"[DEBUG] forward_head_ratio={fhr:.2f}")

    # 3. NECK ANGLE - Forward neck bending
    # Good: < 12 degrees
    # Mild: 12-22 degrees
    # Severe: > 22 degrees
    if na > 22:
        issues.append('Severe neck strain')
        suggestions.append('✓ Bring screen closer to eye level')
        confs.append(min(1.0, (na - 22) / 15))
    elif na > 12:
        issues.append('Mild neck strain')
        suggestions.append('✓ Adjust posture to reduce neck bend')
        confs.append(min(1.0, (na - 12) / 10))
    print(f"[DEBUG] neck_angle={na:.2f}")

    # 4. SHOULDER TILT - Uneven shoulders
    # Good: < 0.08
    # Mild: 0.08-0.14
    # Severe: > 0.14
    if str_ > 0.19:
        issues.append('Severe shoulder imbalance')
        suggestions.append('✓ Balance weight equally on both sides')
        confs.append(min(1.0, (str_ - 0.14) / 0.1))
    elif str_ > 0.12:
        issues.append('Mild shoulder imbalance')
        suggestions.append('✓ Keep shoulders level')
        confs.append(min(1.0, (str_ - 0.08) / 0.06))
    print(f"[DEBUG] shoulder_tilt={str_:.2f}")

    # 5. HIP TILT - Hip imbalance
    # Good: < 0.10
    # Mild: 0.10-0.16
    # Severe: > 0.16
    if htr > 0.16:
        issues.append('Severe hip imbalance')
        suggestions.append('✓ Sit evenly on both sitting bones')
        confs.append(min(1.0, (htr - 0.16) / 0.1))
    elif htr > 0.12:
        issues.append('Mild hip imbalance')
        suggestions.append('✓ Keep hips level')
        confs.append(min(1.0, (htr - 0.12) / 0.06))
    print(f"[DEBUG] hip_tilt={htr:.2f}")

    # 6. SPINE CURVE - Spinal curvature
    # Good: < 0.18
    # Mild: 0.18-0.28
    # Severe: > 0.28
    if scr > 0.28:
        issues.append('Severe spinal curvature')
        suggestions.append('✓ Use firm backrest support')
        confs.append(min(1.0, (scr - 0.28) / 0.15))
    elif scr > 0.18:
        issues.append('Mild spinal curvature')
        suggestions.append('✓ Maintain neutral spine')
        confs.append(min(1.0, (scr - 0.18) / 0.1))
    print(f"[DEBUG] spine_curve={scr:.2f}")

    # Determine severity
    severity = 'good'
    if any('Severe' in issue for issue in issues):
        severity = 'severe'
    elif any('Mild' in issue for issue in issues):
        severity = 'moderate'

    # Generate label and confidence
    if not issues:
        label = '✅ Good posture'
        confidence = 1.0
        suggestions = [
            '✓ Maintain this neutral alignment',
            '✓ Keep shoulders relaxed',
            '✓ Take regular stretch breaks'
        ]
    else:
        label = ' ; '.join(issues)
        confidence = float(round(max(confs) if confs else 0.5, 2))

    return label, severity, confidence, suggestions

class SimpleTracker:
    def __init__(self, max_distance=90, timeout=12.0):
        self.next_id = 0
        self.objects = {}  
        self.max_distance = max_distance; self.timeout = timeout

    def update(self, boxes):
        centroids = [(int((b[0]+b[2])/2), int((b[1]+b[3])/2)) for b in boxes]
        results = []
        now = time.time()

        for i, c in enumerate(centroids):
            best_id, best_dist = None, None
            for oid, rec in self.objects.items():
                oc = rec['centroid']
                d = (c[0]-oc[0])**2 + (c[1]-oc[1])**2
                if best_dist is None or d < best_dist:
                    best_dist, best_id = d, oid
            if best_id is None or best_dist is None or best_dist > self.max_distance**2:
                oid = self.next_id; self.next_id += 1
                self.objects[oid] = {'centroid': c, 'last_seen': now}
                results.append((oid, boxes[i]))
            else:
                self.objects[best_id] = {'centroid': c, 'last_seen': now}
                results.append((best_id, boxes[i]))

        for oid in list(self.objects.keys()):
            if now - self.objects[oid]['last_seen'] > self.timeout:
                del self.objects[oid]
        return results

trackers: Dict[str, SimpleTracker] = {sid: SimpleTracker() for sid in STREAMS}

def log_event(stream_id, pid, user_id, label, severity, confidence, suggestions):
    try:
        conn = sqlite3.connect(DB_PATH); c = conn.cursor()
        c.execute("INSERT INTO posture_events(ts, stream_id, person_id, user_id, label, severity, confidence, suggestions) VALUES (?,?,?,?,?,?,?,?)",
                  (int(time.time()), stream_id, int(pid), user_id or '', label, severity, float(confidence), "; ".join(suggestions)))
        conn.commit(); conn.close()
    except Exception:
        pass


def open_capture(src):
    try:
        src_idx = int(src)
        cap = cv2.VideoCapture(src_idx)
    except ValueError:
        cap = cv2.VideoCapture(src)  
    return cap

def generate_frames(stream_id):
    cap = open_capture(stream_id)
    if not cap.isOpened():
        raise RuntimeError(f'Could not open video source: {stream_id}')

    _ = detector.predict(np.zeros((640,640,3),dtype=np.uint8))
    _ = movenet.signatures['serving_default'](
        tf.cast(tf.image.resize_with_pad(np.zeros((1,192,192,3),dtype=np.uint8),192,192), tf.int32)
    )

    frame_idx = 0
    while True:
        ret, frame = cap.read()
        if not ret:
            break

        if FRAME_SKIP > 1 and (frame_idx % FRAME_SKIP != 0):
            frame_idx += 1
            continue

        h, w, _ = frame.shape
        yolo_results = detector(frame, classes=[0])
        boxes = []
        for box in yolo_results[0].boxes:
            x1, y1, x2, y2 = map(int, box.xyxy[0])
            x1, y1 = max(0, x1), max(0, y1)
            x2, y2 = min(w-1, x2), min(h-1, y2)
            if (x2-x1) < MIN_BOX_AREA or (y2-y1) < MIN_BOX_AREA: continue
            boxes.append((x1, y1, x2, y2))

        tracked = trackers[stream_id].update(boxes)
        current_time = time.time()

        with state_lock:
            last_eval_by_id.setdefault(stream_id, {})
            last_postures.setdefault(stream_id, {})
            feature_ema.setdefault(stream_id, {})

        for pid, box in tracked:
            x1, y1, x2, y2 = box
            crop = frame[y1:y2, x1:x2]
            label_display = 'Analyzing...'; color = (0, 255, 0)

            last_eval = last_eval_by_id[stream_id].get(pid, 0.0)
            analyze_now = (current_time - last_eval) > REFRESH_INTERVAL

            if analyze_now:
                try:
                    kpts = extract_keypoints(crop)
                    feats_raw = compute_posture_features(kpts)

                    user_id = user_map.get(pid, {}).get('user_id')

                    feats_cal = apply_calibration(user_id, feats_raw)

                    fprev = feature_ema[stream_id].get(pid, {})
                    feats = {k: ema(fprev.get(k), v, alpha=0.5) for k, v in feats_cal.items()}
                    feature_ema[stream_id][pid] = feats

                    label_text, severity, confidence, suggestions = classify_posture_from_features(feats)

                    with state_lock:
                        prev = last_postures[stream_id].get(pid)
                        if prev and prev.get('label') == label_text:
                            prev['duration'] = current_time - prev.get('since', current_time)
                        else:
                            last_postures[stream_id][pid] = {
                                'label': label_text, 'severity': severity, 'confidence': confidence,
                                'since': current_time, 'duration': 0.0, 'suggestions': suggestions,
                                'user_id': user_id
                            }

                        if label_text != 'Good posture':
                            dur = last_postures[stream_id][pid]['duration']
                            if dur >= ALERT_PERSIST_SECONDS:
                                payload = {
                                    'stream_id': stream_id,
                                    'person_id': pid,
                                    'user_id': user_id,
                                    'label': label_text,
                                    'severity': severity,
                                    'confidence': confidence,
                                    'suggestions': suggestions,
                                    'box': [x1, y1, x2, y2]
                                }
                                socketio.emit('posture_alert', payload)
                                user_rec = user_map.get(pid)
                                if user_rec and user_rec.get('socket_id'):
                                    socketio.emit('posture_alert', payload, to=user_rec['socket_id'])
                                log_event(stream_id, pid, user_id, label_text, severity, confidence, suggestions)

                    label_display = f"{label_text} ({severity}, conf {confidence})"
                    color = (0, 0, 255) if severity != 'good' else (0, 200, 0)
                    last_eval_by_id[stream_id][pid] = current_time
                except Exception as e:
                    label_display = f'Error: {str(e)}'
                    color = (0, 0, 255)
            else:
                with state_lock:
                    info = last_postures[stream_id].get(pid, {})
                    label_display = info.get('label', 'Analyzing...')
                    sev = info.get('severity', 'good')
                    color = (0, 200, 0) if sev == 'good' else (0, 0, 255)

            cv2.rectangle(frame, (x1, y1), (x2, y2), color, 2)
            # Add person ID label
            person_label = f"Person {pid}"
            cv2.putText(frame, person_label, (x1, max(0, y1 - 25)), cv2.FONT_HERSHEY_SIMPLEX, 0.6, color, 2)
            # Add posture status label
            cv2.putText(frame, str(label_display), (x1, max(0, y1 - 8)), cv2.FONT_HERSHEY_SIMPLEX, 0.55, color, 2)

        ret2, jpeg = cv2.imencode('.jpg', frame)
        if not ret2: 
            frame_idx += 1
            continue
        yield (b'--frame\r\n'
               b'Content-Type: image/jpeg\r\n\r\n' + jpeg.tobytes() + b'\r\n')
        frame_idx += 1

@app.route('/video_feed')
def video_feed_default():
    return Response(generate_frames(STREAMS[0]), mimetype='multipart/x-mixed-replace; boundary=frame')

@app.route('/video_feed/<path:stream_id>')
def video_feed_stream(stream_id):
    if stream_id not in STREAMS:
        return jsonify({'error': 'Unknown stream'}), 404
    return Response(generate_frames(stream_id), mimetype='multipart/x-mixed-replace; boundary=frame')

@app.route('/streams', methods=['GET', 'POST'])
def streams():
    if request.method == 'GET':
        return jsonify({'streams': STREAMS})
    data = request.get_json(silent=True) or {}
    new_streams = data.get('streams')
    if not isinstance(new_streams, list) or not new_streams:
        return jsonify({'error': 'Provide list of streams'}), 400
    STREAMS[:] = new_streams
    for sid in new_streams:
        trackers[sid] = trackers.get(sid) or SimpleTracker()
    return jsonify({'streams': STREAMS})





@app.route('/upload_photo', methods=['POST', 'OPTIONS'])
def upload_photo():
    # Handle CORS preflight requests
    if request.method == 'OPTIONS':
        response = jsonify({'status': 'ok'})
        response.headers.add('Access-Control-Allow-Origin', '*')
        response.headers.add('Access-Control-Allow-Headers', 'Content-Type')
        response.headers.add('Access-Control-Allow-Methods', 'POST, OPTIONS')
        return response, 200

    try:
        # Validate file exists
        file = request.files.get('image')
        if not file:
            return jsonify({'error': 'No image file provided in request'}), 400
        
        # Validate file name
        if file.filename == '':
            return jsonify({'error': 'No file selected'}), 400
        
        # Validate file extension
        allowed_extensions = {'jpg', 'jpeg', 'png', 'gif', 'bmp'}
        file_ext = file.filename.rsplit('.', 1)[1].lower() if '.' in file.filename else ''
        if file_ext not in allowed_extensions:
            return jsonify({'error': f'Invalid file type. Allowed: {", ".join(allowed_extensions)}'}), 400
        
        # Read and validate image
        file_data = file.read()
        if len(file_data) == 0:
            return jsonify({'error': 'Empty file provided'}), 400
        
        arr = np.frombuffer(file_data, dtype=np.uint8)
        img = cv2.imdecode(arr, cv2.IMREAD_COLOR)
        
        if img is None:
            return jsonify({'error': 'Failed to decode image. Ensure it is a valid image file'}), 400
        
        if img.size == 0:
            return jsonify({'error': 'Image has no data'}), 400
        
        # Detect persons using YOLO
        results = detector(img, classes=[0])
        
        if len(results[0].boxes) == 0:
            return jsonify({'error': 'No person detected in the image. Please upload a clear photo with your full body visible'}), 400
        
        # Get largest detection
        best = max(results[0].boxes, key=lambda b: (b.xyxy[0][2]-b.xyxy[0][0])*(b.xyxy[0][3]-b.xyxy[0][1]))
        x1, y1, x2, y2 = map(int, best.xyxy[0])
        
        # Validate crop dimensions
        if x1 >= x2 or y1 >= y2:
            return jsonify({'error': 'Invalid detection box dimensions'}), 400
        
        crop = img[y1:y2, x1:x2]
        
        if crop.size == 0:
            return jsonify({'error': 'Crop area is empty'}), 400
        
        # Extract keypoints
        kpts = extract_keypoints(crop)
        
        if kpts is None or len(kpts) == 0:
            return jsonify({'error': 'Failed to extract body keypoints. Please ensure you are in a clear, well-lit area'}), 400
        
        # Compute posture features
        feats = compute_posture_features(kpts)
        
        # Apply user calibration if provided
        user_id = request.form.get('user_id', None)
        if user_id:
            feats = apply_calibration(user_id, feats)
        
        # Classify posture
        label, severity, confidence, suggestions = classify_posture_from_features(feats)
        
        # Extract detailed issues from label
        detected_issues = [issue.strip() for issue in label.split(';') if issue.strip()]
        
        # Get AI recommendation if Botpress is configured
        ai_recommendation = None
        conversation_id = None
        if BOTPRESS_AVAILABLE:
            try:
                ai_query = f"""I analyzed someone's posture from a photo and found:

                    Posture Status: {label}

                    Severity: {severity}
                    Confidence: {confidence * 100:.1f}%
                    Issues: {', '.join(detected_issues) if detected_issues else 'Good posture'}
                    Please provide: exercises to fix these issues."""
                print(label)
                print(confidence)
                # Call Botpress REST API
                headers = {
                    'Authorization': f'Bearer {BOTPRESS_PAT}',
                    'Content-Type': 'application/json'
                }
                
                conversation_id = str(uuid.uuid4())
                payload = {'text': ai_query, 'type': 'text'}
                
                url = f"https://api.botpress.cloud/v1/bots/{BOTPRESS_BOT_ID}/conversations/{conversation_id}/messages"
                resp = requests.post(url, json=payload, headers=headers, timeout=5)
                resp.raise_for_status()
                
                result = resp.json()
                ai_recommendation = result.get('response', 'Exercise recommendations will be generated')
            except Exception as e:
                print(f'AI recommendation error: {str(e)}')
                ai_recommendation = None
        
        response = jsonify({
            'label': label,
            'severity': severity,
            'confidence': confidence,
            'features': feats,
            'suggestions': suggestions,
            'detected_issues': detected_issues,
            'ai_recommendation': ai_recommendation,
            'conversation_id': conversation_id,
            'status': 'success'
        })
        response.headers.add('Access-Control-Allow-Origin', '*')
        return response
        
    except ValueError as ve:
        return jsonify({'error': f'Validation error: {str(ve)}'}), 400
    except Exception as e:
        import traceback
        error_trace = traceback.format_exc()
        print(f'Upload error: {error_trace}')
        return jsonify({'error': f'Server error during posture analysis: {str(e)}'}), 500

@app.route('/recommendation')
def recommendation():
    with state_lock:
        if not last_postures:
            return jsonify({'recommendations': [], 'message': 'No posture data yet. Keep the camera on for analysis.'})
        
        recommendations = []
        for sid, persons in last_postures.items():
            for pid, info in persons.items():
                tips = " ".join(info.get('suggestions', []))
                recommendation_item = {
                    'stream_id': sid,
                    'person_id': pid,
                    'label': info.get('label', 'Unknown'),
                    'severity': info.get('severity', 'unknown'),
                    'confidence': info.get('confidence', 0),
                    'tips': tips or 'Maintain neutral alignment.'
                }
                recommendations.append(recommendation_item)
        
        if not recommendations:
            return jsonify({'recommendations': [], 'message': 'All good. Maintain neutral alignment.'})
        
        # Sort by severity (bad first) and confidence
        severity_order = {'severe': 0, 'moderate': 1, 'good': 2}
        recommendations.sort(key=lambda x: (severity_order.get(x['severity'], 3), -x['confidence']))
        
        return jsonify({'recommendations': recommendations, 'message': f'Found {len(recommendations)} person(s) in frame'})

@app.route('/analytics')
def analytics():
    day = time.strftime("%Y-%m-%d")
    conn = sqlite3.connect(DB_PATH); c = conn.cursor()
    start = int(time.mktime(time.strptime(day, "%Y-%m-%d"))); end = start + 86400
    c.execute("SELECT confidence, severity FROM posture_events WHERE ts BETWEEN ? AND ?", (start, end))
    rows = c.fetchall(); conn.close()
    if not rows: return jsonify({'day': day, 'avg_confidence': None, 'bad_ratio': None, 'notes': 'No data today'})
    avg_conf = sum(r[0] for r in rows) / len(rows)
    bad_ratio = sum(1 for _, s in rows if s != 'good') / len(rows)
    return jsonify({'day': day, 'avg_confidence': round(avg_conf,2), 'bad_ratio': round(bad_ratio,2)})

@app.route('/manager/overview')
def manager_overview():
    conn = sqlite3.connect(DB_PATH); c = conn.cursor()
    c.execute("SELECT ts, stream_id, person_id, user_id, label, severity, confidence FROM posture_events ORDER BY ts DESC LIMIT 50")
    rows = c.fetchall(); conn.close()
    events = [{'ts': r[0], 'stream_id': r[1], 'person_id': r[2], 'user_id': r[3], 'label': r[4], 'severity': r[5], 'confidence': r[6]} for r in rows]
    return jsonify({'events': events})

@app.route('/chat', methods=['POST'])
def chat():
    data = request.get_json(silent=True) or {}
    question = data.get('question', '').strip()

    if not question:
        return jsonify({'error': 'Empty question'}), 400

    try:
        conversation_id = str(uuid.uuid4())

        if not text_generator:
            return jsonify({'error': 'Local NLP model not initialized. Install transformers: pip install transformers torch'}), 503

        # Use local text generation model
        prompt = f"Answer this question concisely: {question}"
        result = text_generator(prompt, max_length=150, do_sample=True)
        answer = result[0]['generated_text'].strip() if result else 'No response generated.'

        return jsonify({
            'status': 'success',
            'question': question,
            'answer': answer,
            'conversation_id': conversation_id
        })

    except Exception as e:
        print(f"❌ Chatbot error: {str(e)}")
        return jsonify({'error': f'Chatbot request failed: {str(e)}'}), 500


@app.route('/get_chatbot_advice', methods=['POST'])
def get_chatbot_advice():
    data = request.get_json(silent=True) or {}
    user_id = data.get('user_id', 'anonymous')
    posture_summary = data.get('posture_summary', '').strip() 

    if not posture_summary:
        posture_summary = "All currently tracked individuals show generally good posture. The user wants general tips for maintaining neutral alignment and avoiding stiffness during long work sessions."

    ai_query = f"""
        I am an AI Posture Analysis server for user: {user_id}. I have detected the following current or recent posture issues from video analysis:
        {posture_summary}
        Please provide a conversational, easy-to-understand response focused on immediate steps, simple exercises, and general ergonomic advice (e.g., monitor height, break reminders) to correct the specific issues mentioned. Frame the response as an empathetic digital coach.
        """

    try:
        conversation_id = str(uuid.uuid4())

        if not text_generator:
            return jsonify({'error': 'Local NLP model not initialized. Install transformers: pip install transformers torch'}), 503

        # Use local text generation model for posture advice
        prompt = f"As a posture coach, provide exercise recommendations for this situation: {ai_query}"
        result = text_generator(prompt, max_length=250, do_sample=True)
        chatbot_answer = result[0]['generated_text'].strip() if result else 'Unable to generate advice.'

        return jsonify({
            'status': 'success',
            'chatbot_response': chatbot_answer,
            'conversation_id': conversation_id
        })
    except Exception as e:
        print(f'❌ Chatbot Advice Error: {str(e)}')
        return jsonify({'error': f'Internal server error during chatbot analysis: {str(e)}'}), 500


  



@app.route('/posture_ai_analysis', methods=['POST'])
def posture_ai_analysis():
    """
    Analyzes posture and gets AI recommendations via Botpress
    """
    data = request.get_json(silent=True) or {}
    posture_label = data.get('posture_label', '')
    posture_severity = data.get('posture_severity', '')
    posture_confidence = data.get('posture_confidence', 0)
    detected_issues = data.get('detected_issues', [])
    
    if not posture_label:
        return jsonify({'error': 'Posture data required'}), 400
    
    issues_str = ', '.join(detected_issues) if detected_issues else 'No specific issues'
    
    ai_query = f"""I have analyzed someone's posture and found the following:

        Posture Status: {posture_label}
        Severity Level: {posture_severity}
        Confidence Score: {posture_confidence * 100:.1f}%
        Detected Issues: {issues_str}

        Please provide recommendations to improve posture."""

    try:
        conversation_id = str(uuid.uuid4())

        if not text_generator:
            return jsonify({'error': 'Local NLP model not initialized. Install transformers: pip install transformers torch'}), 503

        # Use local text generation model for posture analysis
        prompt = f"Provide posture improvement recommendations: {ai_query}"
        result = text_generator(prompt, max_length=250, do_sample=True)
        ai_recommendation = result[0]['generated_text'].strip() if result else 'Unable to generate recommendation'

        return jsonify({
            'status': 'success',
            'posture_label': posture_label,
            'ai_recommendation': ai_recommendation,
            'conversation_id': conversation_id
        })
    except Exception as e:
        print(f'❌ AI Analysis error: {str(e)}')
        return jsonify({'error': f'AI analysis failed: {str(e)}'}), 500




@app.route('/')
def index():
    return send_from_directory('../frontend', 'dashboard.html')

if __name__ == '__main__':
    socketio.run(app, host='0.0.0.0', port=5000)
