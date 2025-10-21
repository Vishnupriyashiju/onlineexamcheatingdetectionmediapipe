from flask import Flask, render_template, Response, request, redirect, url_for, jsonify
import cv2
import torch
import mediapipe as mp
import os
import time

app = Flask(__name__)
app.secret_key = "exam_secret_key"

# -------------------------------
# Load YOLOv5 model
# -------------------------------
model = torch.hub.load('yolov5', 'custom', path='best.pt', source='local', force_reload=True)
print("✅ YOLO model loaded:", model.names)

# -------------------------------
# Initialize MediaPipe
# -------------------------------
mp_face_mesh = mp.solutions.face_mesh
face_mesh = mp_face_mesh.FaceMesh(refine_landmarks=True, max_num_faces=2)

# -------------------------------
# Globals & Parameters
# -------------------------------
confidence_threshold = 0.45
HEAD_TURN_THRESHOLD = 18     # left/right turn threshold
LIP_MOVEMENT_THRESHOLD = 3.5
LOOK_DOWN_THRESHOLD = 10     # looking down within this range won't count as cheating
FRAME_SKIP = 5
HOLD_TIME = 2.5
EXAM_DURATION = 60  # seconds

cheating_count = 0
non_cheating_count = 0
last_state = "No Face Detected"
last_detection_time = 0
frame_counter = 0
exam_start_time = None
exam_terminated = False
tab_switch_count = 0


# -------------------------------
# Detection Logic
# -------------------------------
def detect_cheating(frame):
    global cheating_count, non_cheating_count, last_detection_time, last_state, frame_counter

    frame_counter += 1
    h, w, _ = frame.shape

    if frame_counter % FRAME_SKIP != 0:
        color = (0, 0, 255) if last_state == "Cheating" else (0, 255, 0) if last_state == "Not Cheating" else (255, 255, 0)
        cv2.putText(frame, last_state, (20, 50), cv2.FONT_HERSHEY_SIMPLEX, 0.9, color, 2)
        return frame

    cheating_detected = False
    face_detected = False

    # ---------- 1️⃣ MediaPipe Detection ----------
    frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    results_mp = face_mesh.process(frame_rgb)

    if results_mp.multi_face_landmarks:
        face_detected = True

        # 🚨 Multiple faces → cheating
        if len(results_mp.multi_face_landmarks) > 1:
            cheating_detected = True
        else:
            for face_landmarks in results_mp.multi_face_landmarks:
                left_eye = face_landmarks.landmark[33]
                right_eye = face_landmarks.landmark[263]
                nose_tip = face_landmarks.landmark[1]
                top_lip = face_landmarks.landmark[13]
                bottom_lip = face_landmarks.landmark[14]

                face_center_x = (left_eye.x + right_eye.x) / 2
                head_turn = (nose_tip.x - face_center_x) * 100
                lip_distance = abs(top_lip.y - bottom_lip.y) * h * 10
                head_tilt = (nose_tip.y - ((left_eye.y + right_eye.y) / 2)) * 100

                # ✅ looking down only (not cheating)
                if abs(head_turn) < HEAD_TURN_THRESHOLD and head_tilt > LOOK_DOWN_THRESHOLD:
                    cheating_detected = False
                # 🚨 head turn left/right or lip movement = cheating
                elif abs(head_turn) > HEAD_TURN_THRESHOLD or lip_distance > LIP_MOVEMENT_THRESHOLD:
                    cheating_detected = True
                else:
                    cheating_detected = False

    # ---------- 2️⃣ YOLO Fallback ----------
    if not face_detected:
        results = model(frame)
        detections = results.xyxy[0]

        if len(detections) > 1:
            cheating_detected = True  # multiple people
        else:
            for *box, conf, cls in detections:
                if conf >= confidence_threshold:
                    label = model.names[int(cls)].lower()
                    if label == "cheating":
                        cheating_detected = True
                    elif label == "non-cheating":
                        cheating_detected = False
                    face_detected = True
                    break

    # ---------- 3️⃣ Update Detection ----------
    current_time = time.time()
    if current_time - last_detection_time > HOLD_TIME:
        if not face_detected:
            last_state = "No Face Detected"
        elif cheating_detected:
            last_state = "Cheating"
            cheating_count += 1
        else:
            last_state = "Not Cheating"
            non_cheating_count += 1
        last_detection_time = current_time

    # ---------- 4️⃣ Display ----------
    color = (0, 0, 255) if last_state == "Cheating" else (0, 255, 0) if last_state == "Not Cheating" else (0, 255, 255)
    cv2.putText(frame, last_state, (20, 50), cv2.FONT_HERSHEY_SIMPLEX, 0.9, color, 2)

    return frame


# -------------------------------
# Video Stream
# -------------------------------
def generate_frames():
    global exam_start_time, exam_terminated
    cap = cv2.VideoCapture(0)
    while True:
        success, frame = cap.read()
        if not success:
            break

        frame = cv2.flip(frame, 1)
        if exam_terminated or not exam_start_time or (time.time() - exam_start_time > EXAM_DURATION):
            break

        frame = detect_cheating(frame)
        ret, buffer = cv2.imencode('.jpg', frame)
        yield (b'--frame\r\nContent-Type: image/jpeg\r\n\r\n' + buffer.tobytes() + b'\r\n')
    cap.release()


# -------------------------------
# Routes
# -------------------------------
@app.route('/')
def index():
    return render_template('login.html')


@app.route('/login', methods=['POST'])
def login():
    global exam_start_time, exam_terminated, cheating_count, non_cheating_count, tab_switch_count

    username = request.form['username']
    password = request.form['password']
    role = request.form['role']

    if role == "admin" and username == "admin" and password == "admin":
        return redirect(url_for('admin_dashboard'))
    elif role == "student" and username == "student" and password == "123":
        exam_start_time = time.time()
        exam_terminated = False
        cheating_count = 0
        non_cheating_count = 0
        tab_switch_count = 0
        return render_template('index.html', exam_duration=EXAM_DURATION)
    else:
        return render_template('login.html', error="Invalid credentials")


@app.route('/video_feed')
def video_feed():
    return Response(generate_frames(), mimetype='multipart/x-mixed-replace; boundary=frame')


@app.route('/admin_dashboard')
def admin_dashboard():
    conclusion = "Cheating Detected" if cheating_count > non_cheating_count else "No Cheating"
    total = cheating_count + non_cheating_count
    return render_template('admin_dashboard.html',
                           total=total,
                           cheating=cheating_count,
                           conclusion=conclusion,
                           current_timer=EXAM_DURATION)


@app.route('/set_timer', methods=['POST'])
def set_timer():
    global EXAM_DURATION
    new_time = int(request.form['timer'])
    EXAM_DURATION = new_time * 60
    return redirect(url_for('admin_dashboard'))


@app.route('/tab_switch', methods=['POST'])
def tab_switch():
    global tab_switch_count, exam_terminated
    tab_switch_count += 1
    if tab_switch_count == 1:
        return jsonify({"warning": True})
    else:
        exam_terminated = True
        return jsonify({"terminated": True})


@app.route('/result')
def result():
    conclusion = "Cheating Detected" if cheating_count > non_cheating_count else "No Cheating"
    return render_template('result.html',
                           total=cheating_count + non_cheating_count,
                           cheating=cheating_count,
                           non_cheating=non_cheating_count,
                           conclusion=conclusion)


if __name__ == '__main__':
    os.makedirs("static", exist_ok=True)
    app.run(debug=True)
"""from flask import Flask, render_template, Response, request, redirect, url_for, session
import cv2
import torch
import time
import os
from werkzeug.utils import secure_filename

app = Flask(__name__, static_folder='statics')
app.secret_key = "supersecretkey"

users = {
    "student": {"password": "123", "role": "student"},
    "admin": {"password": "admin", "role": "admin"}
}

# Load YOLOv5 model
model = torch.hub.load('yolov5', 'custom', path='best.pt', source='local', force_reload=True)
print("✅ Model Loaded:", model.names)

# Globals
recording = False
cheating_count = 0
non_cheating_count = 0
cheating_reasons = {}
cap = cv2.VideoCapture(0)

cheating_classes = [0]  # your cheating labels

# Detection tuning
confidence_threshold = 0.25      # was 0.65 → now slightly relaxed
hold_time = 2                    # was 3 → faster response
frame_skip = 3
resize_dim = (640, 480)
frame_count = 0

last_state = "Not Cheating"
last_reason = "None"
last_change_time = time.time()

UPLOAD_FOLDER = 'static/photos'
os.makedirs(UPLOAD_FOLDER, exist_ok=True)
students = {}

@app.route('/')
def index():
    return redirect(url_for("login"))

@app.route('/register', methods=['GET', 'POST'])
def register():
    if request.method == 'POST':
        username = request.form['username']
        password = request.form['password']
        email = request.form['email']
        roll_no = request.form['roll_no']
        photo = request.files['photo']

        if username in students:
            return "❌ Username already exists", 400

        filename = secure_filename(f"{username}_{photo.filename}")
        filepath = os.path.join(UPLOAD_FOLDER, filename)
        photo.save(filepath)

        students[username] = {
            "password": password,
            "email": email,
            "roll_no": roll_no,
            "photo": filepath
        }
        return redirect(url_for('login'))
    return render_template('register.html')

@app.route('/login', methods=['GET', 'POST'])
def login():
    if request.method == 'POST':
        username = request.form['username']
        password = request.form['password']
        role = request.form['role']

        if role == "student" and username in users and users[username]["password"] == password:
            session['role'] = 'student'
            session['username'] = username
            return redirect(url_for('student_exam'))
        elif role == "admin" and username in users and users[username]["password"] == password:
            session['role'] = 'admin'
            session['username'] = username
            return redirect(url_for('admin_dashboard'))
        else:
            return "❌ Invalid credentials", 401
    return render_template("login.html")

@app.route('/student_exam')
def student_exam():
    if session.get("role") != "student":
        return redirect(url_for("login"))
    return render_template("index.html")

@app.route('/admin_dashboard')
def admin_dashboard():
    if session.get("role") != "admin":
        return redirect(url_for("login"))
    return render_template("admin_dashboard.html")

@app.route('/video_feed')
def video_feed():
    return Response(generate_frames(), mimetype='multipart/x-mixed-replace; boundary=frame')

@app.route('/start')
def start_recording():
    global recording, cheating_count, non_cheating_count, cheating_reasons
    recording = True
    cheating_count = 0
    non_cheating_count = 0
    cheating_reasons = {}
    return "Recording started!"

@app.route('/stop')
def stop_recording():
    global recording, cheating_count, non_cheating_count, cheating_reasons
    recording = False

    total = cheating_count + non_cheating_count
    cheating_ratio = (cheating_count / total) if total > 0 else 0
    threshold = 0.4

    conclusion = "Cheating Detected" if cheating_ratio > threshold else "No Cheating Detected"

    return render_template(
        "result.html",
        total=total,
        cheating=cheating_count,
        non_cheating=non_cheating_count,
        conclusion=conclusion,
        reasons=cheating_reasons
    )

def generate_frames():
    global recording, cheating_count, non_cheating_count, cheating_reasons
    global last_state, last_reason, last_change_time, frame_count
    
    while True:
        ret, frame = cap.read()
        if not ret:
            break

        frame_count += 1
        
        if recording and frame_count % frame_skip == 0:
            small_frame = cv2.resize(frame, resize_dim)
            results = model(small_frame)
            detections = results.xyxy[0]

            current_detection = "Not Cheating"
            reason = "None"
            
            valid_detections = []
            frame_area = small_frame.shape[0] * small_frame.shape[1]
            min_box_area = 0.01 * frame_area 

            for *box, conf, cls in detections:
                x1, y1, x2, y2 = box
                box_area = (x2 - x1) * (y2 - y1)
                if conf >= confidence_threshold and box_area > min_box_area:
                    valid_detections.append((int(cls), float(conf), box))

            # Prioritize multiple-person detection
            person_class_id = 0
            persons = [d for d in valid_detections if d[0] == person_class_id]
            if len(persons) > 1:
                current_detection = "Cheating"
                reason = "Multiple People"
            else:
                # Then check for other cheating behaviors
                cheating_detected = False
                for cls_id, conf, box in valid_detections:
                    if cls_id in cheating_classes:
                        cheating_detected = True
                        reason = model.names[cls_id]
                        break
                
                if cheating_detected:
                    current_detection = "Cheating"
                
            # State persistence (keep this logic)
            current_time = time.time()
            if current_detection != last_state:
                if (current_time - last_change_time) > hold_time:
                    last_state = current_detection
                    last_reason = reason
                    last_change_time = current_time
            else:
                current_detection = last_state
                reason = last_reason

            # Counting and text overlay logic
            if current_detection == "Cheating":
                cheating_count += 1
                if reason not in ["None", "Not Cheating"]:
                    cheating_reasons[reason] = cheating_reasons.get(reason, 0) + 1
                color = (0, 0, 255)
                label_text = f"Cheating: {reason} ({cheating_count})"
            else:
                non_cheating_count += 1
                color = (0, 255, 0)
                label_text = f"Not Cheating ({non_cheating_count})"

            cv2.putText(frame, label_text, (30, 50), cv2.FONT_HERSHEY_SIMPLEX, 1, color, 2)
            
        _, buffer = cv2.imencode('.jpg', frame)
        yield (b'--frame\r\n'
               b'Content-Type: image/jpeg\r\n\r\n' + buffer.tobytes() + b'\r\n')

if __name__ == '__main__':
    app.run(debug=True)"""
"""from flask import Flask, render_template, Response
import cv2
import torch
import mediapipe as mp
import time
import os

app = Flask(__name__)

# -------------------------------
# Load YOLOv5 model
# -------------------------------
model = torch.hub.load('yolov5', 'custom', path='best.pt', source='local', force_reload=True)
print("✅ YOLO model loaded:", model.names)

# -------------------------------
# Initialize MediaPipe
# -------------------------------
mp_face_mesh = mp.solutions.face_mesh
face_mesh = mp_face_mesh.FaceMesh(refine_landmarks=True, max_num_faces=1)

# -------------------------------
# Globals & Parameters
# -------------------------------
cheating_count = 0
non_cheating_count = 0
confidence_threshold = 0.45
HEAD_TURN_THRESHOLD = 15
LIP_MOVEMENT_THRESHOLD = 3.5

# -------------------------------
# YOLO + MediaPipe Detection
# -------------------------------
def detect_cheating(frame):
    global cheating_count, non_cheating_count
    h, w, _ = frame.shape

    # ---------- 1️⃣ YOLO DETECTION ----------
    results = model(frame)
    detections = results.xyxy[0]
    cheating_detected = False
    reason = "None"

    for *box, conf, cls in detections:
        if conf >= confidence_threshold:
            label = model.names[int(cls)]
            if label.lower() == "cheating":
                cheating_detected = True
                reason = "YOLO: Cheating Detected"
                break
            elif label.lower() == "non-cheating":
                cheating_detected = False
                reason = "YOLO: Non-Cheating"
                break

    # ---------- 2️⃣ MEDIAPIPE FALLBACK ----------
    if len(detections) == 0:  # No YOLO result
        frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        results_mp = face_mesh.process(frame_rgb)

        if results_mp.multi_face_landmarks:
            for face_landmarks in results_mp.multi_face_landmarks:
                left_eye = face_landmarks.landmark[33]
                right_eye = face_landmarks.landmark[263]
                nose_tip = face_landmarks.landmark[1]
                top_lip = face_landmarks.landmark[13]
                bottom_lip = face_landmarks.landmark[14]

                face_center_x = (left_eye.x + right_eye.x) / 2
                head_turn = (nose_tip.x - face_center_x) * 100
                lip_distance = abs(top_lip.y - bottom_lip.y) * h * 10

                if head_turn > HEAD_TURN_THRESHOLD:
                    cheating_detected = True
                    reason = "MediaPipe: Head turned right"
                elif head_turn < -HEAD_TURN_THRESHOLD:
                    cheating_detected = True
                    reason = "MediaPipe: Head turned left"
                elif lip_distance > LIP_MOVEMENT_THRESHOLD:
                    cheating_detected = True
                    reason = "MediaPipe: Lip movement detected"
                else:
                    cheating_detected = False
                    reason = "MediaPipe: Normal posture"

    # ---------- 3️⃣ COUNTS & DISPLAY ----------
    if cheating_detected:
        cheating_count += 1
        color = (0, 0, 255)
        label_text = f"Cheating ({cheating_count}) - {reason}"
        if not os.path.exists("static/cheating_sample.jpg"):
            cv2.imwrite("static/cheating_sample.jpg", frame)
    else:
        non_cheating_count += 1
        color = (0, 255, 0)
        label_text = f"Not Cheating ({non_cheating_count}) - {reason}"
        if not os.path.exists("static/non_cheating_sample.jpg"):
            cv2.imwrite("static/non_cheating_sample.jpg", frame)

    cv2.putText(frame, label_text, (20, 50), cv2.FONT_HERSHEY_SIMPLEX, 0.9, color, 2)
    return frame


# -------------------------------
# Flask Routes
# -------------------------------
def generate_frames():
    cap = cv2.VideoCapture(0)
    while True:
        success, frame = cap.read()
        if not success:
            break
        frame = cv2.flip(frame, 1)
        frame = detect_cheating(frame)

        ret, buffer = cv2.imencode('.jpg', frame)
        frame = buffer.tobytes()
        yield (b'--frame\r\n'
               b'Content-Type: image/jpeg\r\n\r\n' + frame + b'\r\n')


@app.route('/')
def index():
    return render_template('index.html')


@app.route('/video_feed')
def video_feed():
    return Response(generate_frames(), mimetype='multipart/x-mixed-replace; boundary=frame')


if __name__ == '__main__':
    os.makedirs("static", exist_ok=True)
    app.run(debug=True)"""
"""from flask import Flask, render_template, Response
import cv2
import torch
import mediapipe as mp
import os
import time

app = Flask(__name__)

# -------------------------------
# Load YOLOv5 model
# -------------------------------
model = torch.hub.load('yolov5', 'custom', path='best.pt', source='local', force_reload=True)
print("✅ YOLO model loaded:", model.names)

# -------------------------------
# Initialize MediaPipe
# -------------------------------
mp_face_mesh = mp.solutions.face_mesh
face_mesh = mp_face_mesh.FaceMesh(refine_landmarks=True, max_num_faces=1)

# -------------------------------
# Globals & Parameters
# -------------------------------
cheating_count = 0
non_cheating_count = 0
confidence_threshold = 0.45
HEAD_TURN_THRESHOLD = 15
LIP_MOVEMENT_THRESHOLD = 3.5

FRAME_SKIP = 5      # Process every 5th frame
HOLD_TIME = 2.5     # Keep last result for 2.5 seconds

last_detection_time = 0
last_state = "No Face Detected"
frame_counter = 0


# -------------------------------
# Detection Logic
# -------------------------------
def detect_cheating(frame):
    global cheating_count, non_cheating_count, last_detection_time, last_state, frame_counter

    frame_counter += 1
    h, w, _ = frame.shape

    # Only process every Nth frame
    if frame_counter % FRAME_SKIP != 0:
        label_text = last_state
        color = (0, 0, 255) if last_state == "Cheating" else (0, 255, 0) if last_state == "Not Cheating" else (255, 255, 0)
        cv2.putText(frame, label_text, (20, 50), cv2.FONT_HERSHEY_SIMPLEX, 0.9, color, 2)
        return frame

    cheating_detected = False
    face_detected = False

    # ---------- 1️⃣ Try MediaPipe First ----------
    frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    results_mp = face_mesh.process(frame_rgb)

    if results_mp.multi_face_landmarks:
        face_detected = True
        for face_landmarks in results_mp.multi_face_landmarks:
            left_eye = face_landmarks.landmark[33]
            right_eye = face_landmarks.landmark[263]
            nose_tip = face_landmarks.landmark[1]
            top_lip = face_landmarks.landmark[13]
            bottom_lip = face_landmarks.landmark[14]

            face_center_x = (left_eye.x + right_eye.x) / 2
            head_turn = (nose_tip.x - face_center_x) * 100
            lip_distance = abs(top_lip.y - bottom_lip.y) * h * 10

            if abs(head_turn) > HEAD_TURN_THRESHOLD or lip_distance > LIP_MOVEMENT_THRESHOLD:
                cheating_detected = True
                break

    # ---------- 2️⃣ Fallback to YOLO if no face detected ----------
    if not face_detected:
        results = model(frame)
        detections = results.xyxy[0]
        for *box, conf, cls in detections:
            if conf >= confidence_threshold:
                label = model.names[int(cls)].lower()
                if label == "cheating":
                    cheating_detected = True
                    face_detected = True
                elif label == "non-cheating":
                    cheating_detected = False
                    face_detected = True
                break

    # ---------- 3️⃣ Update Detection State ----------
    current_time = time.time()
    if current_time - last_detection_time > HOLD_TIME:
        if not face_detected:
            last_state = "No Face Detected"
        elif cheating_detected:
            last_state = "Cheating"
            cheating_count += 1
        else:
            last_state = "Not Cheating"
            non_cheating_count += 1
        last_detection_time = current_time

    # ---------- 4️⃣ Display ----------
    color = (0, 0, 255) if last_state == "Cheating" else (0, 255, 0) if last_state == "Not Cheating" else (255, 255, 0)
    cv2.putText(frame, last_state, (20, 50), cv2.FONT_HERSHEY_SIMPLEX, 0.9, color, 2)

    # Save sample images
    if last_state == "Cheating" and not os.path.exists("static/cheating_sample.jpg"):
        cv2.imwrite("static/cheating_sample.jpg", frame)
    elif last_state == "Not Cheating" and not os.path.exists("static/non_cheating_sample.jpg"):
        cv2.imwrite("static/non_cheating_sample.jpg", frame)

    return frame


# -------------------------------
# Flask Routes
# -------------------------------
def generate_frames():
    cap = cv2.VideoCapture(0)
    while True:
        success, frame = cap.read()
        if not success:
            break
        frame = cv2.flip(frame, 1)
        frame = detect_cheating(frame)

        ret, buffer = cv2.imencode('.jpg', frame)
        frame = buffer.tobytes()
        yield (b'--frame\r\n'
               b'Content-Type: image/jpeg\r\n\r\n' + frame + b'\r\n')


@app.route('/')
def index():
    return render_template('index.html')


@app.route('/video_feed')
def video_feed():
    return Response(generate_frames(), mimetype='multipart/x-mixed-replace; boundary=frame')


if __name__ == '__main__':
    os.makedirs("static", exist_ok=True)
    app.run(debug=True)"""
"""from flask import Flask, render_template, Response, request, redirect, url_for, session, jsonify
import cv2
import torch
import mediapipe as mp
import os
import time

app = Flask(__name__)
app.secret_key = "exam_secret_key"

# -------------------------------
# Load YOLOv5 model
# -------------------------------
model = torch.hub.load('yolov5', 'custom', path='best.pt', source='local', force_reload=True)
print("✅ YOLO model loaded:", model.names)

# -------------------------------
# Initialize MediaPipe
# -------------------------------
mp_face_mesh = mp.solutions.face_mesh
face_mesh = mp_face_mesh.FaceMesh(refine_landmarks=True, max_num_faces=1)

# -------------------------------
# Globals & Parameters
# -------------------------------
confidence_threshold = 0.45
HEAD_TURN_THRESHOLD = 15
LIP_MOVEMENT_THRESHOLD = 3.5
FRAME_SKIP = 5
HOLD_TIME = 2.5
EXAM_DURATION = 60  # default 1 minute

cheating_count = 0
non_cheating_count = 0
last_state = "No Face Detected"
last_detection_time = 0
frame_counter = 0
exam_start_time = None
exam_terminated = False
tab_switch_count = 0


# -------------------------------
# Detection Logic
# -------------------------------
def detect_cheating(frame):
    global cheating_count, non_cheating_count, last_detection_time, last_state, frame_counter

    frame_counter += 1
    h, w, _ = frame.shape

    # Only process every Nth frame
    if frame_counter % FRAME_SKIP != 0:
        label_text = last_state
        color = (0, 0, 255) if last_state == "Cheating" else (0, 255, 0) if last_state == "Not Cheating" else (255, 255, 0)
        cv2.putText(frame, label_text, (20, 50), cv2.FONT_HERSHEY_SIMPLEX, 0.9, color, 2)
        return frame

    cheating_detected = False
    face_detected = False

    # ---------- 1️⃣ Try MediaPipe First ----------
    frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    results_mp = face_mesh.process(frame_rgb)

    if results_mp.multi_face_landmarks:
        face_detected = True
        for face_landmarks in results_mp.multi_face_landmarks:
            left_eye = face_landmarks.landmark[33]
            right_eye = face_landmarks.landmark[263]
            nose_tip = face_landmarks.landmark[1]
            top_lip = face_landmarks.landmark[13]
            bottom_lip = face_landmarks.landmark[14]

            face_center_x = (left_eye.x + right_eye.x) / 2
            head_turn = (nose_tip.x - face_center_x) * 100
            lip_distance = abs(top_lip.y - bottom_lip.y) * h * 10

            if abs(head_turn) > HEAD_TURN_THRESHOLD or lip_distance > LIP_MOVEMENT_THRESHOLD:
                cheating_detected = True
                break

    # ---------- 2️⃣ Fallback to YOLO if no face detected ----------
    if not face_detected:
        results = model(frame)
        detections = results.xyxy[0]
        for *box, conf, cls in detections:
            if conf >= confidence_threshold:
                label = model.names[int(cls)].lower()
                if label == "cheating":
                    cheating_detected = True
                    face_detected = True
                elif label == "non-cheating":
                    cheating_detected = False
                    face_detected = True
                break

    # ---------- 3️⃣ Update Detection State ----------
    current_time = time.time()
    if current_time - last_detection_time > HOLD_TIME:
        if not face_detected:
            last_state = "No Face Detected"
        elif cheating_detected:
            last_state = "Cheating"
            cheating_count += 1
        else:
            last_state = "Not Cheating"
            non_cheating_count += 1
        last_detection_time = current_time

    # ---------- 4️⃣ Display ----------
    color = (0, 0, 255) if last_state == "Cheating" else (0, 255, 0) if last_state == "Not Cheating" else (255, 255, 0)
    cv2.putText(frame, last_state, (20, 50), cv2.FONT_HERSHEY_SIMPLEX, 0.9, color, 2)
    return frame


# -------------------------------
# Video Stream
# -------------------------------
def generate_frames():
    global exam_start_time, exam_terminated
    cap = cv2.VideoCapture(0)
    while True:
        success, frame = cap.read()
        if not success:
            break

        frame = cv2.flip(frame, 1)

        # Stop if time’s up or terminated
        if exam_terminated or not exam_start_time or (time.time() - exam_start_time > EXAM_DURATION):
            break

        frame = detect_cheating(frame)
        ret, buffer = cv2.imencode('.jpg', frame)
        yield (b'--frame\r\nContent-Type: image/jpeg\r\n\r\n' + buffer.tobytes() + b'\r\n')
    cap.release()


# -------------------------------
# Routes
# -------------------------------
@app.route('/')
def index():
    return render_template('login.html')


@app.route('/login', methods=['POST'])
def login():
    global exam_start_time, exam_terminated, cheating_count, non_cheating_count, tab_switch_count

    username = request.form['username']
    password = request.form['password']
    role = request.form['role']

    if role == "admin" and username == "admin" and password == "admin":
        return redirect(url_for('admin_dashboard'))
    elif role == "student" and username == "student" and password == "123":
        exam_start_time = time.time()
        exam_terminated = False
        cheating_count = 0
        non_cheating_count = 0
        tab_switch_count = 0
        return redirect(url_for('student_exam'))
    else:
        return render_template('login.html', error="Invalid credentials")


@app.route('/student_exam')
def student_exam():
    return render_template('index.html')


@app.route('/video_feed')
def video_feed():
    return Response(generate_frames(), mimetype='multipart/x-mixed-replace; boundary=frame')


@app.route('/admin_dashboard')
def admin_dashboard():
    global cheating_count, non_cheating_count, exam_start_time, exam_terminated
    elapsed = 0 if not exam_start_time else int(time.time() - exam_start_time)
    remaining = max(0, int(EXAM_DURATION - elapsed))
    conclusion = "Cheating Detected" if cheating_count > non_cheating_count else "No Cheating"
    return render_template('admin_dashboard.html',
                           total=cheating_count + non_cheating_count,
                           cheating=cheating_count,
                           conclusion=conclusion,
                           remaining=remaining,
                           duration=EXAM_DURATION)


@app.route('/set_time', methods=['POST'])
def set_time():
    global EXAM_DURATION
    new_time = int(request.form['duration'])
    EXAM_DURATION = new_time
    return redirect(url_for('admin_dashboard'))


@app.route('/tab_switch', methods=['POST'])
def tab_switch():
    global tab_switch_count, exam_terminated
    tab_switch_count += 1
    if tab_switch_count == 1:
        return jsonify({"warning": True})
    else:
        exam_terminated = True
        return jsonify({"terminated": True})


@app.route('/result')
def result():
    conclusion = "Cheating Detected" if cheating_count > non_cheating_count else "No Cheating"
    return render_template('result.html',
                           total=cheating_count + non_cheating_count,
                           cheating=cheating_count,
                           non_cheating=non_cheating_count,
                           conclusion=conclusion)


if __name__ == '__main__':
    os.makedirs("static", exist_ok=True)
    app.run(debug=True)"""
from flask import Flask, render_template, Response, request, redirect, url_for, jsonify
import cv2
import torch
import mediapipe as mp
import os
import time

app = Flask(__name__)
app.secret_key = "exam_secret_key"

# -------------------------------
# Load YOLOv5 model
# -------------------------------
model = torch.hub.load('yolov5', 'custom', path='best.pt', source='local', force_reload=True)
print("✅ YOLO model loaded:", model.names)

# -------------------------------
# Initialize MediaPipe
# -------------------------------
mp_face_mesh = mp.solutions.face_mesh
face_mesh = mp_face_mesh.FaceMesh(refine_landmarks=True, max_num_faces=1)

# -------------------------------
# Globals & Parameters
# -------------------------------
confidence_threshold = 0.45
HEAD_TURN_THRESHOLD = 15
LIP_MOVEMENT_THRESHOLD = 3.5
FRAME_SKIP = 5
HOLD_TIME = 2.5
EXAM_DURATION = 60  # default 1 minute

cheating_count = 0
non_cheating_count = 0
last_state = "No Face Detected"
last_detection_time = 0
frame_counter = 0
exam_start_time = None
exam_terminated = False
tab_switch_count = 0


# -------------------------------
# Detection Logic
# -------------------------------
def detect_cheating(frame):
    global cheating_count, non_cheating_count, last_detection_time, last_state, frame_counter

    frame_counter += 1
    h, w, _ = frame.shape

    if frame_counter % FRAME_SKIP != 0:
        label_text = last_state
        color = (0, 0, 255) if last_state == "Cheating" else (0, 255, 0) if last_state == "Not Cheating" else (255, 255, 0)
        cv2.putText(frame, label_text, (20, 50), cv2.FONT_HERSHEY_SIMPLEX, 0.9, color, 2)
        return frame

    cheating_detected = False
    face_detected = False

    frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    results_mp = face_mesh.process(frame_rgb)

    if results_mp.multi_face_landmarks:
        face_detected = True
        for face_landmarks in results_mp.multi_face_landmarks:
            left_eye = face_landmarks.landmark[33]
            right_eye = face_landmarks.landmark[263]
            nose_tip = face_landmarks.landmark[1]
            top_lip = face_landmarks.landmark[13]
            bottom_lip = face_landmarks.landmark[14]

            face_center_x = (left_eye.x + right_eye.x) / 2
            head_turn = (nose_tip.x - face_center_x) * 100
            lip_distance = abs(top_lip.y - bottom_lip.y) * h * 10

            if abs(head_turn) > HEAD_TURN_THRESHOLD or lip_distance > LIP_MOVEMENT_THRESHOLD:
                cheating_detected = True
                break

    if not face_detected:
        results = model(frame)
        detections = results.xyxy[0]
        for *box, conf, cls in detections:
            if conf >= confidence_threshold:
                label = model.names[int(cls)].lower()
                if label == "cheating":
                    cheating_detected = True
                elif label == "non-cheating":
                    cheating_detected = False
                face_detected = True
                break

    current_time = time.time()
    if current_time - last_detection_time > HOLD_TIME:
        if not face_detected:
            last_state = "No Face Detected"
        elif cheating_detected:
            last_state = "Cheating"
            cheating_count += 1
        else:
            last_state = "Not Cheating"
            non_cheating_count += 1
        last_detection_time = current_time

    color = (0, 0, 255) if last_state == "Cheating" else (0, 255, 0) if last_state == "Not Cheating" else (255, 255, 0)
    cv2.putText(frame, last_state, (20, 50), cv2.FONT_HERSHEY_SIMPLEX, 0.9, color, 2)
    return frame


# -------------------------------
# Video Stream
# -------------------------------
def generate_frames():
    global exam_start_time, exam_terminated
    cap = cv2.VideoCapture(0)
    while True:
        success, frame = cap.read()
        if not success:
            break

        frame = cv2.flip(frame, 1)
        if exam_terminated or not exam_start_time or (time.time() - exam_start_time > EXAM_DURATION):
            break

        frame = detect_cheating(frame)
        ret, buffer = cv2.imencode('.jpg', frame)
        yield (b'--frame\r\nContent-Type: image/jpeg\r\n\r\n' + buffer.tobytes() + b'\r\n')
    cap.release()


# -------------------------------
# Routes
# -------------------------------
@app.route('/')
def index():
    return render_template('login.html')


@app.route('/login', methods=['POST'])
def login():
    global exam_start_time, exam_terminated, cheating_count, non_cheating_count, tab_switch_count

    username = request.form['username']
    password = request.form['password']
    role = request.form['role']

    if role == "admin" and username == "admin" and password == "admin":
        return redirect(url_for('admin_dashboard'))
    elif role == "student" and username == "student" and password == "123":
        exam_start_time = time.time()
        exam_terminated = False
        cheating_count = 0
        non_cheating_count = 0
        tab_switch_count = 0
        return render_template('index.html', exam_duration=EXAM_DURATION)
    else:
        return render_template('login.html', error="Invalid credentials")


@app.route('/video_feed')
def video_feed():
    return Response(generate_frames(), mimetype='multipart/x-mixed-replace; boundary=frame')


@app.route('/admin_dashboard')
def admin_dashboard():
    conclusion = "Cheating Detected" if cheating_count > non_cheating_count else "No Cheating"
    total = cheating_count + non_cheating_count
    return render_template('admin_dashboard.html',
                           total=total,
                           cheating=cheating_count,
                           conclusion=conclusion,
                           current_timer=EXAM_DURATION)


@app.route('/set_timer', methods=['POST'])
def set_timer():
    global EXAM_DURATION
    new_time = int(request.form['timer'])
    EXAM_DURATION = new_time * 60  # convert minutes to seconds
    return redirect(url_for('admin_dashboard'))


@app.route('/tab_switch', methods=['POST'])
def tab_switch():
    global tab_switch_count, exam_terminated
    tab_switch_count += 1
    if tab_switch_count == 1:
        return jsonify({"warning": True})
    else:
        exam_terminated = True
        return jsonify({"terminated": True})


@app.route('/result')
def result():
    conclusion = "Cheating Detected" if cheating_count > non_cheating_count else "No Cheating"
    return render_template('result.html',
                           total=cheating_count + non_cheating_count,
                           cheating=cheating_count,
                           non_cheating=non_cheating_count,
                           conclusion=conclusion)


if __name__ == '__main__':
    os.makedirs("static", exist_ok=True)
    app.run(debug=True)




