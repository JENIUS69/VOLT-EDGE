from flask import Flask, render_template_string, Response, jsonify, request, redirect, url_for
from flask_login import LoginManager, UserMixin, login_user, login_required, logout_user, current_user
import cv2
import numpy as np
import time
import requests
import threading
from datetime import datetime
from concurrent.futures import ThreadPoolExecutor
from ultralytics import YOLO
import mediapipe as mp

app = Flask(__name__)
app.secret_key = 'voltedge_vnit_key'

# --- 1. MULTI-ROLE AUTHENTICATION (RBAC) ---
login_manager = LoginManager()
login_manager.init_app(app)
login_manager.login_view = 'login'

class User(UserMixin):
    def __init__(self, id, username, password, role):
        self.id = id
        self.username = username
        self.password = password
        self.role = role

USERS = {
    'admin': User(1, 'admin', '12345678', 'admin'),
    'occupant': User(2, 'occupant', '12345678', 'occupant'),
    'staff': User(3, 'staff', '12345678', 'staff')
}

@login_manager.user_loader
def load_user(user_id):
    for u in USERS.values():
        if str(u.id) == str(user_id):
            return u
    return None

# --- 2. CONFIGURATION & STATE ---
ESP_IP = "http://192.168.4.1"
executor = ThreadPoolExecutor(max_workers=4)

# Custom Multi-Point Polygon Zones
POLYGON_ZONES = {
    "zone1": np.array([[20, 20], [300, 20], [260, 220], [20, 230]], np.int32),
    "zone2": np.array([[310, 20], [620, 20], [620, 230], [270, 220]], np.int32),
    "zone3": np.array([[20, 240], [310, 240], [310, 460], [20, 460]], np.int32),
    "zone4": np.array([[320, 240], [620, 240], [620, 460], [320, 460]], np.int32),
}

relay_states = {f"zone{i}": {"light": False, "fan": False} for i in range(1, 5)}
manual_overrides = {f"zone{i}": "AUTO" for i in range(1, 5)}
last_seen_times = {f"zone{i}": 0 for i in range(1, 5)}

service_alerts = []
total_idle_seconds = 0
POWER_RATING_KW_PER_ZONE = 0.08  # 15W Light + 65W Fan
TARIFF_PER_KWH = 10.00          # ₹10 / kWh

# --- 3. AI MODELS ---
yolo_model = YOLO("yolov8n.pt")
mp_hands = mp.solutions.hands
hands = mp_hands.Hands(max_num_hands=1, min_detection_confidence=0.7)

# --- 4. HIGH-SPEED DECOUPLED CAMERA CORE ---
class CameraStream:
    def __init__(self, src=0):
        self.cap = cv2.VideoCapture(src)
        self.ret, self.frame = self.cap.read()
        self.running = True
        self.thread = threading.Thread(target=self.update, args=())
        self.thread.daemon = True
        self.thread.start()

    def update(self):
        while self.running:
            ret, frame = self.cap.read()
            if ret:
                self.frame = cv2.resize(frame, (640, 480))
            time.sleep(0.01)

    def get_frame(self):
        return self.frame.copy() if self.frame is not None else None

cam = CameraStream()

# --- 5. HARDWARE & COMPUTER VISION PIPELINE ---
def async_send(url):
    try:
        requests.get(url, timeout=0.3)
    except Exception:
        pass

def update_relay(zone, appliance, turn_on):
    if relay_states[zone][appliance] != turn_on:
        relay_states[zone][appliance] = turn_on
        state_str = "on" if turn_on else "off"
        executor.submit(async_send, f"{ESP_IP}/{zone}/{appliance}/{state_str}")

last_alert_time = 0

def gen_frames():
    global total_idle_seconds, last_alert_time

    while True:
        frame = cam.get_frame()
        if frame is None:
            continue

        current_time = time.time()

        # 1. YOLOv8 Detection (Person Only - Class 0)
        results = yolo_model(frame, verbose=False, imgsz=320)
        for box in results[0].boxes:
            if int(box.cls[0]) == 0:
                x1, y1, x2, y2 = map(int, box.xyxy[0])
                cx, feet_y = (x1 + x2) // 2, y2

                for z_name, poly in POLYGON_ZONES.items():
                    if cv2.pointPolygonTest(poly, (cx, feet_y), False) >= 0:
                        last_seen_times[z_name] = current_time

                cv2.rectangle(frame, (x1, y1), (x2, y2), (0, 229, 255), 2)
                cv2.putText(frame, "PERSON", (x1, y1 - 8), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 229, 255), 2)

        # 2. Finger Counting Gesture AI
        rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        gesture_results = hands.process(rgb_frame)

        if gesture_results.multi_hand_landmarks and (current_time - last_alert_time > 4):
            for hand_landmarks in gesture_results.multi_hand_landmarks:
                lm = hand_landmarks.landmark

                fingers_up = 0
                if lm[8].y < lm[5].y: fingers_up += 1   # Index Finger
                if lm[12].y < lm[9].y: fingers_up += 1  # Middle Finger
                if lm[16].y < lm[13].y: fingers_up += 1 # Ring Finger
                if lm[20].y < lm[17].y: fingers_up += 1 # Pinky Finger

                alert_type = None
                if fingers_up == 1:
                    alert_type = "Water Request 💧"
                elif fingers_up == 3:
                    alert_type = "Coffee Request ☕"
                elif fingers_up >= 4:
                    alert_type = "Assistant Needed ✋"

                if alert_type:
                    alert_time_str = datetime.now().strftime("%H:%M:%S")
                    service_alerts.append({"type": alert_type, "time": alert_time_str})
                    last_alert_time = current_time
                    cv2.putText(frame, f"TRIGGERED: {alert_type}", (30, 40), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)

        # 3. Zone Automation & Savings Tracking
        for z_name, poly in POLYGON_ZONES.items():
            mode = manual_overrides[z_name]
            is_active = (current_time - last_seen_times[z_name]) <= 5.0

            if mode == "FORCE_ON":
                update_relay(z_name, "light", True)
                update_relay(z_name, "fan", True)
            elif mode == "FORCE_OFF":
                update_relay(z_name, "light", False)
                update_relay(z_name, "fan", False)
                total_idle_seconds += (1 / 30.0)
            else:  # AUTO
                update_relay(z_name, "light", is_active)
                update_relay(z_name, "fan", is_active)
                if not is_active:
                    total_idle_seconds += (1 / 30.0)

            color = (76, 175, 80) if is_active else (244, 67, 54)
            cv2.polylines(frame, [poly], True, color, 2)
            cv2.putText(frame, z_name.upper(), (poly[0][0] + 5, poly[0][1] + 20), cv2.FONT_HERSHEY_SIMPLEX, 0.5, color, 2)

        ret, buffer = cv2.imencode('.jpg', frame)
        yield (b'--frame\r\n' b'Content-Type: image/jpeg\r\n\r\n' + buffer.tobytes() + b'\r\n')

# --- 6. PROFESSIONAL DASHBOARD UI TEMPLATES ---
LOGIN_TEMPLATE = """
<!DOCTYPE html>
<html>
<head>
    <title>VoltEdge - Login</title>
    <style>
        body { background: #05070f; color: white; font-family: 'Segoe UI', sans-serif; display: flex; justify-content: center; align-items: center; height: 100vh; margin: 0; }
        .card { background: rgba(13, 21, 39, 0.8); border: 1px solid rgba(0, 229, 255, 0.2); backdrop-filter: blur(12px); border-radius: 16px; padding: 40px; width: 340px; box-shadow: 0 10px 30px rgba(0,0,0,0.5); text-align: center; }
        h1 { color: #00e5ff; margin-bottom: 24px; font-size: 24px; }
        input { width: 100%; padding: 12px; margin: 8px 0; background: rgba(255,255,255,0.05); border: 1px solid rgba(255,255,255,0.1); border-radius: 8px; color: white; box-sizing: border-box; }
        button { width: 100%; padding: 12px; margin-top: 16px; background: #00e5ff; border: none; border-radius: 8px; font-weight: bold; cursor: pointer; color: #05070f; }
        button:hover { background: #00b3cc; }
        .error { color: #ff5252; font-size: 13px; margin-top: 10px; }
    </style>
</head>
<body>
    <div class="card">
        <h1>⚡ VoltEdge Enterprise</h1>
        <form method="POST">
            <input type="text" name="username" placeholder="Username (admin / staff)" required>
            <input type="password" name="password" placeholder="Password (12345678)" required>
            <button type="submit">Log In</button>
        </form>
        {% if error %}<div class="error">{{ error }}</div>{% endif %}
    </div>
</body>
</html>
"""

ADMIN_TEMPLATE = """
<!DOCTYPE html>
<html>
<head>
    <title>VoltEdge - Command Center</title>
    <style>
        body { background: #05070f; color: white; font-family: 'Segoe UI', sans-serif; margin: 0; padding: 20px; }
        .header { display: flex; justify-content: space-between; align-items: center; background: rgba(13, 21, 39, 0.8); padding: 16px 24px; border-radius: 12px; border: 1px solid rgba(0, 229, 255, 0.2); margin-bottom: 20px; }
        .grid { display: grid; grid-template-columns: 2fr 1fr; gap: 20px; }
        .card { background: rgba(13, 21, 39, 0.8); border: 1px solid rgba(0, 229, 255, 0.2); border-radius: 12px; padding: 20px; margin-bottom: 20px; }
        .video-box img { width: 100%; border-radius: 8px; border: 1px solid rgba(0, 229, 255, 0.3); }
        .stat-val { font-size: 28px; font-weight: bold; color: #00e5ff; margin-top: 4px; }
        .btn { padding: 6px 12px; border: none; border-radius: 4px; cursor: pointer; font-size: 12px; font-weight: bold; }
        .btn-auto { background: #2196F3; color: white; }
        .btn-on { background: #4CAF50; color: white; }
        .btn-off { background: #F44336; color: white; }
        ul { list-style: none; padding: 0; margin: 0; }
        li { padding: 8px 0; border-bottom: 1px solid rgba(255,255,255,0.05); color: #ffeb3b; }
    </style>
</head>
<body>
    <div class="header">
        <h2 style="margin:0; color:#00e5ff;">⚡ Enterprise Command Center</h2>
        <a href="/logout" style="color:#ff5252; text-decoration:none; font-weight:bold;">Logout</a>
    </div>

    <div class="grid">
        <div>
            <div class="card video-box">
                <h3 style="margin-top:0; color:#00e5ff;">📹 Real-Time Edge Analytics</h3>
                <img src="/video_feed">
            </div>
        </div>

        <div>
            <div class="card">
                <h3 style="margin-top:0; color:#00e5ff;">💰 Energy Savings Engine</h3>
                <div style="display:flex; justify-content:space-between;">
                    <div>
                        <div style="font-size:12px; color:#94a3b8;">KWH SAVED</div>
                        <div id="kwh" class="stat-val">0.00</div>
                    </div>
                    <div>
                        <div style="font-size:12px; color:#94a3b8;">MONEY SAVED</div>
                        <div id="rs" class="stat-val" style="color:#4caf50;">₹0.00</div>
                    </div>
                </div>
            </div>

            <div class="card">
                <h3 style="margin-top:0; color:#00e5ff;">🎛️ Dynamic Overrides</h3>
                {% for z in ["zone1", "zone2", "zone3", "zone4"] %}
                <div style="margin-bottom:10px; display:flex; justify-content:space-between; align-items:center;">
                    <span>{{ z.upper() }}</span>
                    <div>
                        <button class="btn btn-auto" onclick="setOverride('{{z}}', 'AUTO')">AUTO</button>
                        <button class="btn btn-on" onclick="setOverride('{{z}}', 'FORCE_ON')">ON</button>
                        <button class="btn btn-off" onclick="setOverride('{{z}}', 'FORCE_OFF')">OFF</button>
                    </div>
                </div>
                {% endfor %}
            </div>

            <div class="card">
                <h3 style="margin-top:0; color:#00e5ff;">🔔 Active Service Alerts</h3>
                <ul id="alerts"><li>No active alerts</li></ul>
            </div>
        </div>
    </div>

    <script>
        async function setOverride(zone, mode) {
            await fetch('/api/override', { method: 'POST', headers: { 'Content-Type': 'application/json' }, body: JSON.stringify({ zone, mode }) });
        }

        setInterval(async () => {
            let res = await fetch('/api/stats');
            let data = await res.json();
            document.getElementById('kwh').innerText = data.kwh_saved;
            document.getElementById('rs').innerText = '₹' + data.rupees_saved;
            if (data.alerts.length > 0) {
                document.getElementById('alerts').innerHTML = data.alerts.map(a => `<li>📍 ${a.type} at ${a.time}</li>`).join('');
            }
        }, 2000);
    </script>
</body>
</html>
"""

STAFF_TEMPLATE = """
<!DOCTYPE html>
<html>
<head>
    <title>VoltEdge - Staff Terminal</title>
    <style>
        body { background: #05070f; color: white; font-family: 'Segoe UI', sans-serif; padding: 20px; text-align: center; }
        .card { background: rgba(13, 21, 39, 0.8); border: 1px solid rgba(0, 229, 255, 0.2); border-radius: 16px; padding: 30px; display: inline-block; width: 60%; }
        h1 { color: #00e5ff; margin-top: 0; }
        ul { list-style: none; padding: 0; text-align: left; }
        li { background: rgba(255,255,255,0.03); margin-bottom: 10px; padding: 16px; border-radius: 8px; border-left: 4px solid #00e5ff; font-size: 18px; }
    </style>
</head>
<body>
    <div style="text-align:right; margin-bottom:10px;"><a href="/logout" style="color:#ff5252; text-decoration:none;">Logout</a></div>
    <div class="card">
        <h1>🛎️ Incoming Attendant Requests</h1>
        <ul id="alert-list"><li>Waiting for requests...</li></ul>
    </div>
    <script>
        setInterval(async () => {
            let res = await fetch('/api/stats');
            let data = await res.json();
            if (data.alerts.length > 0) {
                document.getElementById('alert-list').innerHTML = data.alerts.reverse().map(a => `<li><b>${a.type}</b> <span style="float:right; color:#94a3b8; font-size:14px;">${a.time}</span></li>`).join('');
            }
        }, 2000);
    </script>
</body>
</html>
"""

# --- 7. ROUTING & CONTROLLERS ---
@app.route('/login', methods=['GET', 'POST'])
def login():
    error = None
    if request.method == 'POST':
        username = request.form.get('username')
        password = request.form.get('password')
        user = USERS.get(username)

        if user and user.password == password:
            login_user(user)
            if user.role == 'staff':
                return redirect(url_for('staff_dash'))
            return redirect(url_for('admin_dash'))
        else:
            error = "Invalid Username or Password"
    return render_template_string(LOGIN_TEMPLATE, error=error)

@app.route('/logout')
@login_required
def logout():
    logout_user()
    return redirect(url_for('login'))

@app.route('/')
@app.route('/admin')
@login_required
def admin_dash():
    if current_user.role not in ['admin', 'occupant']:
        return redirect(url_for('staff_dash'))
    return render_template_string(ADMIN_TEMPLATE)

@app.route('/staff')
@login_required
def staff_dash():
    if current_user.role != 'staff':
        return redirect(url_for('admin_dash'))
    return render_template_string(STAFF_TEMPLATE)

@app.route('/video_feed')
@login_required
def video_feed():
    return Response(gen_frames(), mimetype='multipart/x-mixed-replace; boundary=frame')

@app.route('/api/override', methods=['POST'])
@login_required
def set_override():
    data = request.json
    zone, mode = data.get("zone"), data.get("mode")
    if zone in manual_overrides and mode in ["AUTO", "FORCE_ON", "FORCE_OFF"]:
        manual_overrides[zone] = mode
        return jsonify({"status": "success"})
    return jsonify({"status": "error"}), 400

@app.route('/api/stats')
@login_required
def get_stats():
    kwh_saved = (total_idle_seconds / 3600.0) * POWER_RATING_KW_PER_ZONE
    rupees_saved = kwh_saved * TARIFF_PER_KWH
    return jsonify({
        "kwh_saved": round(kwh_saved, 4),
        "rupees_saved": round(rupees_saved, 2),
        "alerts": service_alerts[-5:]
    })

if __name__ == '__main__':
    app.run(host='0.0.0.0', port=5000, debug=False, threaded=True)
