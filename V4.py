from flask import Flask, render_template_string, Response, jsonify, request, redirect, url_for
from flask_login import LoginManager, UserMixin, login_user, login_required, logout_user
import cv2
import numpy as np
import time
import requests
import threading
import collections
from datetime import datetime
from concurrent.futures import ThreadPoolExecutor
from ultralytics import YOLO
import mediapipe as mp

app = Flask(__name__)
app.secret_key = 'voltedge_vnit_key'

# --- 1. MULTI-ROLE AUTHENTICATION ---
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
    'admin': User(1, 'admin', 'admin', 'admin'),
    'judge': User(2, 'vnit', 'vnit2026', 'judge')
}

@login_manager.user_loader
def load_user(user_id):
    for u in USERS.values():
        if str(u.id) == str(user_id):
            return u
    return None

# --- 2. CONFIGURATION & QUADRANT ZONE DIVISION ---
ESP_IP = "http://192.168.4.1" 
executor = ThreadPoolExecutor(max_workers=4)

FRAME_W, FRAME_H = 640, 480
MID_X, MID_Y = FRAME_W // 2, FRAME_H // 2

# Math-based 4-Quadrant boxes: Top-Left, Top-Right, Bottom-Left, Bottom-Right
POLYGON_ZONES = {
    "zone1": np.array([[0, 0], [MID_X, 0], [MID_X, MID_Y], [0, MID_Y]], np.int32),         # Top-Left
    "zone2": np.array([[MID_X, 0], [FRAME_W, 0], [FRAME_W, MID_Y], [MID_X, MID_Y]], np.int32), # Top-Right
    "zone3": np.array([[0, MID_Y], [MID_X, MID_Y], [MID_X, FRAME_H], [0, FRAME_H]], np.int32), # Bottom-Left
    "zone4": np.array([[MID_X, MID_Y], [FRAME_W, MID_Y], [FRAME_W, FRAME_H], [MID_X, FRAME_H]], np.int32) # Bottom-Right
}

relay_states = {f"zone{i}": {"light": False, "fan": False} for i in range(1, 5)}
manual_overrides = {f"zone{i}": "AUTO" for i in range(1, 5)}
last_seen_times = {f"zone{i}": 0 for i in range(1, 5)}
hourly_activity = collections.defaultdict(lambda: {f"zone{i}": 0 for i in range(1, 5)})

service_alerts = []
total_idle_seconds = 0
POWER_RATING_KW = 0.08  
TARIFF_PER_KWH = 10.00          

# --- 3. AI MODELS ---
yolo_model = YOLO("yolov8n.pt")
mp_hands = mp.solutions.hands
hands = mp_hands.Hands(max_num_hands=1, min_detection_confidence=0.7)

# --- 4. DECOUPLED CAMERA STREAM ---
class CameraStream:
    def __init__(self, src=0):
        self.cap = cv2.VideoCapture(src)
        self.cap.set(cv2.CAP_PROP_FRAME_WIDTH, FRAME_W)
        self.cap.set(cv2.CAP_PROP_FRAME_HEIGHT, FRAME_H)
        self.ret, self.frame = self.cap.read()
        self.running = True
        self.thread = threading.Thread(target=self.update, args=())
        self.thread.daemon = True
        self.thread.start()

    def update(self):
        while self.running:
            self.ret, self.frame = self.cap.read()
            time.sleep(0.01)

    def get_frame(self):
        return self.frame.copy() if self.ret else None

cam = CameraStream()

# --- 5. VISION & HARDWARE PIPELINE ---
def async_send(url):
    try: requests.get(url, timeout=0.3)
    except: pass

def update_relay(zone, appliance, turn_on):
    if zone in relay_states and relay_states[zone][appliance] != turn_on:
        relay_states[zone][appliance] = turn_on
        state_str = "on" if turn_on else "off"
        executor.submit(async_send, f"{ESP_IP}/{zone}/{appliance}/{state_str}")

last_alert_time = 0

def gen_frames():
    global total_idle_seconds, last_alert_time

    while True:
        frame = cam.get_frame()
        if frame is None: continue

        current_time = time.time()
        current_hour = datetime.now().strftime("%H:00")

        # A. GESTURE DETECTION
        rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        gesture_results = hands.process(rgb_frame)

        if gesture_results.multi_hand_landmarks and (current_time - last_alert_time > 4):
            for hand_landmarks in gesture_results.multi_hand_landmarks:
                lm = hand_landmarks.landmark
                fingers_up = sum(1 for tip, mcp in [(8,5), (12,9), (16,13), (20,17)] if lm[tip].y < lm[mcp].y)
                
                alert_type = None
                if fingers_up == 5: alert_type = "EMERGENCY ASSIST ✋"
                elif fingers_up == 2: alert_type = "SERVICE REQUEST ✌️"

                if alert_type:
                    service_alerts.append({"type": alert_type, "time": datetime.now().strftime("%H:%M:%S")})
                    last_alert_time = current_time

        # B. YOLOv8 BYTETRACKING WITH QUADRANT LOGIC
        results = yolo_model.track(frame, persist=True, tracker="bytetrack.yaml", verbose=False, imgsz=640, conf=0.45)
        
        if results[0].boxes.id is not None:
            boxes = results[0].boxes.xyxy.int().cpu().tolist()
            class_ids = results[0].boxes.cls.int().cpu().tolist()
            track_ids = results[0].boxes.id.int().cpu().tolist()

            for box, cls_id, track_id in zip(boxes, class_ids, track_ids):
                if cls_id == 0:  # Person
                    x1, y1, x2, y2 = box
                    cx = (x1 + x2) // 2
                    cy = (y1 + y2) // 2

                    # Determine quadrant mathematically
                    if cx < MID_X and cy < MID_Y: active_zone_name = "zone1"
                    elif cx >= MID_X and cy < MID_Y: active_zone_name = "zone2"
                    elif cx < MID_X and cy >= MID_Y: active_zone_name = "zone3"
                    else: active_zone_name = "zone4"

                    last_seen_times[active_zone_name] = current_time

                    cv2.rectangle(frame, (x1, y1), (x2, y2), (255, 255, 255), 2)
                    cv2.circle(frame, (cx, cy), 4, (0, 0, 255), -1)
                    cv2.putText(frame, f"ID: {track_id} ({active_zone_name.upper()})", 
                                (x1, y1 - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 2)

        # C. RELAY CONTROL & BLACK QUADRANT OVERLAYS
        for z_name, poly in POLYGON_ZONES.items():
            mode = manual_overrides.get(z_name, "AUTO")
            is_active = (current_time - last_seen_times.get(z_name, 0)) <= 5.0

            if mode == "FORCE_ON":
                update_relay(z_name, "light", True)
            elif mode == "FORCE_OFF":
                update_relay(z_name, "light", False)
                total_idle_seconds += (1 / 30.0)
            else:
                update_relay(z_name, "light", is_active)
                if is_active: hourly_activity[current_hour][z_name] += (1 / 30.0)
                else: total_idle_seconds += (1 / 30.0)

            # Draw zone boundaries in pure black, slightly thicker to see them on video
            cv2.polylines(frame, [poly], True, (0, 0, 0), 3)
            
            # Label positioning (offset slightly from the top-left of each quadrant)
            x_start = poly[0][0] + 10
            y_start = poly[0][1] + 30
            # If it's active, label is white, otherwise black
            text_color = (255, 255, 255) if is_active else (0, 0, 0)
            cv2.putText(frame, z_name.upper(), (x_start, y_start), 
                        cv2.FONT_HERSHEY_SIMPLEX, 0.6, text_color, 2)

        ret, buffer = cv2.imencode('.jpg', frame)
        yield (b'--frame\r\n' b'Content-Type: image/jpeg\r\n\r\n' + buffer.tobytes() + b'\r\n')

# --- 6. DASHBOARD TEMPLATES (Pure Black Backgrounds) ---
LOGIN_TEMPLATE = """
<body style="background:#000000; color:white; font-family:sans-serif; text-align:center; padding-top:100px;">
    <h1 style="color:#00e5ff;">⚡ VoltEdge Command</h1>
    <form method="POST" style="background:#111111; display:inline-block; padding:40px; border-radius:16px;">
        <input type="text" name="username" placeholder="Username" style="padding:10px; width:250px; margin-bottom:10px;"><br>
        <input type="password" name="password" placeholder="Password" style="padding:10px; width:250px;"><br><br>
        <button type="submit" style="padding:10px 20px; background:#00e5ff; border:none; font-weight:bold; color:black;">Log In</button>
    </form>
</body>
"""

ADMIN_TEMPLATE = """
<!DOCTYPE html>
<html>
<head>
    <title>VoltEdge - Command Dashboard</title>
    <script src="https://cdn.jsdelivr.net/npm/chart.js"></script>
    <style>
        body { background: #000000; color: white; font-family: 'Segoe UI', sans-serif; padding: 20px; margin: 0; }
        .grid { display: grid; grid-template-columns: 2fr 1fr; gap: 20px; }
        .card { background: #111111; border: 1px solid #333333; border-radius: 12px; padding: 20px; }
        .btn { padding: 6px 12px; border: none; border-radius: 4px; cursor: pointer; font-weight: bold; margin: 2px; }
        .btn-auto { background: #2196F3; color: white; }
        .btn-on { background: #4CAF50; color: white; }
        .btn-off { background: #F44336; color: white; }
        .btn-alarm { background: #ff1744; color: white; width: 100%; padding: 15px; font-size: 18px; text-transform: uppercase; }
        #video-feed { width: 640px; height: 480px; border: 2px solid #333333; border-radius: 8px; }
    </style>
</head>
<body>
    <div style="display:flex; justify-content:space-between; align-items:center; margin-bottom:20px;">
        <h2 style="color:#00e5ff; margin:0;">⚡ VoltEdge Live Control Panel</h2>
        <a href="/logout" style="color:#ff5252; text-decoration:none;">Logout</a>
    </div>

    <div class="grid">
        <div>
            <div class="card">
                <h3>📹 4-Quadrant Stream (Top/Bottom, Left/Right)</h3>
                <img id="video-feed" src="/video_feed">
            </div>
            
            <div class="card" style="margin-top:20px;">
                <h3>📊 Dwell Time Analytics (Mins / Hour)</h3>
                <canvas id="activityChart" height="100"></canvas>
            </div>
        </div>

        <div>
            <div class="card">
                <h3>💰 Energy Savings Engine</h3>
                <p>KWH SAVED: <b id="kwh" style="color:#00e5ff; font-size:24px;">0.00</b></p>
                <p>MONEY SAVED: <b id="rs" style="color:#4caf50; font-size:24px;">₹0.00</b></p>
            </div>

            <div class="card" style="margin-top:20px;">
                <h3>🎛️ Zone Relay Overrides</h3>
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

            <div class="card" style="margin-top:20px;">
                <h3>🚨 Security System</h3>
                <button class="btn btn-alarm" onclick="triggerAlarm('on')">Sound Buzzer</button>
                <button class="btn btn-off" style="width:100%; padding:10px; margin-top:5px;" onclick="triggerAlarm('off')">Silence</button>
                
                <h4 style="margin-top:20px; color:#ffeb3b;">Gesture Alerts:</h4>
                <ul id="alerts" style="list-style:none; padding:0; font-size:14px; color:#00e5ff;"></ul>
            </div>
        </div>
    </div>

    <script>
        // Set chart text color to white for dark mode
        Chart.defaults.color = '#ffffff';
        let chartCtx = document.getElementById('activityChart').getContext('2d');
        let activityChart = new Chart(chartCtx, {
            type: 'bar',
            data: { labels: [], datasets: [] },
            options: { scales: { x: { stacked: true }, y: { stacked: true } } }
        });

        async function setOverride(zone, mode) {
            await fetch('/api/override', { method: 'POST', headers: {'Content-Type': 'application/json'}, body: JSON.stringify({zone, mode}) });
        }
        async function triggerAlarm(state) {
            await fetch('/api/alarm', { method: 'POST', headers: {'Content-Type': 'application/json'}, body: JSON.stringify({state}) });
        }

        setInterval(async () => {
            let res = await fetch('/api/stats');
            let data = await res.json();
            document.getElementById('kwh').innerText = data.kwh_saved;
            document.getElementById('rs').innerText = '₹' + data.rupees_saved;
            document.getElementById('alerts').innerHTML = data.alerts.map(a => `<li>${a.type} [${a.time}]</li>`).join('');

            let chartRes = await fetch('/api/activity_data');
            let chartData = await chartRes.json();
            activityChart.data.labels = chartData.labels;
            activityChart.data.datasets = chartData.datasets;
            activityChart.update();
        }, 2000);
    </script>
</body>
</html>
"""

# --- 7. FLASK CONTROLLERS ---
@app.route('/login', methods=['GET', 'POST'])
def login():
    if request.method == 'POST':
        user = USERS.get(request.form.get('username'))
        if user and user.password == request.form.get('password'):
            login_user(user)
            return redirect(url_for('admin_dash'))
    return render_template_string(LOGIN_TEMPLATE)

@app.route('/logout')
@login_required
def logout():
    logout_user()
    return redirect(url_for('login'))

@app.route('/')
@login_required
def admin_dash():
    return render_template_string(ADMIN_TEMPLATE)

@app.route('/video_feed')
@login_required
def video_feed():
    return Response(gen_frames(), mimetype='multipart/x-mixed-replace; boundary=frame')

@app.route('/api/override', methods=['POST'])
@login_required
def set_override():
    data = request.json
    manual_overrides[data['zone']] = data['mode']
    return jsonify({"status": "success"})

@app.route('/api/alarm', methods=['POST'])
@login_required
def trigger_alarm():
    state = request.json.get("state")
    executor.submit(async_send, f"{ESP_IP}/alarm/{state}")
    return jsonify({"status": "sent", "state": state})

@app.route('/api/stats')
@login_required
def get_stats():
    kwh = (total_idle_seconds / 3600.0) * POWER_RATING_KW
    return jsonify({
        "kwh_saved": round(kwh, 4),
        "rupees_saved": round(kwh * TARIFF_PER_KWH, 2),
        "alerts": service_alerts[-5:]
    })

@app.route('/api/activity_data')
@login_required
def get_activity_data():
    labels = list(hourly_activity.keys())
    if not labels:
        labels = [datetime.now().strftime("%H:00")]
    colors = ['#FF5722', '#2196F3', '#4CAF50', '#FFEB3B']
    datasets = [{"label": z.upper(), "data": [round(hourly_activity[h].get(z, 0)/60.0, 1) for h in labels], "backgroundColor": colors[i]} for i, z in enumerate(POLYGON_ZONES.keys())]
    return jsonify({"labels": labels, "datasets": datasets})

if __name__ == '__main__':
    app.run(host='0.0.0.0', port=5000, debug=False, threaded=True)
