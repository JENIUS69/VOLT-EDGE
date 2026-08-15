from flask import Flask, render_template_string, Response, jsonify, request, redirect, url_for, send_file
from flask_login import LoginManager, UserMixin, login_user, login_required, logout_user, current_user
import cv2
import numpy as np
import time
import requests
import threading
import json
import os
import collections
from datetime import datetime
from concurrent.futures import ThreadPoolExecutor
from ultralytics import YOLO
import mediapipe as mp
import io

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

# --- 2. CONFIGURATION & DYNAMIC ZONES ---
ESP_IP = "http://192.168.4.1"
executor = ThreadPoolExecutor(max_workers=4)
ZONE_FILE = 'zones.json'

DEFAULT_ZONES = {
    "zone1": [[20, 20], [300, 20], [260, 220], [20, 230]],
    "zone2": [[310, 20], [620, 20], [620, 230], [270, 220]],
    "zone3": [[20, 240], [310, 240], [310, 460], [20, 460]],
    "zone4": [[320, 240], [620, 240], [620, 460], [320, 460]]
}

def load_zones():
    if os.path.exists(ZONE_FILE):
        with open(ZONE_FILE, 'r') as f:
            data = json.load(f)
            return {k: np.array(v, np.int32) for k, v in data.items()}
    return {k: np.array(v, np.int32) for k, v in DEFAULT_ZONES.items()}

def save_zones_to_file():
    serializable = {k: v.tolist() for k, v in POLYGON_ZONES.items()}
    with open(ZONE_FILE, 'w') as f:
        json.dump(serializable, f)

POLYGON_ZONES = load_zones()

relay_states = {f"zone{i}": {"light": False, "fan": False} for i in range(1, 5)}
manual_overrides = {f"zone{i}": "AUTO" for i in range(1, 5)}
last_seen_times = {f"zone{i}": 0 for i in range(1, 5)}
hourly_activity = collections.defaultdict(lambda: {f"zone{i}": 0 for i in range(1, 5)})

service_alerts = []
total_idle_seconds = 0
POWER_RATING_KW_PER_ZONE = 0.08  
TARIFF_PER_KWH = 10.00          

# --- 3. AI MODELS ---
yolo_model = YOLO("yolov8n.pt")
mp_hands = mp.solutions.hands
hands = mp_hands.Hands(max_num_hands=1, min_detection_confidence=0.7)

# --- 4. HIGH-SPEED CAMERA CORE ---
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

# --- 5. VISION PIPELINE ---
def async_send(url):
    try: requests.get(url, timeout=0.3)
    except: pass

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
        if frame is None: continue

        current_time = time.time()
        current_hour = datetime.now().strftime("%H:00")

        # 1. YOLOv8
        results = yolo_model(frame, verbose=False, imgsz=320)
        for box in results[0].boxes:
            if int(box.cls[0]) == 0:
                x1, y1, x2, y2 = map(int, box.xyxy[0])
                cx, feet_y = (x1 + x2) // 2, y2

                for z_name, poly in POLYGON_ZONES.items():
                    if cv2.pointPolygonTest(poly, (cx, feet_y), False) >= 0:
                        last_seen_times[z_name] = current_time

                cv2.rectangle(frame, (x1, y1), (x2, y2), (0, 229, 255), 2)

        # 2. Gestures
        rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        gesture_results = hands.process(rgb_frame)

        if gesture_results.multi_hand_landmarks and (current_time - last_alert_time > 4):
            for hand_landmarks in gesture_results.multi_hand_landmarks:
                lm = hand_landmarks.landmark
                fingers_up = sum(1 for tip, mcp in [(8,5), (12,9), (16,13), (20,17)] if lm[tip].y < lm[mcp].y)

                alert_type = None
                if fingers_up == 1: alert_type = "Water Request 💧"
                elif fingers_up == 3: alert_type = "Coffee Request ☕"
                elif fingers_up >= 4: alert_type = "Assistant Needed ✋"

                if alert_type:
                    service_alerts.append({"type": alert_type, "time": datetime.now().strftime("%H:%M:%S")})
                    last_alert_time = current_time

        # 3. Automation & Drawing
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
            else:
                update_relay(z_name, "light", is_active)
                update_relay(z_name, "fan", is_active)
                if is_active: hourly_activity[current_hour][z_name] += (1 / 30.0)
                else: total_idle_seconds += (1 / 30.0)

            color = (76, 175, 80) if is_active else (244, 67, 54)
            cv2.polylines(frame, [poly], True, color, 2)
            cv2.putText(frame, z_name.upper(), (poly[0][0], poly[0][1] - 5), cv2.FONT_HERSHEY_SIMPLEX, 0.5, color, 2)

        ret, buffer = cv2.imencode('.jpg', frame)
        yield (b'--frame\r\n' b'Content-Type: image/jpeg\r\n\r\n' + buffer.tobytes() + b'\r\n')

# --- 6. UI TEMPLATES ---
LOGIN_TEMPLATE = """
<body style="background:#05070f; color:white; font-family:sans-serif; text-align:center; padding-top:100px;">
    <h1 style="color:#00e5ff;">⚡ VoltEdge Login</h1>
    <form method="POST" style="background:#0d1527; display:inline-block; padding:40px; border-radius:16px;">
        <input type="text" name="username" placeholder="Username" style="padding:10px; width:250px; margin-bottom:10px;"><br>
        <input type="password" name="password" placeholder="Password" style="padding:10px; width:250px;"><br><br>
        <button type="submit" style="padding:10px 20px; background:#00e5ff; border:none; font-weight:bold;">Log In</button>
    </form>
</body>
"""

ADMIN_TEMPLATE = """
<!DOCTYPE html>
<html>
<head>
    <title>VoltEdge - Command Center</title>
    <script src="https://cdn.jsdelivr.net/npm/chart.js"></script>
    <style>
        body { background: #05070f; color: white; font-family: 'Segoe UI', sans-serif; padding: 20px; }
        .grid { display: grid; grid-template-columns: 2fr 1fr; gap: 20px; }
        .card { background: rgba(13, 21, 39, 0.8); border: 1px solid rgba(0, 229, 255, 0.2); border-radius: 12px; padding: 20px; margin-bottom: 20px; }
        .btn { padding: 6px 12px; border: none; border-radius: 4px; cursor: pointer; font-weight: bold; margin: 2px; }
        .btn-auto { background: #2196F3; color: white; }
        .btn-on { background: #4CAF50; color: white; }
        .btn-off { background: #F44336; color: white; }
        .btn-alarm { background: #ff9800; color: black; width: 100%; padding: 12px; font-size: 16px; margin-top: 10px; }
        #zoneCanvas { border: 2px solid #00e5ff; cursor: crosshair; border-radius: 8px; width: 640px; height: 480px; }
    </style>
</head>
<body>
    <div style="display:flex; justify-content:space-between; align-items:center; margin-bottom:20px;">
        <h2 style="color:#00e5ff; margin:0;">⚡ Enterprise Command Center</h2>
        <a href="/logout" style="color:#ff5252; text-decoration:none;">Logout</a>
    </div>

    <div class="grid">
        <div>
            <!-- Live Feed -->
            <div class="card">
                <h3 style="margin-top:0;">📹 Real-Time Analytics</h3>
                <img src="/video_feed" style="width:100%; border-radius:8px;">
            </div>

            <!-- Custom Zone Editor -->
            <div class="card">
                <h3 style="margin-top:0; color:#ffeb3b;">📐 Draw Custom Zones</h3>
                <p style="font-size:12px; color:#aaa;">Click on the image to draw a shape. You need at least 3 points.</p>
                <button class="btn btn-auto" onclick="refreshSnapshot()">1. Refresh Image</button>
                <button class="btn btn-off" onclick="clearCanvas()">2. Clear Points</button>
                <div style="margin-top:10px; margin-bottom:10px;">
                    <button class="btn btn-on" onclick="saveZone('zone1')">Save Zone 1</button>
                    <button class="btn btn-on" onclick="saveZone('zone2')">Save Zone 2</button>
                    <button class="btn btn-on" onclick="saveZone('zone3')">Save Zone 3</button>
                    <button class="btn btn-on" onclick="saveZone('zone4')">Save Zone 4</button>
                </div>
                <canvas id="zoneCanvas" width="640" height="480"></canvas>
            </div>
            
            <!-- Activity Chart -->
            <div class="card">
                <h3 style="margin-top:0;">📊 Dwell Time Graph (Mins/Hour)</h3>
                <canvas id="activityChart" height="100"></canvas>
            </div>
        </div>

        <div>
            <!-- Savings -->
            <div class="card">
                <h3 style="margin-top:0;">💰 Energy Engine</h3>
                <p>KWH SAVED: <b id="kwh" style="color:#00e5ff; font-size:24px;">0.00</b></p>
                <p>MONEY SAVED: <b id="rs" style="color:#4caf50; font-size:24px;">₹0.00</b></p>
            </div>

            <!-- Overrides -->
            <div class="card">
                <h3 style="margin-top:0;">🎛️ Overrides</h3>
                {% for z in ["zone1", "zone2", "zone3", "zone4"] %}
                <div style="margin-bottom:10px; display:flex; justify-content:space-between;">
                    <span>{{ z.upper() }}</span>
                    <div>
                        <button class="btn btn-auto" onclick="setOverride('{{z}}', 'AUTO')">AUTO</button>
                        <button class="btn btn-on" onclick="setOverride('{{z}}', 'FORCE_ON')">ON</button>
                        <button class="btn btn-off" onclick="setOverride('{{z}}', 'FORCE_OFF')">OFF</button>
                    </div>
                </div>
                {% endfor %}
            </div>

            <!-- Security & Alerts -->
            <div class="card">
                <h3 style="margin-top:0;">🔔 Alerts</h3>
                <ul id="alerts" style="list-style:none; padding:0; color:#ffeb3b;"></ul>
                <button class="btn btn-alarm" onclick="toggleAlarm('on')">🚨 SOUND ALARM</button>
                <button class="btn btn-off" style="width:100%; padding:12px; margin-top:5px;" onclick="toggleAlarm('off')">SILENCE</button>
            </div>
        </div>
    </div>

    <script>
        // --- ZONE EDITOR LOGIC ---
        let points = [];
        let canvas = document.getElementById('zoneCanvas');
        let ctx = canvas.getContext('2d');

        function refreshSnapshot() {
            canvas.style.backgroundImage = "url('/api/snapshot?" + new Date().getTime() + "')";
            clearCanvas();
        }
        
        canvas.onclick = (e) => {
            let rect = canvas.getBoundingClientRect();
            let x = Math.round(e.clientX - rect.left);
            let y = Math.round(e.clientY - rect.top);
            points.push([x, y]);
            
            ctx.fillStyle = '#00e5ff';
            ctx.beginPath();
            ctx.arc(x, y, 4, 0, 2*Math.PI);
            ctx.fill();
            
            if(points.length > 1) {
                ctx.strokeStyle = '#00e5ff';
                ctx.lineWidth = 2;
                ctx.beginPath();
                ctx.moveTo(points[points.length-2][0], points[points.length-2][1]);
                ctx.lineTo(x, y);
                ctx.stroke();
            }
        };

        function clearCanvas() {
            points = [];
            ctx.clearRect(0, 0, canvas.width, canvas.height);
        }

        async function saveZone(zoneName) {
            if(points.length < 3) return alert('Need at least 3 points for a polygon!');
            await fetch('/api/save_zone', {
                method: 'POST',
                headers: {'Content-Type': 'application/json'},
                body: JSON.stringify({zone: zoneName, points: points})
            });
            alert(zoneName + ' updated successfully!');
            clearCanvas();
        }

        // --- DASHBOARD LOGIC ---
        let chartCtx = document.getElementById('activityChart').getContext('2d');
        let activityChart = new Chart(chartCtx, {
            type: 'bar',
            data: { labels: [], datasets: [] },
            options: { scales: { x: { stacked: true }, y: { stacked: true } } }
        });

        async function setOverride(zone, mode) {
            await fetch('/api/override', { method: 'POST', headers: {'Content-Type': 'application/json'}, body: JSON.stringify({zone, mode}) });
        }
        async function toggleAlarm(state) {
            await fetch('/api/alarm', { method: 'POST', headers: {'Content-Type': 'application/json'}, body: JSON.stringify({state}) });
        }

        setInterval(async () => {
            let res = await fetch('/api/stats');
            let data = await res.json();
            document.getElementById('kwh').innerText = data.kwh_saved;
            document.getElementById('rs').innerText = '₹' + data.rupees_saved;
            document.getElementById('alerts').innerHTML = data.alerts.map(a => `<li style="padding:4px 0; border-bottom:1px solid #333;">${a.type} <span style="float:right; font-size:12px;">${a.time}</span></li>`).join('');

            let chartRes = await fetch('/api/activity_data');
            let chartData = await chartRes.json();
            activityChart.data.labels = chartData.labels;
            activityChart.data.datasets = chartData.datasets;
            activityChart.update();
        }, 2000);

        // Init
        refreshSnapshot();
    </script>
</body>
</html>
"""

# --- 7. ROUTES & CONTROLLERS ---
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
@app.route('/admin')
@login_required
def admin_dash():
    return render_template_string(ADMIN_TEMPLATE)

@app.route('/video_feed')
@login_required
def video_feed():
    return Response(gen_frames(), mimetype='multipart/x-mixed-replace; boundary=frame')

@app.route('/api/snapshot')
@login_required
def get_snapshot():
    frame = cam.get_frame()
    if frame is None: return "", 404
    _, buffer = cv2.imencode('.jpg', frame)
    return send_file(io.BytesIO(buffer), mimetype='image/jpeg')

@app.route('/api/save_zone', methods=['POST'])
@login_required
def save_zone():
    data = request.json
    zone = data.get('zone')
    points = data.get('points')
    if zone in POLYGON_ZONES and len(points) >= 3:
        POLYGON_ZONES[zone] = np.array(points, np.int32)
        save_zones_to_file()
        return jsonify({"status": "success"})
    return jsonify({"status": "error"}), 400

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
    kwh = (total_idle_seconds / 3600.0) * POWER_RATING_KW_PER_ZONE
    return jsonify({
        "kwh_saved": round(kwh, 4),
        "rupees_saved": round(kwh * TARIFF_PER_KWH, 2),
        "alerts": service_alerts[-5:]
    })

@app.route('/api/activity_data')
@login_required
def get_activity_data():
    labels = list(hourly_activity.keys())
    colors = ['#FF5722', '#2196F3', '#4CAF50', '#FFEB3B']
    datasets = [{"label": z.upper(), "data": [round(hourly_activity[h][z]/60.0, 1) for h in labels], "backgroundColor": colors[i]} for i, z in enumerate(POLYGON_ZONES.keys())]
    return jsonify({"labels": labels, "datasets": datasets})

if __name__ == '__main__':
    app.run(host='0.0.0.0', port=5000, debug=False, threaded=True)
