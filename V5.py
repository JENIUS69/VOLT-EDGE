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

# --- 1. AUTHENTICATION ---
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

# --- 2. CONFIGURATION & SENSORS ---
ESP_IP = "http://192.168.4.1"
executor = ThreadPoolExecutor(max_workers=8)

FRAME_W, FRAME_H = 640, 480
COLS, ROWS = 4, 2
COL_W = FRAME_W // COLS
ROW_H = FRAME_H // ROWS

POLYGON_ZONES = {}
for r in range(ROWS):
    for c in range(COLS):
        z_idx = r * COLS + c + 1
        x1, y1 = c * COL_W, r * ROW_H
        x2, y2 = (c + 1) * COL_W, (r + 1) * ROW_H
        POLYGON_ZONES[f"zone{z_idx}"] = np.array([[x1, y1], [x2, y1], [x2, y2], [x1, y2]], np.int32)

relay_states = {f"zone{i}": {"light": False} for i in range(1, 9)}
manual_overrides = {f"zone{i}": "AUTO" for i in range(1, 9)}
last_seen_times = {f"zone{i}": 0 for i in range(1, 9)}
hourly_activity = collections.defaultdict(lambda: {f"zone{i}": 0 for i in range(1, 9)})

sensor_data = {"temp": "--", "humidity": "--"}
service_alerts = []
total_idle_seconds = 0
POWER_RATING_KW = 0.08  
TARIFF_PER_KWH = 10.00          

# --- 3. BACKGROUND SENSOR POLLING ---
def poll_environment_sensor():
    global sensor_data
    while True:
        try:
            r = requests.get(f"{ESP_IP}/sensor", timeout=1.0)
            if r.status_code == 200:
                sensor_data = r.json()
        except:
            pass
        time.sleep(3)

threading.Thread(target=poll_environment_sensor, daemon=True).start()

# --- 4. AI MODELS & CAMERA ---
yolo_model = YOLO("yolov8n.pt")
mp_hands = mp.solutions.hands
hands = mp_hands.Hands(max_num_hands=1, min_detection_confidence=0.7)

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

# --- 5. VISION PIPELINE ---
def async_send(url):
    try: requests.get(url, timeout=0.3)
    except: pass

def update_relay(zone, turn_on):
    if zone in relay_states and relay_states[zone]["light"] != turn_on:
        relay_states[zone]["light"] = turn_on
        state_str = "on" if turn_on else "off"
        executor.submit(async_send, f"{ESP_IP}/{zone}/{state_str}")

last_alert_time = 0

def gen_frames():
    global total_idle_seconds, last_alert_time

    while True:
        frame = cam.get_frame()
        if frame is None: continue

        current_time = time.time()
        current_hour = datetime.now().strftime("%H:00")

        # Hand Gesture Recognition
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

        # YOLO Detection & Grid Mapping
        results = yolo_model.track(frame, persist=True, tracker="bytetrack.yaml", verbose=False, imgsz=640, conf=0.45)
        
        if results[0].boxes.id is not None:
            boxes = results[0].boxes.xyxy.int().cpu().tolist()
            class_ids = results[0].boxes.cls.int().cpu().tolist()
            track_ids = results[0].boxes.id.int().cpu().tolist()

            for box, cls_id, track_id in zip(boxes, class_ids, track_ids):
                if cls_id == 0:  
                    x1, y1, x2, y2 = box
                    cx = (x1 + x2) // 2
                    cy = (y1 + y2) // 2

                    grid_c = min(max(cx // COL_W, 0), COLS - 1)
                    grid_r = min(max(cy // ROW_H, 0), ROWS - 1)
                    z_idx = grid_r * COLS + grid_c + 1
                    active_zone_name = f"zone{z_idx}"

                    last_seen_times[active_zone_name] = current_time

                    cv2.rectangle(frame, (x1, y1), (x2, y2), (255, 255, 255), 2)
                    cv2.circle(frame, (cx, cy), 4, (0, 0, 255), -1)
                    cv2.putText(frame, f"ID: {track_id} ({active_zone_name.upper()})", 
                                (x1, y1 - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 229, 255), 2)

        # Relay Automation Logic
        for z_name, poly in POLYGON_ZONES.items():
            mode = manual_overrides.get(z_name, "AUTO")
            is_active = (current_time - last_seen_times.get(z_name, 0)) <= 5.0

            if mode == "FORCE_ON":
                update_relay(z_name, True)
            elif mode == "FORCE_OFF":
                update_relay(z_name, False)
                total_idle_seconds += (1 / 30.0)
            else:
                update_relay(z_name, is_active)
                if is_active: hourly_activity[current_hour][z_name] += (1 / 30.0)
                else: total_idle_seconds += (1 / 30.0)

            cv2.polylines(frame, [poly], True, (0, 0, 0), 2)
            x_start, y_start = poly[0][0] + 10, poly[0][1] + 25
            text_color = (0, 255, 0) if is_active else (100, 100, 100)
            cv2.putText(frame, z_name.upper(), (x_start, y_start), cv2.FONT_HERSHEY_DUPLEX, 0.5, text_color, 1)

        ret, buffer = cv2.imencode('.jpg', frame)
        yield (b'--frame\r\n' b'Content-Type: image/jpeg\r\n\r\n' + buffer.tobytes() + b'\r\n')

# --- 6. UI TEMPLATES ---
LOGIN_TEMPLATE = """
<body style="background:#000000; color:white; font-family:'Segoe UI', sans-serif; display:flex; justify-content:center; align-items:center; height:100vh; margin:0;">
    <div style="background:#0a0a0a; padding:50px; border-radius:16px; border: 1px solid #1a1a1a; box-shadow: 0 0 30px rgba(0, 229, 255, 0.15); text-align:center;">
        <h1 style="color:#00e5ff; margin-bottom: 30px; letter-spacing: 2px;">⚡ VoltEdge OS</h1>
        <form method="POST" style="display:flex; flex-direction:column; gap:15px;">
            <input type="text" name="username" placeholder="Admin ID" style="padding:15px; width:280px; background:#111; border:1px solid #333; color:white; border-radius:8px; outline:none; font-size:16px;">
            <input type="password" name="password" placeholder="Passcode" style="padding:15px; width:280px; background:#111; border:1px solid #333; color:white; border-radius:8px; outline:none; font-size:16px;">
            <button type="submit" style="padding:15px; margin-top:15px; background:linear-gradient(90deg, #00e5ff, #0077ff); border:none; border-radius:8px; font-weight:bold; color:black; cursor:pointer; font-size:16px; text-transform:uppercase; letter-spacing:1px;">Access Dashboard</button>
        </form>
    </div>
</body>
"""

ADMIN_TEMPLATE = """
<!DOCTYPE html>
<html>
<head>
    <title>VoltEdge Command Center</title>
    <script src="https://cdn.jsdelivr.net/npm/chart.js"></script>
    <style>
        :root { --accent: #00e5ff; --bg: #000; --panel: #0a0a0a; --text: #e0e0e0; }
        body { background: var(--bg); color: var(--text); font-family: 'Segoe UI', sans-serif; padding: 25px; margin: 0; }
        
        .header { display: flex; justify-content: space-between; align-items: center; margin-bottom: 25px; padding-bottom: 15px; border-bottom: 1px solid #222; }
        .header h2 { color: var(--accent); margin: 0; letter-spacing: 2px; font-weight: 700; text-transform: uppercase; }
        .logout-btn { color: #ff4757; text-decoration: none; background: #111; padding: 10px 20px; border-radius: 8px; border: 1px solid #333; font-weight: bold; }
        
        .grid { display: grid; grid-template-columns: 1.3fr 1fr; gap: 25px; }
        .card { background: var(--panel); border: 1px solid #1a1a1a; border-radius: 16px; padding: 25px; box-shadow: 0 10px 30px rgba(0,0,0,0.5); }
        .card h3 { margin-top: 0; color: #fff; font-size: 15px; border-left: 4px solid var(--accent); padding-left: 12px; text-transform: uppercase; letter-spacing: 1px; }
        
        #video-feed { width: 100%; border-radius: 12px; border: 2px solid #222; }
        
        .stat-grid { display: grid; grid-template-columns: 1fr 1fr; gap: 15px; margin-bottom: 25px; }
        .stat-box { background: #111; padding: 15px; border-radius: 12px; border: 1px solid #222; text-align: center; }
        .stat-title { font-size: 11px; color: #888; text-transform: uppercase; letter-spacing: 1px; display: block; margin-bottom: 8px; }
        .stat-val { font-size: 24px; font-weight: bold; }
        
        .zones-grid { display: grid; grid-template-columns: 1fr 1fr; gap: 12px; }
        .zone-card { background: #111; padding: 12px; border-radius: 12px; border: 1px solid #222; display: flex; flex-direction: column; gap: 8px; }
        .zone-header { font-weight: bold; color: #fff; font-size: 13px; text-align: center; }
        .btn-group { display: flex; gap: 5px; }
        
        .btn { padding: 6px; border: none; border-radius: 6px; cursor: pointer; font-weight: bold; font-size: 10px; flex: 1; color: #fff; }
        .btn-auto { background: #2980b9; }
        .btn-on { background: #27ae60; }
        .btn-off { background: #c0392b; }
        
        .btn-alarm { background: linear-gradient(45deg, #ff4757, #ff6b81); color: white; width: 100%; padding: 14px; font-size: 15px; text-transform: uppercase; border-radius: 12px; font-weight: bold; cursor: pointer; border: none; }
        .btn-silence { background: #222; color: #fff; width: 100%; padding: 10px; margin-top: 10px; border: 1px solid #444; border-radius: 12px; cursor: pointer; font-weight: bold; }
        
        .alert-item { background: rgba(0, 229, 255, 0.05); border: 1px solid rgba(0, 229, 255, 0.2); padding: 10px; margin-bottom: 8px; border-radius: 8px; color: var(--accent); font-size: 12px; display: flex; justify-content: space-between; }
    </style>
</head>
<body>
    <div class="header">
        <h2>⚡ VoltEdge OS Command</h2>
        <a href="/logout" class="logout-btn">DISCONNECT</a>
    </div>

    <div class="grid">
        <div style="display: flex; flex-direction: column; gap: 25px;">
            <div class="card">
                <h3>Optical Sensor Array (8-Zone Grid)</h3>
                <img id="video-feed" src="/video_feed">
            </div>
            
            <div class="card">
                <h3>Zone Dwell Kinetics</h3>
                <canvas id="activityChart" height="100"></canvas>
            </div>
        </div>

        <div style="display: flex; flex-direction: column; gap: 25px;">
            <div class="stat-grid">
                <div class="stat-box">
                    <span class="stat-title">Energy Conserved</span>
                    <span class="stat-val" style="color:#00e5ff;" id="kwh">0.00 <span style="font-size:14px">kWh</span></span>
                </div>
                <div class="stat-box">
                    <span class="stat-title">Capital Return</span>
                    <span class="stat-val" style="color:#2ecc71;" id="rs">₹0.00</span>
                </div>
                <div class="stat-box">
                    <span class="stat-title">Temperature</span>
                    <span class="stat-val" style="color:#ff9f43;" id="temp">-- °C</span>
                </div>
                <div class="stat-box">
                    <span class="stat-title">Humidity</span>
                    <span class="stat-val" style="color:#0abde3;" id="hum">-- %</span>
                </div>
            </div>

            <div class="card">
                <h3>Hardware Override Matrix</h3>
                <div class="zones-grid">
                    {% for z in ["zone1", "zone2", "zone3", "zone4", "zone5", "zone6", "zone7", "zone8"] %}
                    <div class="zone-card">
                        <span class="zone-header">{{ z.upper() }}</span>
                        <div class="btn-group">
                            <button class="btn btn-auto" onclick="setOverride('{{z}}', 'AUTO')">AUTO</button>
                            <button class="btn btn-on" onclick="setOverride('{{z}}', 'FORCE_ON')">ON</button>
                            <button class="btn btn-off" onclick="setOverride('{{z}}', 'FORCE_OFF')">OFF</button>
                        </div>
                    </div>
                    {% endfor %}
                </div>
            </div>

            <div class="card">
                <h3>Emergency & Security</h3>
                <button class="btn-alarm" onclick="triggerAlarm('on')">🚨 Engage Siren</button>
                <button class="btn-silence" onclick="triggerAlarm('off')">Silence Protocol</button>
                
                <h4 style="margin-top:20px; color:#666; font-size: 11px; text-transform: uppercase;">Live Gesture Logs</h4>
                <div id="alerts"></div>
            </div>
        </div>
    </div>

    <script>
        Chart.defaults.color = '#888';
        Chart.defaults.borderColor = '#222';
        let chartCtx = document.getElementById('activityChart').getContext('2d');
        let activityChart = new Chart(chartCtx, {
            type: 'bar',
            data: { labels: [], datasets: [] },
            options: { scales: { x: { stacked: true }, y: { stacked: true } }, plugins: { legend: { position: 'bottom', labels:{boxWidth: 10, font:{size: 10}} } } }
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
            document.getElementById('kwh').innerHTML = data.kwh_saved + ' <span style="font-size:14px">kWh</span>';
            document.getElementById('rs').innerText = '₹' + data.rupees_saved;
            document.getElementById('temp').innerText = data.temp + ' °C';
            document.getElementById('hum').innerText = data.humidity + ' %';
            
            let alertsHtml = data.alerts.map(a => `<div class="alert-item"><span>${a.type}</span> <span>${a.time}</span></div>`).join('');
            document.getElementById('alerts').innerHTML = alertsHtml || '<div style="color:#444; font-size:12px; text-align:center; padding: 10px 0;">No active threats detected.</div>';

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

# --- 7. CONTROLLERS ---
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
        "temp": sensor_data.get("temp", "--"),
        "humidity": sensor_data.get("humidity", "--"),
        "alerts": service_alerts[-4:]
    })

@app.route('/api/activity_data')
@login_required
def get_activity_data():
    labels = list(hourly_activity.keys())
    if not labels:
        labels = [datetime.now().strftime("%H:00")]
    
    colors = ['#00e5ff', '#2980b9', '#27ae60', '#f1c40f', '#e67e22', '#e74c3c', '#9b59b6', '#34495e']
    datasets = []
    for i, z in enumerate(POLYGON_ZONES.keys()):
        data = [round(hourly_activity[h].get(z, 0)/60.0, 1) for h in labels]
        datasets.append({"label": z.upper(), "data": data, "backgroundColor": colors[i]})
        
    return jsonify({"labels": labels, "datasets": datasets})

if __name__ == '__main__':
    app.run(host='0.0.0.0', port=5000, debug=False, threaded=True)
