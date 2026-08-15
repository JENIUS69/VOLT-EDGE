from flask import Flask, render_template_string, Response, jsonify, request, redirect, url_for
from flask_login import LoginManager, UserMixin, login_user, login_required, logout_user, current_user
import cv2
import numpy as np
import time
import requests
import collections
from datetime import datetime
from concurrent.futures import ThreadPoolExecutor
from ultralytics import YOLO
import mediapipe as mp

app = Flask(__name__)
app.secret_key = 'voltedge_vnit_key'

# --- Flask-Login Setup ---
login_manager = LoginManager()
login_manager.init_app(app)
login_manager.login_view = 'login'

class User(UserMixin):
    def __init__(self, id, username, role):
        self.id = id
        self.username = username
        self.role = role

USERS = {
    'admin': User(1, 'admin', 'admin'),
    'occupant': User(2, 'occupant', 'occupant'),
    'attendant': User(3, 'attendant', 'attendant')
}

@login_manager.user_loader
def load_user(user_id):
    for u in USERS.values():
        if str(u.id) == str(user_id):
            return u
    return None

# --- Configuration & Hardware State ---
ESP_IP = "http://192.168.4.1"
executor = ThreadPoolExecutor(max_workers=4)

# 4-Zone Boundaries (Coordinates for 640x480 resolution)
POLYGON_ZONES = {
    "zone1": np.array([[0, 0], [160, 0], [160, 480], [0, 480]], np.int32),
    "zone2": np.array([[160, 0], [320, 0], [320, 480], [160, 480]], np.int32),
    "zone3": np.array([[320, 0], [480, 0], [480, 480], [320, 480]], np.int32),
    "zone4": np.array([[480, 0], [640, 0], [640, 480], [480, 480]], np.int32),
}

relay_states = {f"zone{i}": {"light": False, "fan": False} for i in range(1, 5)}
manual_overrides = {f"zone{i}": "AUTO" for i in range(1, 5)} 
last_seen_times = {f"zone{i}": 0 for i in range(1, 5)}

service_alerts = []
total_idle_seconds = 0
hourly_activity = collections.defaultdict(lambda: {f"zone{i}": 0 for i in range(1, 5)})
dht_data = {"temperature": "--", "humidity": "--"}

# Tariff Math: 15W Light + 65W Fan = 80W (0.08 kW) @ ₹10/kWh
POWER_RATING_KW_PER_ZONE = 0.08 
TARIFF_PER_KWH = 10.00 

# --- AI Models ---
yolo_model = YOLO("yolov8n.pt")
mp_hands = mp.solutions.hands
hands = mp_hands.Hands(max_num_hands=2, min_detection_confidence=0.7)

# --- Hardware Worker Functions ---
def async_send(url):
    try:
        requests.get(url, timeout=0.3)
    except Exception:
        pass

def update_relay(zone, appliance, turn_on):
    current = relay_states[zone][appliance]
    if current != turn_on:
        relay_states[zone][appliance] = turn_on
        state_str = "on" if turn_on else "off"
        executor.submit(async_send, f"{ESP_IP}/{zone}/{appliance}/{state_str}")

def fetch_dht_sensor():
    global dht_data
    try:
        r = requests.get(f"{ESP_IP}/environment", timeout=0.5)
        if r.status_code == 200:
            dht_data = r.json()
    except Exception:
        pass

# --- Camera & AI Loop ---
cap = cv2.VideoCapture(0)

def gen_frames():
    global total_idle_seconds
    last_dht_fetch = 0

    while True:
        success, frame = cap.read()
        if not success:
            break

        frame = cv2.resize(frame, (640, 480))
        current_time = time.time()
        
        if current_time - last_dht_fetch > 3.0:
            executor.submit(fetch_dht_sensor)
            last_dht_fetch = current_time

        active_detected_zones = set()

        # 1. YOLOv8 Person Detection
        results = yolo_model(frame, verbose=False, imgsz=320)
        for box in results[0].boxes:
            if int(box.cls[0]) == 0:  # Person Class
                x1, y1, x2, y2 = map(int, box.xyxy[0])
                cx, feet_y = (x1 + x2) // 2, y2

                for z_name, poly in POLYGON_ZONES.items():
                    if cv2.pointPolygonTest(poly, (cx, feet_y), False) >= 0:
                        active_detected_zones.add(z_name)
                        last_seen_times[z_name] = current_time

                cv2.rectangle(frame, (x1, y1), (x2, y2), (0, 229, 255), 2)
                cv2.putText(frame, "PERSON", (x1, y1 - 8), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 229, 255), 2)

        # 2. MediaPipe Gesture Tracking
        rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        gesture_results = hands.process(rgb_frame)
        if gesture_results.multi_hand_landmarks:
            for hand_landmarks in gesture_results.multi_hand_landmarks:
                index_tip = hand_landmarks.landmark[8]
                thumb_tip = hand_landmarks.landmark[4]
                dist = np.hypot(index_tip.x - thumb_tip.x, index_tip.y - thumb_tip.y)

                if dist < 0.04:  # Pinch Call
                    alert_time = datetime.now().strftime("%H:%M:%S")
                    if not service_alerts or service_alerts[-1]["time"] != alert_time:
                        service_alerts.append({"type": "Coffee Request ☕", "time": alert_time})
                    cv2.putText(frame, "GESTURE TRIGGERED: Coffee", (30, 40), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 235, 59), 2)

        # 3. Automation Engine & Savings
        current_hour = datetime.now().strftime("%H:00")
        for z_name in POLYGON_ZONES.keys():
            mode = manual_overrides[z_name]
            is_active = (current_time - last_seen_times[z_name]) <= 5.0 # 5s Hold Buffer

            if mode == "FORCE_ON":
                update_relay(z_name, "light", True)
                update_relay(z_name, "fan", True)
            elif mode == "FORCE_OFF":
                update_relay(z_name, "light", False)
                update_relay(z_name, "fan", False)
                total_idle_seconds += (1 / 30.0)
            else: # AUTO
                update_relay(z_name, "light", is_active)
                update_relay(z_name, "fan", is_active)
                if is_active:
                    hourly_activity[current_hour][z_name] += (1 / 30.0)
                else:
                    total_idle_seconds += (1 / 30.0)

            color = (76, 175, 80) if is_active else (244, 67, 54)
            cv2.polylines(frame, [POLYGON_ZONES[z_name]], True, color, 2)
            cv2.putText(frame, z_name.upper(), (POLYGON_ZONES[z_name][0][0] + 10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.6, color, 2)

        ret, buffer = cv2.imencode('.jpg', frame)
        yield (b'--frame\r\n' b'Content-Type: image/jpeg\r\n\r\n' + buffer.tobytes() + b'\r\n')

# --- EMBEDDED DASHBOARD HTML ---
HTML_TEMPLATE = """
<!DOCTYPE html>
<html lang="en">
<head>
    <meta charset="UTF-8">
    <title>VoltEdge Command Center</title>
    <script src="https://cdn.jsdelivr.net/npm/chart.js"></script>
    <style>
        :root {
            --bg-color: #05070f;
            --card-bg: rgba(13, 21, 39, 0.7);
            --accent: #00e5ff;
            --text-main: #ffffff;
            --text-dim: #94a3b8;
            --border: rgba(0, 229, 255, 0.2);
        }
        body { background-color: var(--bg-color); color: var(--text-main); font-family: 'Segoe UI', Roboto, sans-serif; margin: 0; padding: 20px; }
        .navbar { display: flex; justify-content: space-between; align-items: center; padding: 16px 24px; background: var(--card-bg); backdrop-filter: blur(12px); border: 1px solid var(--border); border-radius: 16px; margin-bottom: 20px; }
        .brand { font-size: 22px; font-weight: 700; color: var(--accent); }
        .grid-container { display: grid; grid-template-columns: 2.2fr 1fr; gap: 20px; }
        .card { background: var(--card-bg); backdrop-filter: blur(12px); border: 1px solid var(--border); border-radius: 16px; padding: 20px; margin-bottom: 20px; }
        .video-container img { width: 100%; border-radius: 12px; border: 1px solid var(--border); box-shadow: 0 8px 32px rgba(0, 0, 0, 0.5); }
        .stats-grid { display: grid; grid-template-columns: repeat(2, 1fr); gap: 12px; margin-bottom: 10px; }
        .stat-card { background: rgba(255, 255, 255, 0.03); border: 1px solid rgba(255, 255, 255, 0.05); padding: 12px; border-radius: 12px; text-align: center; }
        .stat-val { font-size: 24px; font-weight: 700; color: var(--accent); margin-top: 4px; }
        .zone-controls { display: grid; grid-template-columns: repeat(2, 1fr); gap: 10px; }
        .zone-box { background: rgba(255, 255, 255, 0.02); border: 1px solid rgba(255, 255, 255, 0.08); padding: 10px; border-radius: 8px; }
        .btn { padding: 6px 10px; border: none; border-radius: 6px; cursor: pointer; font-size: 11px; font-weight: 600; margin-right: 2px; }
        .btn-auto { background: #2196F3; color: white; }
        .btn-on { background: #4CAF50; color: white; }
        .btn-off { background: #F44336; color: white; }
        .btn-alarm { background: #ff9800; color: black; font-weight: bold; width: 100%; padding: 10px; }
        ul { list-style: none; padding: 0; margin: 0; }
        li { padding: 6px 0; border-bottom: 1px solid rgba(255,255,255,0.05); color: #ffeb3b; font-size: 13px; }
    </style>
</head>
<body>
    <div class="navbar">
        <div class="brand">⚡ VoltEdge Enterprise Command Center</div>
        <a href="/logout" style="color: #ff5252; text-decoration: none; font-weight: 600;">Logout</a>
    </div>

    <div class="grid-container">
        <div>
            <div class="card video-container">
                <h3 style="margin-top: 0; color: var(--accent);">📹 Edge Processing Feed</h3>
                <img src="/video_feed" alt="Camera Feed Loading...">
            </div>
            <div class="card">
                <h3 style="margin-top: 0; color: var(--accent);">📊 Zone Dwell Time (Mins/Hour)</h3>
                <canvas id="activityChart" height="100"></canvas>
            </div>
        </div>

        <div>
            <div class="card">
                <h3 style="margin-top: 0; color: var(--accent);">💰 Real-Time Energy Savings</h3>
                <div class="stats-grid">
                    <div class="stat-card">
                        <div style="color: var(--text-dim); font-size: 11px;">SAVED POWER</div>
                        <div id="kwhSaved" class="stat-val">0.00</div>
                        <div style="font-size: 10px; color: var(--text-dim);">kWh</div>
                    </div>
                    <div class="stat-card">
                        <div style="color: var(--text-dim); font-size: 11px;">SAVED MONEY</div>
                        <div id="rupeesSaved" class="stat-val" style="color: #4caf50;">₹0.00</div>
                        <div style="font-size: 10px; color: var(--text-dim);">@ ₹10/kWh</div>
                    </div>
                </div>
            </div>

            <div class="card">
                <h3 style="margin-top: 0; color: var(--accent);">🌡️ Room Environment</h3>
                <div class="stats-grid">
                    <div class="stat-card">
                        <div style="color: var(--text-dim); font-size: 11px;">TEMPERATURE</div>
                        <div id="tempVal" class="stat-val" style="color: #ff9800;">--°C</div>
                    </div>
                    <div class="stat-card">
                        <div style="color: var(--text-dim); font-size: 11px;">HUMIDITY</div>
                        <div id="humVal" class="stat-val" style="color: #2196f3;">--%</div>
                    </div>
                </div>
            </div>

            <div class="card">
                <h3 style="margin-top: 0; color: var(--accent);">🎛️ Zone Overrides</h3>
                <div class="zone-controls">
                    <div class="zone-box">
                        <div style="font-size: 11px; font-weight: bold; margin-bottom: 4px;">ZONE 1</div>
                        <button class="btn btn-auto" onclick="setOverride('zone1', 'AUTO')">AUTO</button>
                        <button class="btn btn-on" onclick="setOverride('zone1', 'FORCE_ON')">ON</button>
                        <button class="btn btn-off" onclick="setOverride('zone1', 'FORCE_OFF')">OFF</button>
                    </div>
                    <div class="zone-box">
                        <div style="font-size: 11px; font-weight: bold; margin-bottom: 4px;">ZONE 2</div>
                        <button class="btn btn-auto" onclick="setOverride('zone2', 'AUTO')">AUTO</button>
                        <button class="btn btn-on" onclick="setOverride('zone2', 'FORCE_ON')">ON</button>
                        <button class="btn btn-off" onclick="setOverride('zone2', 'FORCE_OFF')">OFF</button>
                    </div>
                    <div class="zone-box">
                        <div style="font-size: 11px; font-weight: bold; margin-bottom: 4px;">ZONE 3</div>
                        <button class="btn btn-auto" onclick="setOverride('zone3', 'AUTO')">AUTO</button>
                        <button class="btn btn-on" onclick="setOverride('zone3', 'FORCE_ON')">ON</button>
                        <button class="btn btn-off" onclick="setOverride('zone3', 'FORCE_OFF')">OFF</button>
                    </div>
                    <div class="zone-box">
                        <div style="font-size: 11px; font-weight: bold; margin-bottom: 4px;">ZONE 4</div>
                        <button class="btn btn-auto" onclick="setOverride('zone4', 'AUTO')">AUTO</button>
                        <button class="btn btn-on" onclick="setOverride('zone4', 'FORCE_ON')">ON</button>
                        <button class="btn btn-off" onclick="setOverride('zone4', 'FORCE_OFF')">OFF</button>
                    </div>
                </div>
            </div>

            <div class="card">
                <h3 style="margin-top: 0; color: var(--accent);">🔔 Active Requests</h3>
                <ul id="alertList"><li>No active calls</li></ul>
            </div>

            <div class="card">
                <button class="btn btn-alarm" onclick="toggleAlarm('on')">🚨 TRIGGER SECURITY ALARM</button>
                <button class="btn btn-off" style="width:100%; margin-top:6px; padding:6px;" onclick="toggleAlarm('off')">SILENCE ALARM</button>
            </div>
        </div>
    </div>

    <script>
        let ctx = document.getElementById('activityChart').getContext('2d');
        let activityChart = new Chart(ctx, {
            type: 'bar',
            data: { labels: [], datasets: [] },
            options: {
                responsive: true,
                scales: {
                    x: { stacked: true, ticks: { color: '#94a3b8' } },
                    y: { stacked: true, beginAtZero: true, ticks: { color: '#94a3b8' } }
                },
                plugins: { legend: { labels: { color: '#fff' } } }
            }
        });

        async function setOverride(zone, mode) {
            await fetch('/api/override', { method: 'POST', headers: { 'Content-Type': 'application/json' }, body: JSON.stringify({ zone, mode }) });
        }

        async function toggleAlarm(state) {
            await fetch('/api/alarm', { method: 'POST', headers: { 'Content-Type': 'application/json' }, body: JSON.stringify({ state }) });
        }

        async function updateDashboard() {
            try {
                let statsRes = await fetch('/api/stats');
                let stats = await statsRes.json();
                document.getElementById('kwhSaved').innerText = stats.kwh_saved;
                document.getElementById('rupeesSaved').innerText = '₹' + stats.rupees_saved;

                if (stats.environment.temperature) {
                    document.getElementById('tempVal').innerText = stats.environment.temperature + '°C';
                    document.getElementById('humVal').innerText = stats.environment.humidity + '%';
                }

                let alertList = document.getElementById('alertList');
                if (stats.alerts.length > 0) {
                    alertList.innerHTML = stats.alerts.map(a => `<li>📍 ${a.type} at ${a.time}</li>`).join('');
                }

                let chartRes = await fetch('/api/activity_data');
                let chartData = await chartRes.json();
                activityChart.data.labels = chartData.labels;
                activityChart.data.datasets = chartData.datasets;
                activityChart.update();
            } catch(e) { console.error(e); }
        }

        setInterval(updateDashboard, 2000);
        updateDashboard();
    </script>
</body>
</html>
"""

# --- HTTP Web Routes ---
@app.route('/login', methods=['GET', 'POST'])
def login():
    if request.method == 'POST':
        username = request.form.get('username')
        if username in USERS:
            login_user(USERS[username])
            return redirect(url_for('admin_dashboard'))
    return '''
        <body style="background:#05070f; color:white; font-family:sans-serif; text-align:center; padding-top:120px;">
            <h1 style="color:#00e5ff;">⚡ VoltEdge Enterprise</h1>
            <form method="POST" style="background:#0d1527; display:inline-block; padding:30px; border-radius:15px; border:1px solid #1e2d4a;">
                <input type="text" name="username" placeholder="Enter Role (admin/occupant/attendant)" style="padding:12px; width:260px; border-radius:6px; border:none;"><br><br>
                <button type="submit" style="padding:12px 25px; background:#00e5ff; border:none; border-radius:6px; font-weight:bold; cursor:pointer;">Login to Command Center</button>
            </form>
        </body>
    '''

@app.route('/logout')
@login_required
def logout():
    logout_user()
    return redirect(url_for('login'))

@app.route('/video_feed')
@login_required
def video_feed():
    return Response(gen_frames(), mimetype='multipart/x-mixed-replace; boundary=frame')

@app.route('/')
@app.route('/admin')
@login_required
def admin_dashboard():
    return render_template_string(HTML_TEMPLATE)

@app.route('/api/override', methods=['POST'])
@login_required
def set_override():
    data = request.json
    zone = data.get("zone")
    mode = data.get("mode")
    if zone in manual_overrides and mode in ["AUTO", "FORCE_ON", "FORCE_OFF"]:
        manual_overrides[zone] = mode
        return jsonify({"status": "success", "zone": zone, "mode": mode})
    return jsonify({"status": "error"}), 400

@app.route('/api/alarm', methods=['POST'])
@login_required
def trigger_alarm():
    state = request.json.get("state")
    executor.submit(async_send, f"{ESP_IP}/alarm/{state}")
    return jsonify({"status": "sent", "state": state})

@app.route('/api/stats')
@login_required
def get_stats():
    kwh_saved = (total_idle_seconds / 3600.0) * POWER_RATING_KW_PER_ZONE
    rupees_saved = kwh_saved * TARIFF_PER_KWH
    return jsonify({
        "relays": relay_states,
        "overrides": manual_overrides,
        "kwh_saved": round(kwh_saved, 4),
        "rupees_saved": round(rupees_saved, 2),
        "alerts": service_alerts[-5:],
        "environment": dht_data
    })

@app.route('/api/activity_data')
@login_required
def get_activity_data():
    labels = list(hourly_activity.keys())
    datasets = []
    colors = ['#FF5722', '#2196F3', '#4CAF50', '#FFEB3B']
    for idx, z in enumerate(POLYGON_ZONES.keys()):
        data_in_mins = [round(hourly_activity[h][z] / 60.0, 1) for h in labels]
        datasets.append({
            "label": z.upper(),
            "data": data_in_mins,
            "backgroundColor": colors[idx % 4]
        })
    return jsonify({"labels": labels, "datasets": datasets})

if __name__ == '__main__':
    app.run(host='0.0.0.0', port=5000, debug=False, threaded=True)
