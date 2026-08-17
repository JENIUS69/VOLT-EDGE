from flask import Flask, render_template_string, Response, jsonify, request, redirect, url_for
from flask_login import LoginManager, UserMixin, login_user, login_required, logout_user, current_user
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
app.secret_key = 'voltedge_master_key'

# --- 1. AUTHENTICATION & ROLES ---
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
    'assistant': User(2, 'assistant', '12345678', 'assistant'),
    'guest': User(3, 'guest', '12345678', 'guest')
}

@login_manager.user_loader
def load_user(user_id):
    for u in USERS.values():
        if str(u.id) == str(user_id):
            return u
    return None

# --- 2. CONFIGURATION & STATE MANAGEMENT ---
ESP_IP = "http://192.168.4.1"
executor = ThreadPoolExecutor(max_workers=8)

FRAME_W, FRAME_H = 640, 480
COLS, ROWS = 2, 2
COL_W, ROW_H = FRAME_W // COLS, FRAME_H // ROWS

POLYGON_ZONES = {}
for r in range(ROWS):
    for c in range(COLS):
        z_idx = r * COLS + c + 1
        x1, y1, x2, y2 = c * COL_W, r * ROW_H, (c + 1) * COL_W, (r + 1) * ROW_H
        POLYGON_ZONES[f"zone{z_idx}"] = np.array([[x1, y1], [x2, y1], [x2, y2], [x1, y2]], np.int32)

relay_states = {f"zone{i}": {"light": False, "fan": False} for i in range(1, 5)}
manual_overrides = {f"zone{i}": {"light": "AUTO", "fan": "AUTO"} for i in range(1, 5)}
last_seen_times = {f"zone{i}": 0 for i in range(1, 5)}
timeline_history = collections.deque(maxlen=15)

sensor_data = {"temp": "24.5", "humidity": "55.0"}
service_requests = []
total_idle_seconds = 0
POWER_RATING_KW = 0.16

security_mode = False
alarm_active = False
current_person_count = 0
temp_cutoff = 35.0

# --- 3. SENSOR POLLING THREAD ---
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

# --- 4. VISION PIPELINE (YOLO + HAND GESTURES) ---
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
        self.thread = threading.Thread(target=self.update, daemon=True)
        self.thread.start()

    def update(self):
        while self.running:
            self.ret, self.frame = self.cap.read()
            time.sleep(0.01)

    def get_frame(self):
        return self.frame.copy() if self.ret else None

cam = CameraStream()

def async_send(url):
    try: requests.get(url, timeout=0.3)
    except: pass

def update_appliance(zone, appliance, turn_on):
    if zone in relay_states and relay_states[zone][appliance] != turn_on:
        relay_states[zone][appliance] = turn_on
        state_str = "on" if turn_on else "off"
        executor.submit(async_send, f"{ESP_IP}/{zone}/{appliance}/{state_str}")

last_alert_time, last_timeline_tick = 0, 0
active_gesture_type, gesture_start_time = None, 0
req_id_counter = 1

def gen_frames():
    global total_idle_seconds, last_alert_time, last_timeline_tick
    global current_person_count, alarm_active, security_mode, temp_cutoff
    global active_gesture_type, gesture_start_time, req_id_counter

    while True:
        frame = cam.get_frame()
        if frame is None: continue

        current_time = time.time()
        time_str = datetime.now().strftime("%H:%M:%S")

        try:
            curr_temp = float(sensor_data.get("temp", 0))
            temp_exceeded = curr_temp > temp_cutoff
        except: temp_exceeded = False

        if temp_exceeded and (current_time - last_alert_time > 10):
            service_requests.insert(0, {"id": req_id_counter, "item": f"🔥 TEMP OVERHEAT ({curr_temp}°C)", "time": time_str, "status": "URGENT", "zone": "GLOBAL"})
            req_id_counter += 1
            last_alert_time = current_time

        # Gesture Detection for Service Requests (Water, Coffee, Assistance)
        rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        gesture_results = hands.process(rgb_frame)
        detected_gesture = None
        if gesture_results.multi_hand_landmarks:
            for hand_lm in gesture_results.multi_hand_landmarks:
                lm = hand_lm.landmark
                fingers_up = sum(1 for tip, mcp in [(8,5), (12,9), (16,13), (20,17)] if lm[tip].y < lm[mcp].y)
                if fingers_up == 5: detected_gesture = "EMERGENCY HELP ✋"
                elif fingers_up == 2: detected_gesture = "COFFEE REQUEST ☕"
                elif fingers_up == 1: detected_gesture = "WATER REQUEST 💧"

        if detected_gesture:
            if active_gesture_type != detected_gesture:
                active_gesture_type = detected_gesture
                gesture_start_time = current_time
            else:
                elapsed = current_time - gesture_start_time
                cv2.putText(frame, f"HOLD GESTURE: {elapsed:.1f}s / 1.5s", (15, 65), cv2.FONT_HERSHEY_SIMPLEX, 0.55, (0, 242, 254), 2)
                if elapsed >= 1.5 and (current_time - last_alert_time > 3):
                    service_requests.insert(0, {"id": req_id_counter, "item": detected_gesture, "time": time_str, "status": "PENDING", "zone": "CAMERA Z1"})
                    req_id_counter += 1
                    last_alert_time = current_time
                    active_gesture_type = None
        else:
            active_gesture_type = None

        # YOLO Body & Zone Tracking
        results = yolo_model.track(frame, persist=True, tracker="bytetrack.yaml", verbose=False, imgsz=640, conf=0.45)
        person_count = 0
        if results[0].boxes.id is not None:
            boxes = results[0].boxes.xyxy.int().cpu().tolist()
            class_ids = results[0].boxes.cls.int().cpu().tolist()
            track_ids = results[0].boxes.id.int().cpu().tolist()

            for box, cls_id, track_id in zip(boxes, class_ids, track_ids):
                if cls_id == 0:
                    person_count += 1
                    x1, y1, x2, y2 = box
                    cx, cy = (x1 + x2) // 2, (y1 + y2) // 2
                    z_idx = min(max(cy // ROW_H, 0), ROWS - 1) * COLS + min(max(cx // COL_W, 0), COLS - 1) + 1
                    active_zone = f"zone{z_idx}"
                    last_seen_times[active_zone] = current_time

                    cv2.rectangle(frame, (x1, y1), (x2, y2), (0, 242, 254), 2)
                    cv2.circle(frame, (cx, cy), 5, (0, 0, 255), -1)
                    cv2.putText(frame, f"PERSON #{track_id} [{active_zone.upper()}]", (x1, y1 - 8), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 242, 254), 2)

        current_person_count = person_count

        if security_mode and current_person_count > 0:
            if not alarm_active:
                alarm_active = True
                executor.submit(async_send, f"{ESP_IP}/alarm/on")
            if current_time - last_alert_time > 4:
                service_requests.insert(0, {"id": req_id_counter, "item": "🚨 SECURITY BREACH", "time": time_str, "status": "ALERT", "zone": "ALL"})
                req_id_counter += 1
                last_alert_time = current_time

        # Zone Relay Control
        active_states = {}
        for z_name, poly in POLYGON_ZONES.items():
            is_active = (current_time - last_seen_times.get(z_name, 0)) <= 4.0
            active_states[z_name] = 1 if is_active else 0

            m_light = manual_overrides[z_name]["light"]
            update_appliance(z_name, "light", True if m_light == "FORCE_ON" else (False if m_light == "FORCE_OFF" else is_active))

            m_fan = manual_overrides[z_name]["fan"]
            if temp_exceeded: update_appliance(z_name, "fan", False)
            else: update_appliance(z_name, "fan", True if m_fan == "FORCE_ON" else (False if m_fan == "FORCE_OFF" else is_active))

            if not is_active: total_idle_seconds += (1 / 30.0)

            cv2.polylines(frame, [poly], True, (40, 50, 70), 1)
            x_s, y_s = poly[0][0] + 15, poly[0][1] + 45
            cv2.putText(frame, f"{z_name.upper()}: {'OCCUPIED' if is_active else 'VACANT'}", (x_s, y_s), cv2.FONT_HERSHEY_DUPLEX, 0.45, (0, 255, 128) if is_active else (120, 120, 120), 1)

        # Simple HUD Overlay
        cv2.rectangle(frame, (0, 0), (FRAME_W, 30), (9, 13, 22), -1)
        sec_str = "ARMED" if security_mode else "DISARMED"
        cv2.putText(frame, f"OCCUPANTS: {current_person_count}  |  SECURITY: {sec_str}", (12, 20), cv2.FONT_HERSHEY_DUPLEX, 0.5, (0, 242, 254), 1)

        if current_time - last_timeline_tick >= 2.0:
            timeline_history.append({"time": time_str, "z1": active_states["zone1"], "z2": active_states["zone2"], "z3": active_states["zone3"], "z4": active_states["zone4"]})
            last_timeline_tick = current_time

        ret, buffer = cv2.imencode('.jpg', frame)
        yield (b'--frame\r\n' b'Content-Type: image/jpeg\r\n\r\n' + buffer.tobytes() + b'\r\n')

# --- 5. UNIFIED DARK THEME TEMPLATES ---

COMMON_STYLES = """
<style>
    :root { --bg: #090d16; --card: #111827; --border: #1f2937; --accent: #00f2fe; --danger: #ef4444; }
    body { background: var(--bg); color: #f8fafc; font-family: system-ui, -apple-system, sans-serif; padding: 20px; margin: 0; }
    .nav { display: flex; justify-content: space-between; align-items: center; border-bottom: 2px solid #1e293b; padding-bottom: 14px; margin-bottom: 20px; }
    .nav-title { display: flex; align-items: center; gap: 10px; color: var(--accent); }
    .badge { background: #1e293b; padding: 4px 10px; border-radius: 6px; font-size: 12px; color: #94a3b8; font-weight: 600; text-transform: uppercase; }
    .telemetry-bar { display: grid; grid-template-columns: repeat(4, 1fr); gap: 12px; margin-bottom: 20px; }
    .t-box { background: var(--card); border: 1px solid var(--border); padding: 14px; border-radius: 10px; text-align: center; }
    .t-label { font-size: 11px; color: #64748b; font-weight: bold; }
    .t-val { font-size: 20px; font-weight: bold; color: var(--accent); margin-top: 4px; }
    .card { background: var(--card); border: 1px solid var(--border); border-radius: 12px; padding: 20px; margin-bottom: 20px; }
    .grid-2 { display: grid; grid-template-columns: 1.2fr 1fr; gap: 20px; }
    .zone-grid { display: grid; grid-template-columns: repeat(2, 1fr); gap: 12px; }
    .zone-box { background: #0b1120; border: 1px solid var(--border); border-radius: 8px; padding: 12px; text-align: center; }
    .btn { padding: 6px 12px; border: none; border-radius: 6px; font-weight: bold; cursor: pointer; transition: 0.2s; font-size: 12px; }
    .btn-cyan { background: var(--accent); color: #000; }
    .btn-red { background: var(--danger); color: #fff; }
    .btn-blue { background: #3b82f6; color: #fff; }
    .btn-green { background: #10b981; color: #000; }
    .btn-dark { background: #1f2937; color: #94a3b8; }
    .btn-full { width: 100%; margin-top: 8px; padding: 10px; }
    .logout-btn { color: var(--danger); border: 1px solid var(--danger); text-decoration: none; padding: 6px 14px; border-radius: 6px; font-weight: bold; font-size: 13px; }
    .req-item { display: flex; justify-content: space-between; align-items: center; background: #0b1120; border: 1px solid var(--border); padding: 10px 14px; border-radius: 8px; margin-bottom: 8px; font-size: 13px; }
</style>
"""

LOGIN_TEMPLATE = """
<!DOCTYPE html>
<html>
<head>
    <title>VoltEdge OS | Login</title>
    """ + COMMON_STYLES + """
</head>
<body style="display:flex; justify-content:center; align-items:center; height:90vh;">
    <div class="card" style="width: 320px; text-align: center; padding: 40px;">
        <h2 style="color:var(--accent); margin-top:0;">⚡ VoltEdge OS</h2>
        <p style="color:#64748b; font-size:13px; margin-bottom:25px;">Select role or enter login credentials</p>
        <form method="POST">
            <input type="text" name="username" placeholder="Username (admin / assistant / guest)" required style="width:100%; padding:12px; margin-bottom:12px; background:#070c18; border:1px solid var(--border); color:#fff; border-radius:8px; box-sizing:border-box;">
            <input type="password" name="password" placeholder="Passcode (12345678)" required style="width:100%; padding:12px; margin-bottom:20px; background:#070c18; border:1px solid var(--border); color:#fff; border-radius:8px; box-sizing:border-box;">
            <button type="submit" class="btn btn-cyan btn-full" style="padding:12px;">SIGN IN</button>
        </form>
    </div>
</body>
</html>
"""

ADMIN_DASHBOARD_TEMPLATE = """
<!DOCTYPE html>
<html>
<head>
    <title>VoltEdge OS | Admin Console</title>
    <script src="https://cdn.jsdelivr.net/npm/chart.js"></script>
    """ + COMMON_STYLES + """
</head>
<body>
    <div class="nav">
        <div class="nav-title">
            <h2 style="margin:0;">⚡ VoltEdge OS</h2>
            <span class="badge">ADMIN MASTER CONSOLE</span>
        </div>
        <a href="/logout" class="logout-btn">LOGOUT</a>
    </div>

    <div class="telemetry-bar">
        <div class="t-box"><div class="t-label">TEMPERATURE</div><div class="t-val"><span id="t-temp">--</span> °C</div></div>
        <div class="t-box"><div class="t-label">HUMIDITY</div><div class="t-val"><span id="t-hum">--</span> %</div></div>
        <div class="t-box"><div class="t-label">OCCUPANTS</div><div class="t-val" id="t-count">0</div></div>
        <div class="t-box"><div class="t-label">ENERGY SAVED</div><div class="t-val"><span id="t-kwh">0.0</span> <span style="font-size:12px;">kWh</span></div></div>
    </div>

    <div class="grid-2">
        <div>
            <div class="card">
                <h3 style="margin-top:0;">📹 Optics Stream</h3>
                <img src="/video_feed" style="width:100%; border-radius:8px;">
            </div>
            <div class="card">
                <h3 style="margin-top:0;">💡 Zone Appliance Controls</h3>
                <div class="zone-grid">
                    {% for z in ["zone1", "zone2", "zone3", "zone4"] %}
                    <div class="zone-box">
                        <strong style="color:var(--accent);">{{ z.upper() }}</strong>
                        <div style="margin-top:8px;">
                            <label style="font-size:11px; color:#64748b;">LIGHT:</label>
                            <button class="btn btn-green" onclick="setOverride('{{z}}', 'light', 'FORCE_ON')">ON</button>
                            <button class="btn btn-red" onclick="setOverride('{{z}}', 'light', 'FORCE_OFF')">OFF</button>
                        </div>
                        <div style="margin-top:6px;">
                            <label style="font-size:11px; color:#64748b;">FAN:</label>
                            <button class="btn btn-blue" onclick="setOverride('{{z}}', 'fan', 'FORCE_ON')">ON</button>
                            <button class="btn btn-red" onclick="setOverride('{{z}}', 'fan', 'FORCE_OFF')">OFF</button>
                        </div>
                    </div>
                    {% endfor %}
                </div>
            </div>
            <div class="card">
                <h3 style="margin-top:0;">📈 Zone Occupancy Analytics</h3>
                <canvas id="chart" height="110"></canvas>
            </div>
        </div>
        <div>
            <div class="card">
                <h3 style="margin-top:0;">⚙️ Master Security & Thresholds</h3>
                <label style="font-size:13px; color:#94a3b8;">Max Temp Threshold (°C):</label>
                <div style="display:flex; gap:10px; margin-top:6px; margin-bottom:12px;">
                    <input type="number" id="cutoff-in" value="35.0" step="0.5" style="padding:8px; background:#070c18; border:1px solid var(--border); color:#fff; border-radius:6px; width:90px;">
                    <button class="btn btn-cyan" onclick="setCutoff()">SAVE THRESHOLD</button>
                </div>
                <button class="btn btn-blue btn-full" onclick="toggleSecurity()">TOGGLE SECURITY SYSTEM</button>
                <button class="btn btn-red btn-full" onclick="triggerAlarm(true)">TRIGGER SIREN OVERRIDE</button>
                <button class="btn btn-dark btn-full" onclick="triggerAlarm(false)">SILENCE SIREN</button>
            </div>
            <div class="card">
                <h3 style="margin-top:0;">🛎️ Service & Incident Logs</h3>
                <div id="service-logs" style="max-height:350px; overflow-y:auto;"></div>
            </div>
        </div>
    </div>

    <script>
        let ctx = document.getElementById('chart').getContext('2d');
        let chart = new Chart(ctx, {
            type: 'line',
            data: { labels: [], datasets: [
                { label: 'Z1', data: [], borderColor: '#00f2fe' },
                { label: 'Z2', data: [], borderColor: '#10b981' },
                { label: 'Z3', data: [], borderColor: '#f59e0b' },
                { label: 'Z4', data: [], borderColor: '#ef4444' }
            ]},
            options: {
                scales: {
                    y: { min: 0, max: 1, ticks: { stepSize: 1 } }
                }
            }
        });

        async function setOverride(zone, appliance, mode) {
            await fetch('/api/override', { method: 'POST', headers: {'Content-Type': 'application/json'}, body: JSON.stringify({zone, appliance, mode}) });
        }
        async function setCutoff() {
            let val = parseFloat(document.getElementById('cutoff-in').value);
            await fetch('/api/temp_cutoff', { method: 'POST', headers: {'Content-Type': 'application/json'}, body: JSON.stringify({temp_cutoff: val}) });
        }
        async function toggleSecurity() { await fetch('/api/security_mode', { method: 'POST' }); }
        async function triggerAlarm(enable) { await fetch('/api/alarm', { method: 'POST', headers: {'Content-Type': 'application/json'}, body: JSON.stringify({enable}) }); }

        setInterval(async () => {
            let r = await fetch('/api/stats');
            let d = await r.json();
            document.getElementById('t-temp').innerText = d.temp;
            document.getElementById('t-hum').innerText = d.humidity;
            document.getElementById('t-count').innerText = d.person_count;
            document.getElementById('t-kwh').innerText = d.kwh_saved;
            
            document.getElementById('service-logs').innerHTML = d.requests.map(s => 
                `<div class="req-item"><span>${s.time} - <strong>${s.item}</strong> (${s.zone})</span><span class="badge">${s.status}</span></div>`
            ).join('');

            let tR = await fetch('/api/timeline_data');
            let tD = await tR.json();
            chart.data.labels = tD.labels;
            chart.data.datasets[0].data = tD.z1;
            chart.data.datasets[1].data = tD.z2;
            chart.data.datasets[2].data = tD.z3;
            chart.data.datasets[3].data = tD.z4;
            chart.update();
        }, 2000);
    </script>
</body>
</html>
"""

ASSISTANT_DASHBOARD_TEMPLATE = """
<!DOCTYPE html>
<html>
<head>
    <title>VoltEdge OS | Assistant Operations</title>
    """ + COMMON_STYLES + """
</head>
<body>
    <div class="nav">
        <div class="nav-title">
            <h2 style="margin:0;">⚡ VoltEdge OS</h2>
            <span class="badge" style="background:#064e3b; color:#34d399;">ASSISTANT DISPATCH</span>
        </div>
        <a href="/logout" class="logout-btn">LOGOUT</a>
    </div>

    <div class="telemetry-bar">
        <div class="t-box"><div class="t-label">TEMPERATURE</div><div class="t-val"><span id="t-temp">--</span> °C</div></div>
        <div class="t-box"><div class="t-label">HUMIDITY</div><div class="t-val"><span id="t-hum">--</span> %</div></div>
        <div class="t-box"><div class="t-label">OCCUPANTS</div><div class="t-val" id="t-count">0</div></div>
        <div class="t-box"><div class="t-label">PENDING CALLS</div><div class="t-val" style="color:#f59e0b;" id="t-req-count">0</div></div>
    </div>

    <div class="grid-2">
        <div>
            <div class="card">
                <h3 style="margin-top:0;">🛎️ Incoming Help & Service Calls</h3>
                <p style="color:#64748b; font-size:12px; margin-top:-8px;">Accept calls once fullfilled (Water, Coffee, Assistance)</p>
                <div id="request-queue" style="max-height:450px; overflow-y:auto;"></div>
            </div>
            <div class="card">
                <h3 style="margin-top:0;">💡 Direct Appliance Toggles</h3>
                <div class="zone-grid">
                    {% for z in ["zone1", "zone2", "zone3", "zone4"] %}
                    <div class="zone-box">
                        <strong style="color:var(--accent);">{{ z.upper() }}</strong>
                        <div style="margin-top:8px;">
                            <button class="btn btn-green" onclick="setOverride('{{z}}', 'light', 'FORCE_ON')">L-ON</button>
                            <button class="btn btn-red" onclick="setOverride('{{z}}', 'light', 'FORCE_OFF')">OFF</button>
                        </div>
                        <div style="margin-top:4px;">
                            <button class="btn btn-blue" onclick="setOverride('{{z}}', 'fan', 'FORCE_ON')">F-ON</button>
                            <button class="btn btn-red" onclick="setOverride('{{z}}', 'fan', 'FORCE_OFF')">OFF</button>
                        </div>
                    </div>
                    {% endfor %}
                </div>
            </div>
        </div>
        <div>
            <div class="card">
                <h3 style="margin-top:0;">📹 Real-Time Optics Feed</h3>
                <img src="/video_feed" style="width:100%; border-radius:8px;">
            </div>
        </div>
    </div>

    <script>
        async function acceptReq(reqId) {
            await fetch('/api/complete_request', { method: 'POST', headers: {'Content-Type': 'application/json'}, body: JSON.stringify({id: reqId}) });
        }
        async function setOverride(zone, appliance, mode) {
            await fetch('/api/override', { method: 'POST', headers: {'Content-Type': 'application/json'}, body: JSON.stringify({zone, appliance, mode}) });
        }

        setInterval(async () => {
            let r = await fetch('/api/stats');
            let d = await r.json();
            document.getElementById('t-temp').innerText = d.temp;
            document.getElementById('t-hum').innerText = d.humidity;
            document.getElementById('t-count').innerText = d.person_count;
            
            let pending = d.requests.filter(s => s.status !== 'DONE');
            document.getElementById('t-req-count').innerText = pending.length;

            if(d.requests.length === 0) {
                document.getElementById('request-queue').innerHTML = '<div style="color:#64748b; font-size:13px;">No pending service requests.</div>';
            } else {
                document.getElementById('request-queue').innerHTML = d.requests.map(s => 
                    `<div class="req-item">
                        <div>
                            <strong>${s.item}</strong> 
                            <span style="color:#64748b; font-size:11px;">[${s.zone} @ ${s.time}]</span>
                        </div>
                        ${s.status === 'DONE' 
                            ? `<span class="badge" style="background:#064e3b; color:#34d399;">DONE</span>` 
                            : `<button class="btn btn-green" onclick="acceptReq(${s.id})">ACCEPT & COMPLETE</button>`}
                    </div>`
                ).join('');
            }
        }, 1500);
    </script>
</body>
</html>
"""

GUEST_DASHBOARD_TEMPLATE = """
<!DOCTYPE html>
<html>
<head>
    <title>VoltEdge OS | Guest Dashboard</title>
    """ + COMMON_STYLES + """
</head>
<body>
    <div class="nav">
        <div class="nav-title">
            <h2 style="margin:0;">⚡ VoltEdge OS</h2>
            <span class="badge" style="background:#78350f; color:#fde68a;">GUEST PORTAL</span>
        </div>
        <a href="/logout" class="logout-btn">LOGOUT</a>
    </div>

    <div class="telemetry-bar">
        <div class="t-box"><div class="t-label">ROOM TEMP</div><div class="t-val"><span id="t-temp">--</span> °C</div></div>
        <div class="t-box"><div class="t-label">HUMIDITY</div><div class="t-val"><span id="t-hum">--</span> %</div></div>
        <div class="t-box"><div class="t-label">OCCUPANTS</div><div class="t-val" id="t-count">0</div></div>
        <div class="t-box"><div class="t-label">ACCESS LEVEL</div><div class="t-val" style="color:#f59e0b;">LIGHT & FAN</div></div>
    </div>

    <div class="grid-2">
        <div>
            <div class="card">
                <h3 style="margin-top:0;">💡 Guest Room Controls (Lights & Fans)</h3>
                <div class="zone-grid">
                    {% for z in ["zone1", "zone2", "zone3", "zone4"] %}
                    <div class="zone-box">
                        <strong style="color:var(--accent);">{{ z.upper() }}</strong>
                        <div style="margin-top:10px;">
                            <label style="font-size:12px; color:#94a3b8;">LIGHTS:</label><br>
                            <button class="btn btn-green" style="margin-top:4px;" onclick="setOverride('{{z}}', 'light', 'FORCE_ON')">ON</button>
                            <button class="btn btn-red" style="margin-top:4px;" onclick="setOverride('{{z}}', 'light', 'FORCE_OFF')">OFF</button>
                        </div>
                        <div style="margin-top:10px;">
                            <label style="font-size:12px; color:#94a3b8;">FAN:</label><br>
                            <button class="btn btn-blue" style="margin-top:4px;" onclick="setOverride('{{z}}', 'fan', 'FORCE_ON')">ON</button>
                            <button class="btn btn-red" style="margin-top:4px;" onclick="setOverride('{{z}}', 'fan', 'FORCE_OFF')">OFF</button>
                        </div>
                    </div>
                    {% endfor %}
                </div>
            </div>
            <div class="card">
                <h3 style="margin-top:0;">☕ Request Service</h3>
                <div style="display:flex; gap:10px;">
                    <button class="btn btn-cyan" style="flex:1; padding:12px;" onclick="sendReq('💧 WATER REQUEST')">Request Water</button>
                    <button class="btn btn-blue" style="flex:1; padding:12px;" onclick="sendReq('☕ COFFEE REQUEST')">Request Coffee</button>
                    <button class="btn btn-red" style="flex:1; padding:12px;" onclick="sendReq('✋ ASSISTANCE NEEDED')">Call Help</button>
                </div>
            </div>
        </div>
        <div>
            <div class="card">
                <h3 style="margin-top:0;">📹 Real-Time Optics Feed</h3>
                <img src="/video_feed" style="width:100%; border-radius:8px;">
            </div>
        </div>
    </div>

    <script>
        async function setOverride(zone, appliance, mode) {
            await fetch('/api/override', { method: 'POST', headers: {'Content-Type': 'application/json'}, body: JSON.stringify({zone, appliance, mode}) });
        }
        async function sendReq(item) {
            await fetch('/api/service_request', { method: 'POST', headers: {'Content-Type': 'application/json'}, body: JSON.stringify({item, zone: 'GUEST ROOM'}) });
        }

        setInterval(async () => {
            let r = await fetch('/api/stats');
            let d = await r.json();
            document.getElementById('t-temp').innerText = d.temp;
            document.getElementById('t-hum').innerText = d.humidity;
            document.getElementById('t-count').innerText = d.person_count;
        }, 2000);
    </script>
</body>
</html>
"""

# --- 6. ROUTING & API ENDPOINTS ---
@app.route('/login', methods=['GET', 'POST'])
def login():
    if request.method == 'POST':
        user = USERS.get(request.form.get('username'))
        if user and user.password == request.form.get('password'):
            login_user(user)
            return redirect(url_for('dashboard'))
    return render_template_string(LOGIN_TEMPLATE)

@app.route('/logout')
@login_required
def logout():
    logout_user()
    return redirect(url_for('login'))

@app.route('/')
@login_required
def dashboard():
    if current_user.role == 'admin':
        return render_template_string(ADMIN_DASHBOARD_TEMPLATE, user=current_user)
    elif current_user.role == 'assistant':
        return render_template_string(ASSISTANT_DASHBOARD_TEMPLATE, user=current_user)
    else:
        return render_template_string(GUEST_DASHBOARD_TEMPLATE, user=current_user)

@app.route('/video_feed')
@login_required
def video_feed():
    return Response(gen_frames(), mimetype='multipart/x-mixed-replace; boundary=frame')

@app.route('/api/override', methods=['POST'])
@login_required
def set_override():
    d = request.json
    manual_overrides[d['zone']][d['appliance']] = d['mode']
    return jsonify({"status": "success"})

@app.route('/api/service_request', methods=['POST'])
@login_required
def create_service_request():
    global req_id_counter
    d = request.json
    time_str = datetime.now().strftime("%H:%M:%S")
    service_requests.insert(0, {"id": req_id_counter, "item": d.get("item", "GENERAL ASSIST"), "time": time_str, "status": "PENDING", "zone": d.get("zone", "GUEST")})
    req_id_counter += 1
    return jsonify({"status": "success"})

@app.route('/api/complete_request', methods=['POST'])
@login_required
def complete_service_request():
    if current_user.role not in ['admin', 'assistant']: 
        return jsonify({"error": "Unauthorized"}), 403
    req_id = request.json.get("id")
    for req in service_requests:
        if req["id"] == req_id:
            req["status"] = "DONE"
            break
    return jsonify({"status": "success"})

@app.route('/api/temp_cutoff', methods=['POST'])
@login_required
def set_temp_cutoff():
    if current_user.role != 'admin': return jsonify({"error": "Unauthorized"}), 403
    global temp_cutoff
    temp_cutoff = float(request.json.get('temp_cutoff', 35.0))
    return jsonify({"status": "success"})

@app.route('/api/security_mode', methods=['POST'])
@login_required
def toggle_security_mode():
    if current_user.role != 'admin': return jsonify({"error": "Unauthorized"}), 403
    global security_mode
    security_mode = not security_mode
    return jsonify({"status": "success", "security_mode": security_mode})

@app.route('/api/alarm', methods=['POST'])
@login_required
def control_alarm():
    if current_user.role == 'guest': return jsonify({"error": "Unauthorized"}), 403
    global alarm_active
    alarm_active = request.json.get("enable", False)
    executor.submit(async_send, f"{ESP_IP}/alarm/{'on' if alarm_active else 'off'}")
    return jsonify({"status": "success"})

@app.route('/api/stats')
@login_required
def get_stats():
    return jsonify({
        "person_count": current_person_count,
        "security_mode": security_mode,
        "temp_cutoff": temp_cutoff,
        "kwh_saved": round((total_idle_seconds / 3600.0) * POWER_RATING_KW, 4),
        "temp": sensor_data.get("temp", "24.5"),
        "humidity": sensor_data.get("humidity", "55.0"),
        "requests": service_requests[:10]
    })

@app.route('/api/timeline_data')
@login_required
def get_timeline_data():
    return jsonify({
        "labels": [i["time"] for i in timeline_history],
        "z1": [i["z1"] for i in timeline_history],
        "z2": [i["z2"] for i in timeline_history],
        "z3": [i["z3"] for i in timeline_history],
        "z4": [i["z4"] for i in timeline_history]
    })

if __name__ == '__main__':
    app.run(host='0.0.0.0', port=5000, debug=False, threaded=True)
