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

# --- 2. 4-ZONE GRID & CONFIGURATION ---
ESP_IP = "http://192.168.4.1"
executor = ThreadPoolExecutor(max_workers=8)

FRAME_W, FRAME_H = 640, 480
COLS, ROWS = 2, 2  # 2x2 Grid = 4 Zones
COL_W = FRAME_W // COLS
ROW_H = FRAME_H // ROWS

POLYGON_ZONES = {}
for r in range(ROWS):
    for c in range(COLS):
        z_idx = r * COLS + c + 1
        x1, y1 = c * COL_W, r * ROW_H
        x2, y2 = (c + 1) * COL_W, (r + 1) * ROW_H
        POLYGON_ZONES[f"zone{z_idx}"] = np.array([[x1, y1], [x2, y1], [x2, y2], [x1, y2]], np.int32)

relay_states = {f"zone{i}": {"light": False, "fan": False} for i in range(1, 5)}
manual_overrides = {f"zone{i}": {"light": "AUTO", "fan": "AUTO"} for i in range(1, 5)}

last_seen_times = {f"zone{i}": 0 for i in range(1, 5)}
timeline_history = collections.deque(maxlen=15)

sensor_data = {"temp": "--", "humidity": "--"}
service_alerts = []
total_idle_seconds = 0
POWER_RATING_KW = 0.16
TARIFF_PER_KWH = 10.00        

# Security, Person Detection & Temp Cutoff Globals
security_mode = False
alarm_active = False
current_person_count = 0
temp_cutoff = 35.0  # Default threshold temperature in °C

# --- 3. SENSOR POLLING ---
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

# --- 4. AI MODELS & CAMERA STREAM ---
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

# --- 5. VISION PIPELINE & CONTROL LOGIC ---
def async_send(url):
    try: requests.get(url, timeout=0.3)
    except: pass

def update_appliance(zone, appliance, turn_on):
    if zone in relay_states and relay_states[zone][appliance] != turn_on:
        relay_states[zone][appliance] = turn_on
        state_str = "on" if turn_on else "off"
        executor.submit(async_send, f"{ESP_IP}/{zone}/{appliance}/{state_str}")

last_alert_time = 0
last_timeline_tick = 0

def gen_frames():
    global total_idle_seconds, last_alert_time, last_timeline_tick
    global current_person_count, alarm_active, security_mode, temp_cutoff

    while True:
        frame = cam.get_frame()
        if frame is None: continue

        current_time = time.time()
        time_str = datetime.now().strftime("%H:%M:%S")

        # Temperature Safety Check (Turn OFF all fans if current temp > temp_cutoff)
        try:
            curr_temp = float(sensor_data.get("temp", 0))
            temp_exceeded = curr_temp > temp_cutoff
        except (ValueError, TypeError):
            temp_exceeded = False

        if temp_exceeded and (current_time - last_alert_time > 10):
            service_alerts.append({"type": f"🔥 TEMP EXCEEDED ({curr_temp}°C) - FANS OFF", "time": time_str})
            last_alert_time = current_time

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
                    service_alerts.append({"type": alert_type, "time": time_str})
                    last_alert_time = current_time

        # YOLO Detection & Zone Tracking
        results = yolo_model.track(frame, persist=True, tracker="bytetrack.yaml", verbose=False, imgsz=640, conf=0.45)
        
        person_detected_count = 0
        if results[0].boxes.id is not None:
            boxes = results[0].boxes.xyxy.int().cpu().tolist()
            class_ids = results[0].boxes.cls.int().cpu().tolist()
            track_ids = results[0].boxes.id.int().cpu().tolist()

            for box, cls_id, track_id in zip(boxes, class_ids, track_ids):
                if cls_id == 0:  # Person Class
                    person_detected_count += 1
                    x1, y1, x2, y2 = box
                    cx = (x1 + x2) // 2
                    cy = (y1 + y2) // 2

                    grid_c = min(max(cx // COL_W, 0), COLS - 1)
                    grid_r = min(max(cy // ROW_H, 0), ROWS - 1)
                    z_idx = grid_r * COLS + grid_c + 1
                    active_zone_name = f"zone{z_idx}"

                    last_seen_times[active_zone_name] = current_time

                    cv2.rectangle(frame, (x1, y1), (x2, y2), (0, 229, 255), 2)
                    cv2.circle(frame, (cx, cy), 5, (0, 0, 255), -1)
                    cv2.putText(frame, f"ID:{track_id} ({active_zone_name.upper()})", 
                                (x1, y1 - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 229, 255), 2)

        current_person_count = person_detected_count

        # Security Mode Check
        if security_mode and current_person_count > 0:
            if not alarm_active:
                alarm_active = True
                executor.submit(async_send, f"{ESP_IP}/alarm/on")
            if current_time - last_alert_time > 4:
                service_alerts.append({"type": "🚨 SECURITY BREACH DETECTED", "time": time_str})
                last_alert_time = current_time

        # Zone Automation & Grid Overlay
        active_states = {}
        for z_name, poly in POLYGON_ZONES.items():
            is_active = (current_time - last_seen_times.get(z_name, 0)) <= 4.0
            active_states[z_name] = 1 if is_active else 0

            # Light Control
            mode_light = manual_overrides[z_name]["light"]
            if mode_light == "FORCE_ON": update_appliance(z_name, "light", True)
            elif mode_light == "FORCE_OFF": update_appliance(z_name, "light", False)
            else: update_appliance(z_name, "light", is_active)

            # Fan Control (Auto-off if temperature cutoff exceeded)
            mode_fan = manual_overrides[z_name]["fan"]
            if temp_exceeded:
                update_appliance(z_name, "fan", False)
            elif mode_fan == "FORCE_ON": update_appliance(z_name, "fan", True)
            elif mode_fan == "FORCE_OFF": update_appliance(z_name, "fan", False)
            else: update_appliance(z_name, "fan", is_active)

            if not is_active:
                total_idle_seconds += (1 / 30.0)

            cv2.polylines(frame, [poly], True, (255, 255, 255), 1)
            x_start, y_start = poly[0][0] + 15, poly[0][1] + 35
            text_color = (0, 255, 0) if is_active else (120, 120, 120)
            cv2.putText(frame, f"{z_name.upper()}: {'ACTIVE' if is_active else 'EMPTY'}", 
                        (x_start, y_start), cv2.FONT_HERSHEY_DUPLEX, 0.6, text_color, 1)

        # On-screen camera HUD
        hud_sec_status = "ARMED" if security_mode else "DISARMED"
        hud_color = (0, 0, 255) if security_mode else (0, 255, 0)
        cv2.putText(frame, f"PEOPLE: {current_person_count} | SEC: {hud_sec_status}", 
                    (15, 30), cv2.FONT_HERSHEY_DUPLEX, 0.7, hud_color, 2)

        # Timeline recording interval
        if current_time - last_timeline_tick >= 2.0:
            timeline_history.append({
                "time": time_str,
                "z1": active_states["zone1"],
                "z2": active_states["zone2"],
                "z3": active_states["zone3"],
                "z4": active_states["zone4"]
            })
            last_timeline_tick = current_time

        ret, buffer = cv2.imencode('.jpg', frame)
        yield (b'--frame\r\n' b'Content-Type: image/jpeg\r\n\r\n' + buffer.tobytes() + b'\r\n')

# --- 6. DASHBOARD TEMPLATES ---
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
    <script src="/static/chart.js"></script>
    <style>
        :root { --accent: #00e5ff; --bg: #000; --panel: #0a0a0a; --text: #e0e0e0; }
        body { background: var(--bg); color: var(--text); font-family: 'Segoe UI', sans-serif; padding: 25px; margin: 0; }
        
        .header { display: flex; justify-content: space-between; align-items: center; margin-bottom: 25px; padding-bottom: 15px; border-bottom: 1px solid #222; }
        .header h2 { color: var(--accent); margin: 0; letter-spacing: 2px; font-weight: 700; text-transform: uppercase; }
        .logout-btn { color: #ff4757; text-decoration: none; background: #111; padding: 10px 20px; border-radius: 8px; border: 1px solid #333; font-weight: bold; }
        
        .grid { display: grid; grid-template-columns: 1.2fr 1fr; gap: 25px; }
        .card { background: var(--panel); border: 1px solid #1a1a1a; border-radius: 16px; padding: 25px; box-shadow: 0 10px 30px rgba(0,0,0,0.5); }
        .card h3 { margin-top: 0; color: #fff; font-size: 14px; border-left: 4px solid var(--accent); padding-left: 12px; text-transform: uppercase; letter-spacing: 1px; }
        
        #video-feed { width: 100%; border-radius: 12px; border: 2px solid #222; }
        
        .stat-grid { display: grid; grid-template-columns: 1fr 1fr 1fr; gap: 12px; margin-bottom: 25px; }
        .stat-box { background: #111; padding: 15px; border-radius: 12px; border: 1px solid #222; text-align: center; }
        .stat-title { font-size: 10px; color: #888; text-transform: uppercase; letter-spacing: 1px; display: block; margin-bottom: 8px; }
        .stat-val { font-size: 20px; font-weight: bold; }
        
        .zones-grid { display: grid; grid-template-columns: 1fr 1fr; gap: 15px; }
        .zone-card { background: #111; padding: 15px; border-radius: 12px; border: 1px solid #222; }
        .zone-title { font-weight: bold; color: var(--accent); font-size: 14px; text-align: center; margin-bottom: 10px; }
        
        .ctrl-row { display: flex; align-items: center; justify-content: space-between; margin-top: 8px; font-size: 11px; font-weight: bold; color: #aaa; }
        .btn-group { display: flex; gap: 4px; }
        .btn { padding: 4px 8px; border: none; border-radius: 4px; cursor: pointer; font-weight: bold; font-size: 9px; color: #fff; }
        .btn-auto { background: #2980b9; }
        .btn-on { background: #27ae60; }
        .btn-off { background: #c0392b; }
        
        .btn-sec-mode { background: #111; color: #00e5ff; width: 100%; padding: 12px; font-size: 13px; text-transform: uppercase; border-radius: 10px; font-weight: bold; cursor: pointer; border: 1px solid #00e5ff; margin-bottom: 12px; transition: 0.3s; }
        .btn-sec-mode.armed { background: #ff4757; color: white; border-color: #ff4757; }
        .btn-alarm { background: linear-gradient(45deg, #ff4757, #ff6b81); color: white; width: 100%; padding: 12px; font-size: 13px; text-transform: uppercase; border-radius: 10px; font-weight: bold; cursor: pointer; border: none; }
        .btn-silence { background: #222; color: #fff; width: 100%; padding: 10px; margin-top: 8px; border: 1px solid #444; border-radius: 10px; cursor: pointer; font-weight: bold; }
        
        .alert-item { background: rgba(0, 229, 255, 0.05); border: 1px solid rgba(0, 229, 255, 0.2); padding: 10px; margin-bottom: 8px; border-radius: 8px; color: var(--accent); font-size: 12px; display: flex; justify-content: space-between; }
        .temp-input-box { display: flex; gap: 8px; align-items: center; margin-top: 10px; }
        .temp-input { width: 60px; padding: 6px; background: #222; border: 1px solid #444; color: #fff; border-radius: 6px; text-align: center; }
    </style>
</head>
<body>
    <div class="header">
        <h2>⚡ VoltEdge OS Command (4-Zone Array)</h2>
        <a href="/logout" class="logout-btn">DISCONNECT</a>
    </div>

    <div class="grid">
        <div style="display: flex; flex-direction: column; gap: 25px;">
            <div class="card">
                <h3>Optical Sensor Array (2x2 Quad Grid)</h3>
                <img id="video-feed" src="/video_feed">
            </div>
            
            <div class="card">
                <h3>Real-Time Zone Occupancy Timeline</h3>
                <canvas id="timelineChart" height="110"></canvas>
            </div>
        </div>

        <div style="display: flex; flex-direction: column; gap: 25px;">
            <div class="stat-grid">
                <div class="stat-box">
                    <span class="stat-title">People Present</span>
                    <span class="stat-val" style="color:#ffdd59;" id="person-count">0</span>
                </div>
                <div class="stat-box">
                    <span class="stat-title">Energy Conserved</span>
                    <span class="stat-val" style="color:#00e5ff;" id="kwh">0.00 <span style="font-size:12px">kWh</span></span>
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
                <div class="stat-box">
                    <span class="stat-title">Fan Cutoff Temp</span>
                    <span class="stat-val" style="color:#ff6b6b;" id="cutoff-display">35.0 °C</span>
                </div>
            </div>

            <div class="card">
                <h3>Temperature Safety Threshold</h3>
                <div style="font-size: 12px; color: #888;">If ambient temperature exceeds this limit, all fans will turn OFF automatically:</div>
                <div class="temp-input-box">
                    <input type="number" id="temp-cutoff-input" class="temp-input" value="35.0" step="0.5">
                    <button class="btn btn-auto" style="padding: 8px 14px; font-size:11px;" onclick="updateTempCutoff()">SET CUTOFF (°C)</button>
                </div>
            </div>

            <div class="card">
                <h3>Dual-Appliance Control Matrix</h3>
                <div class="zones-grid">
                    {% for z in ["zone1", "zone2", "zone3", "zone4"] %}
                    <div class="zone-card">
                        <div class="zone-title">{{ z.upper() }}</div>
                        <div class="ctrl-row">
                            <span>💡 LIGHT</span>
                            <div class="btn-group">
                                <button class="btn btn-auto" onclick="setOverride('{{z}}', 'light', 'AUTO')">AUTO</button>
                                <button class="btn btn-on" onclick="setOverride('{{z}}', 'light', 'FORCE_ON')">ON</button>
                                <button class="btn btn-off" onclick="setOverride('{{z}}', 'light', 'FORCE_OFF')">OFF</button>
                            </div>
                        </div>
                        <div class="ctrl-row">
                            <span>❄️ FAN</span>
                            <div class="btn-group">
                                <button class="btn btn-auto" onclick="setOverride('{{z}}', 'fan', 'AUTO')">AUTO</button>
                                <button class="btn btn-on" onclick="setOverride('{{z}}', 'fan', 'FORCE_ON')">ON</button>
                                <button class="btn btn-off" onclick="setOverride('{{z}}', 'fan', 'FORCE_OFF')">OFF</button>
                            </div>
                        </div>
                    </div>
                    {% endfor %}
                </div>
            </div>

            <div class="card">
                <h3>System Audio & Security Alarm</h3>
                <button id="sec-btn" class="btn-sec-mode" onclick="toggleSecurityMode()">🛡️ Security Mode: OFF</button>
                <button class="btn-alarm" onclick="triggerAlarm(true)">🚨 Sound Alarm Siren</button>
                <button class="btn-silence" onclick="triggerAlarm(false)">Silence Siren</button>
                
                <h4 style="margin-top:20px; color:#666; font-size: 11px; text-transform: uppercase;">Live System Alerts</h4>
                <div id="alerts"></div>
            </div>
        </div>
    </div>

    <script>
        Chart.defaults.color = '#888';
        Chart.defaults.borderColor = '#222';
        
        let ctx = document.getElementById('timelineChart').getContext('2d');
        let timelineChart = new Chart(ctx, {
            type: 'line',
            data: {
                labels: [],
                datasets: [
                    { label: 'ZONE 1', data: [], borderColor: '#00e5ff', backgroundColor: '#00e5ff', tension: 0.2 },
                    { label: 'ZONE 2', data: [], borderColor: '#2ecc71', backgroundColor: '#2ecc71', tension: 0.2 },
                    { label: 'ZONE 3', data: [], borderColor: '#f1c40f', backgroundColor: '#f1c40f', tension: 0.2 },
                    { label: 'ZONE 4', data: [], borderColor: '#e74c3c', backgroundColor: '#e74c3c', tension: 0.2 }
                ]
            },
            options: {
                responsive: true,
                scales: {
                    y: { 
                        min: 0, 
                        max: 1.2, 
                        ticks: { stepSize: 1, callback: v => v === 1 ? 'ACTIVE' : (v === 0 ? 'IDLE' : '') } 
                    }
                },
                plugins: { legend: { position: 'top', labels:{ boxWidth: 10, font:{ size: 10 } } } }
            }
        });

        let audioCtx = null;
        let sirenOsc = null;

        function playSirenSound(enable) {
            if (enable) {
                if (!audioCtx) audioCtx = new (window.AudioContext || window.webkitAudioContext)();
                if (sirenOsc) return;
                sirenOsc = audioCtx.createOscillator();
                let gain = audioCtx.createGain();
                sirenOsc.type = 'sawtooth';
                sirenOsc.frequency.setValueAtTime(800, audioCtx.currentTime);
                sirenOsc.frequency.exponentialRampToValueAtTime(400, audioCtx.currentTime + 0.5);
                sirenOsc.connect(gain);
                gain.connect(audioCtx.destination);
                sirenOsc.start();
            } else {
                if (sirenOsc) {
                    sirenOsc.stop();
                    sirenOsc.disconnect();
                    sirenOsc = null;
                }
            }
        }

        async function triggerAlarm(enable) {
            playSirenSound(enable);
            await fetch('/api/alarm', {
                method: 'POST',
                headers: {'Content-Type': 'application/json'},
                body: JSON.stringify({enable})
            });
        }

        async function toggleSecurityMode() {
            let res = await fetch('/api/security_mode', { method: 'POST' });
            let data = await res.json();
            updateSecurityUI(data.security_mode);
        }

        async function updateTempCutoff() {
            let val = parseFloat(document.getElementById('temp-cutoff-input').value);
            await fetch('/api/temp_cutoff', {
                method: 'POST',
                headers: {'Content-Type': 'application/json'},
                body: JSON.stringify({temp_cutoff: val})
            });
        }

        function updateSecurityUI(isArmed) {
            let btn = document.getElementById('sec-btn');
            if (isArmed) {
                btn.innerText = "🛡️ SECURITY MODE: ARMED";
                btn.classList.add('armed');
            } else {
                btn.innerText = "🛡️ SECURITY MODE: OFF";
                btn.classList.remove('armed');
            }
        }

        async function setOverride(zone, appliance, mode) {
            await fetch('/api/override', {
                method: 'POST',
                headers: {'Content-Type': 'application/json'},
                body: JSON.stringify({zone, appliance, mode})
            });
        }

        setInterval(async () => {
            let res = await fetch('/api/stats');
            let data = await res.json();
            
            document.getElementById('person-count').innerText = data.person_count;
            document.getElementById('kwh').innerHTML = data.kwh_saved + ' <span style="font-size:12px">kWh</span>';
            document.getElementById('rs').innerText = '₹' + data.rupees_saved;
            document.getElementById('temp').innerText = data.temp + ' °C';
            document.getElementById('hum').innerText = data.humidity + ' %';
            document.getElementById('cutoff-display').innerText = data.temp_cutoff + ' °C';
            
            updateSecurityUI(data.security_mode);
            playSirenSound(data.alarm_active);

            let alertsHtml = data.alerts.map(a => `<div class="alert-item"><span>${a.type}</span> <span>${a.time}</span></div>`).join('');
            document.getElementById('alerts').innerHTML = alertsHtml || '<div style="color:#444; font-size:12px; text-align:center; padding: 10px 0;">No active alerts.</div>';

            let timelineRes = await fetch('/api/timeline_data');
            let tData = await timelineRes.json();
            timelineChart.data.labels = tData.labels;
            timelineChart.data.datasets[0].data = tData.z1;
            timelineChart.data.datasets[1].data = tData.z2;
            timelineChart.data.datasets[2].data = tData.z3;
            timelineChart.data.datasets[3].data = tData.z4;
            timelineChart.update();
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
    manual_overrides[data['zone']][data['appliance']] = data['mode']
    return jsonify({"status": "success"})

@app.route('/api/temp_cutoff', methods=['POST'])
@login_required
def set_temp_cutoff():
    global temp_cutoff
    data = request.json
    temp_cutoff = float(data.get('temp_cutoff', 35.0))
    return jsonify({"status": "success", "temp_cutoff": temp_cutoff})

@app.route('/api/security_mode', methods=['POST'])
@login_required
def toggle_security_mode():
    global security_mode
    security_mode = not security_mode
    return jsonify({"status": "success", "security_mode": security_mode})

@app.route('/api/alarm', methods=['POST'])
@login_required
def control_alarm():
    global alarm_active
    data = request.json
    enable = data.get("enable", False)
    alarm_active = enable
    state_str = "on" if enable else "off"
    executor.submit(async_send, f"{ESP_IP}/alarm/{state_str}")
    return jsonify({"status": "success", "alarm_active": alarm_active})

@app.route('/api/stats')
@login_required
def get_stats():
    kwh = (total_idle_seconds / 3600.0) * POWER_RATING_KW
    return jsonify({
        "person_count": current_person_count,
        "security_mode": security_mode,
        "alarm_active": alarm_active,
        "temp_cutoff": temp_cutoff,
        "kwh_saved": round(kwh, 4),
        "rupees_saved": round(kwh * TARIFF_PER_KWH, 2),
        "temp": sensor_data.get("temp", "--"),
        "humidity": sensor_data.get("humidity", "--"),
        "alerts": service_alerts[-4:]
    })

@app.route('/api/timeline_data')
@login_required
def get_timeline_data():
    labels = [item["time"] for item in timeline_history]
    z1 = [item["z1"] for item in timeline_history]
    z2 = [item["z2"] for item in timeline_history]
    z3 = [item["z3"] for item in timeline_history]
    z4 = [item["z4"] for item in timeline_history]
    return jsonify({"labels": labels, "z1": z1, "z2": z2, "z3": z3, "z4": z4})

if __name__ == '__main__':
    app.run(host='0.0.0.0', port=5000, debug=False, threaded=True)
