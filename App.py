import threading
import time
import socket
import pyttsx3
import speech_recognition as sr
import RPi.GPIO as GPIO
import numpy as np
import tensorflow as tf
import cv2
from picamera2 import Picamera2
from flask import Flask, Response, render_template_string, request, jsonify
from queue import Queue
import Adafruit_DHT

# ==========================================================
# MOTOR SETUP
# ==========================================================
IN1, IN2, IN3, IN4 = 17, 18, 22, 23
GPIO.setmode(GPIO.BCM)
GPIO.setwarnings(False)
for pin in [IN1, IN2, IN3, IN4]:
    GPIO.setup(pin, GPIO.OUT)

def stop(): GPIO.output([IN1, IN2, IN3, IN4],[0,0,0,0])
def forward(): GPIO.output([IN1,IN2,IN3,IN4],[1,0,1,0])
def backward(): GPIO.output([IN1,IN2,IN3,IN4],[0,1,0,1])
def left(): GPIO.output([IN1,IN2,IN3,IN4],[0,0,1,0])
def right(): GPIO.output([IN1,IN2,IN3,IN4],[1,0,0,0])

# ==========================================================
# SENSOR SETUP
# ==========================================================
TRIG, ECHO = 5, 6
GPIO.setup(TRIG, GPIO.OUT)
GPIO.setup(ECHO, GPIO.IN)

DHT_SENSOR = Adafruit_DHT.DHT11
DHT_PIN = 4

PIR_PIN = 27
GPIO.setup(PIR_PIN, GPIO.IN)

# ==========================================================
# TEXT-TO-SPEECH
# ==========================================================
tts_engine = pyttsx3.init()
tts_engine.setProperty("rate", 220)
tts_engine.setProperty("volume", 1.0)
tts_queue = Queue()

def speak_worker():
    while True:
        msg = tts_queue.get()
        tts_engine.say(msg)
        tts_engine.runAndWait()
        tts_queue.task_done()

def speak(msg): tts_queue.put(msg)

# ==========================================================
# SPEECH-TO-TEXT
# ==========================================================
recognizer = sr.Recognizer()
mic = sr.Microphone()
speech_text = ""

def thread_speech_to_text():
    global speech_text
    with mic as source:
        recognizer.adjust_for_ambient_noise(source)
    while True:
        try:
            with mic as source:
                audio = recognizer.listen(source, timeout=3, phrase_time_limit=4)
            try:
                speech_text = recognizer.recognize_google(audio)
            except:
                speech_text = ""
        except Exception as e:
            print("STT Error:", e)
        time.sleep(0.2)

# ==========================================================
# CAMERA + BODY PART DETECTION
# ==========================================================
MODEL = tf.keras.models.load_model("bodyparts_classification_model_final.keras")
CLASS_LABELS = ['Belly','Ear','Elbow','Eye','Foot','Hand','Knee','Neck','Nose','Shoulders']

picam2 = Picamera2()
picam2.configure(picam2.create_preview_configuration(main={"format": "RGB888", "size": (224,224)}))
picam2.start()

bodypart_label = ""

def generate_frames():
    global bodypart_label
    while True:
        frame = picam2.capture_array()
        img = cv2.resize(frame,(224,224))
        img_input = tf.keras.applications.efficientnet.preprocess_input(img)
        img_input = np.expand_dims(img_input,0)
        pred = MODEL.predict(img_input,verbose=0)
        bodypart_label = CLASS_LABELS[np.argmax(pred)]
        # Draw single label box
        cv2.rectangle(frame,(5,5),(220,50),(0,255,0),-1)
        cv2.putText(frame, bodypart_label, (10,35), cv2.FONT_HERSHEY_SIMPLEX, 1, (0,0,0),2)
        ret, buffer = cv2.imencode('.jpg', frame)
        frame_bytes = buffer.tobytes()
        yield (b"--frame\r\nContent-Type: image/jpeg\r\n\r\n" + frame_bytes + b"\r\n")

# ==========================================================
# SENSOR THREAD
# ==========================================================
sensor_data = {"ultrasonic":0,"temperature":0,"humidity":0,"pir":0}

def thread_sensors():
    global sensor_data
    while True:
        # Ultrasonic
        GPIO.output(TRIG, False)
        time.sleep(0.05)
        GPIO.output(TRIG, True)
        time.sleep(0.00001)
        GPIO.output(TRIG, False)
        pulse_start, pulse_end = time.time(), time.time()
        while GPIO.input(ECHO)==0: pulse_start = time.time()
        while GPIO.input(ECHO)==1: pulse_end = time.time()
        distance = (pulse_end - pulse_start) * 17150
        sensor_data["ultrasonic"] = round(distance,1)
        # DHT11
        humidity, temperature = Adafruit_DHT.read(DHT_SENSOR,DHT_PIN)
        if humidity and temperature:
            sensor_data["humidity"] = round(humidity,1)
            sensor_data["temperature"] = round(temperature,1)
        # PIR
        sensor_data["pir"] = GPIO.input(PIR_PIN)
        time.sleep(1)

# ==========================================================
# FLASK APP
# ==========================================================
html_page = """
<!DOCTYPE html>
<html>
<head>
<title>Robot Dashboard</title>
<style>
body{font-family:Arial;text-align:center;background:#f2f2f2;}
button{padding:12px 20px;margin:5px;font-size:18px;border:none;border-radius:8px;cursor:pointer;}
#motor button{background:#28a745;color:white;}
#motor #stopBtn{background:#dc3545;}
#videoContainer,#sensorContainer,#speechBox,#ttsBox{margin-top:20px;}
#stream{width:80%;border:3px solid #444;border-radius:10px;}
textarea,input{width:60%;padding:10px;margin:5px;font-size:18px;}
</style>
</head>
<body>
<h1>🤖 Robot Dashboard</h1>

<div id="motor">
<button onclick="sendCmd('forward')">⬆ Forward</button>
<button onclick="sendCmd('backward')">⬇ Backward</button>
<button onclick="sendCmd('left')">⬅ Left</button>
<button onclick="sendCmd('right')">➡ Right</button>
<button id="stopBtn" onclick="sendCmd('stop')">🛑 Stop</button>
</div>

<div id="videoContainer">
<h2>Camera Feed</h2>
<img id="stream" src="/stream">
<h3>Detected Body Part: <span id="bodypart">-</span></h3>
</div>

<div id="sensorContainer">
<h2>Sensor Data</h2>
<p>Ultrasonic: <span id="ultra">0</span> cm</p>
<p>Temperature: <span id="temp">0</span>°C</p>
<p>Humidity: <span id="hum">0</span>%</p>
<p>PIR Motion: <span id="pir">0</span></p>
</div>

<div id="speechBox">
<h3>Speech-to-Text</h3>
<textarea id="speech" rows="2" readonly></textarea>
</div>

<div id="ttsBox">
<h3>Text-to-Speech</h3>
<input type="text" id="tts" placeholder="Enter text..." />
<button onclick="speakText()">Speak</button>
</div>

<script>
function sendCmd(cmd){fetch('/motor',{method:'POST',headers:{'Content-Type':'application/json'},body:JSON.stringify({command:cmd})});}
function speakText(){const txt=document.getElementById('tts').value;fetch('/tts',{method:'POST',headers:{'Content-Type':'application/json'},body:JSON.stringify({msg:txt})});}

async function updateData(){
  const resp=await fetch('/stt'); const data=await resp.json();
  document.getElementById('speech').value=data.text;
  document.getElementById('bodypart').textContent=data.bodypart;
  document.getElementById('ultra').textContent=data.ultrasonic;
  document.getElementById('temp').textContent=data.temperature;
  document.getElementById('hum').textContent=data.humidity;
  document.getElementById('pir').textContent=data.pir;
}
setInterval(updateData,500);
</script>

</body>
</html>
"""

app = Flask(__name__)
@app.route("/") 
def index(): 
    return render_template_string(html_page)

@app.route("/stream") 
def stream(): 
    return Response(generate_frames(), mimetype="multipart/x-mixed-replace; boundary=frame")

@app.route("/motor", methods=["POST"])
def motor():
    cmd = request.get_json().get("command")
    if cmd=="forward": forward()
    elif cmd=="backward": backward()
    elif cmd=="left": left()
    elif cmd=="right": right()
    elif cmd=="stop": stop()
    return jsonify({"status":"ok"})

@app.route("/tts", methods=["POST"])
def tts():
    msg = request.get_json().get("msg","")
    if msg: speak(msg)
    return jsonify({"status":"ok"})

@app.route("/stt")
def stt(): 
    global speech_text, bodypart_label, sensor_data
    return jsonify({"text":speech_text,"bodypart":bodypart_label,
                    "ultrasonic":sensor_data["ultrasonic"],
                    "temperature":sensor_data["temperature"],
                    "humidity":sensor_data["humidity"],
                    "pir":sensor_data["pir"]})

# ==========================================================
# NETWORK UTILITIES
# ==========================================================
def get_pi_ip():
    try:
        s = socket.socket(socket.AF_INET,socket.SOCK_DGRAM)
        s.connect(("8.8.8.8",80))
        ip = s.getsockname()[0]; s.close(); return ip
    except: return "0.0.0.0"

def run_flask():
    ip = get_pi_ip()
    port = 5000
    print(f"🌐 Raspberry Pi Dashboard: http://{ip}:{port}")
    speak(f"My IP address is {ip.replace('.', ' dot ')}")
    app.run(host="0.0.0.0", port=port, debug=False, threaded=True)

# ==========================================================
# START THREADS
# ==========================================================
threads = [
    threading.Thread(target=speak_worker, daemon=True),
    threading.Thread(target=thread_speech_to_text, daemon=True),
    threading.Thread(target=thread_sensors, daemon=True),
    threading.Thread(target=run_flask, daemon=True)
]

for t in threads: t.start()

try:
    while True: time.sleep(1)
except KeyboardInterrupt:
    stop(); GPIO.cleanup(); picam2.close(); print("Exiting...")
