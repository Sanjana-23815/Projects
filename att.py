import streamlit as st
import cv2
import os
import numpy as np
import pandas as pd
from datetime import datetime

# Constants
FACE_DIR = "opencv_faces"
ATTENDANCE_FILE = "attendance.csv"
os.makedirs(FACE_DIR, exist_ok=True)

# Load recognizer and face cascade
recognizer = cv2.face.LBPHFaceRecognizer_create()
face_cascade = cv2.CascadeClassifier(cv2.data.haarcascades + "haarcascade_frontalface_default.xml")

# Save attendance
def save_attendance(name):
    now = datetime.now()
    dt_string = now.strftime('%Y-%m-%d %H:%M:%S')
    if not os.path.exists(ATTENDANCE_FILE):
        df = pd.DataFrame(columns=["Name", "DateTime"])
    else:
        df = pd.read_csv(ATTENDANCE_FILE)

    if name not in df['Name'].values:
        df = pd.concat([df, pd.DataFrame([{"Name": name, "DateTime": dt_string}])], ignore_index=True)
        df.to_csv(ATTENDANCE_FILE, index=False)
        st.success(f"✅ {name} marked present at {dt_string}")
    else:
        st.info(f"ℹ️ {name} already marked present.")

# Clear attendance
def clear_attendance():
    if os.path.exists(ATTENDANCE_FILE):
        os.remove(ATTENDANCE_FILE)
        st.success("✅ Attendance cleared.")
    else:
        st.info("📭 No attendance to clear.")

# Capture faces
def capture_faces(name, roll_no):
    path = os.path.join(FACE_DIR, f"{name}_{roll_no}")
    os.makedirs(path, exist_ok=True)
    cam = cv2.VideoCapture(0)
    count = 0
    stframe = st.empty()

    with st.spinner("Capturing face images..."):
        while count < 20:
            ret, frame = cam.read()
            if not ret:
                continue
            rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
            faces = face_cascade.detectMultiScale(gray, 1.3, 5)

            for (x, y, w, h) in faces:
                face_img = gray[y:y+h, x:x+w]
                cv2.imwrite(f"{path}/{count}.jpg", face_img)
                count += 1
                stframe.image(rgb_frame, caption=f"Image {count}/20", use_container_width=True)
                break
    cam.release()
    st.success(f"✅ Face captured for {name} ({roll_no})")

# Train model
def train_model():
    faces = []
    labels = []
    label_map = {}
    i = 0

    for person in os.listdir(FACE_DIR):
        person_path = os.path.join(FACE_DIR, person)
        for image_name in os.listdir(person_path):
            img_path = os.path.join(person_path, image_name)
            img = cv2.imread(img_path, cv2.IMREAD_GRAYSCALE)
            faces.append(img)
            labels.append(i)
        label_map[i] = person
        i += 1

    if faces:
        recognizer.train(faces, np.array(labels))
        return label_map
    else:
        return None

# Recognize face
def recognize_face():
    label_map = train_model()
    if label_map is None:
        st.warning("⚠️ No faces registered yet.")
        return

    cam = cv2.VideoCapture(0)
    stframe = st.empty()
    stop_btn = st.button("🛑 Stop Camera")

    while cam.isOpened():
        if stop_btn:
            break

        ret, frame = cam.read()
        if not ret:
            break

        rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        faces = face_cascade.detectMultiScale(gray, 1.3, 5)

        for (x, y, w, h) in faces:
            face_img = gray[y:y+h, x:x+w]
            label, confidence = recognizer.predict(face_img)
            if confidence < 70:
                name = label_map[label]
                save_attendance(name)
                cv2.rectangle(rgb_frame, (x, y), (x+w, y+h), (0, 255, 0), 2)
                cv2.putText(rgb_frame, name, (x, y-10),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.9, (0, 255, 0), 2)

        stframe.image(rgb_frame, channels="RGB")

    cam.release()

# Attendance summary
def show_summary():
    if os.path.exists(ATTENDANCE_FILE):
        df = pd.read_csv(ATTENDANCE_FILE)
        unique_names = df['Name'].nunique()
        total_registered = len(os.listdir(FACE_DIR))
        percent_present = (unique_names / total_registered) * 100 if total_registered else 0

        col1, col2, col3 = st.columns(3)
        col1.metric("👥 Total Registered", total_registered)
        col2.metric("✅ Present Today", unique_names)
        col3.metric("📊 Attendance %", f"{percent_present:.2f}%")
    else:
        st.info("📭 No attendance records found.")

# Streamlit UI
st.set_page_config(page_title="Biometric Attendance", layout="wide")
st.markdown("""
    <style>
        body {
            background: linear-gradient(to right, #fceabb, #f8b500);
            color: #2e2e2e;
            font-family: 'Segoe UI', sans-serif;
        }
        .stButton>button {
            background-color: #ff8c42;
            color: white;
            font-weight: 600;
            border-radius: 10px;
            padding: 0.5em 1.5em;
            transition: background-color 0.3s;
        }
        .stButton>button:hover {
            background-color: #e67300;
        }
        .stTextInput>div>div>input {
            background-color: #fff8ee;
            border: 1px solid #ffae42;
            border-radius: 10px;
            color: #333;
            padding: 10px;
        }
        .stRadio>div>label {
            color: #333;
            font-weight: 500;
        }
    </style>
    <h1 style='text-align: center; color: #ff8c42;'>✨ Advanced Biometric Attendance System</h1>
""", unsafe_allow_html=True)

with st.sidebar:
    st.markdown("## 🧭 Navigation")
    option = st.radio("Choose an action:", [
        "Register Face", "Mark Attendance", "View Attendance", "Show Summary", "Clear Attendance"])

if option == "Register Face":
    st.subheader("📸 Register Face")
    name = st.text_input("Enter Name")
    roll = st.text_input("Enter Roll Number")
    if st.button("📥 Capture Face"):
        if name and roll:
            capture_faces(name, roll)
        else:
            st.warning("⚠️ Enter both name and roll number!")

elif option == "Mark Attendance":
    st.subheader("📍 Real-time Recognition")
    recognize_face()

elif option == "View Attendance":
    st.subheader("📄 Attendance Records")
    if os.path.exists(ATTENDANCE_FILE):
        df = pd.read_csv(ATTENDANCE_FILE)
        df['Name'] = df['Name'].astype(str)
        st.dataframe(df[['Name', 'DateTime']], use_container_width=True)
    else:
        st.info("📭 No attendance yet.")

elif option == "Show Summary":
    st.subheader("📊 Attendance Summary")
    show_summary()

elif option == "Clear Attendance":
    st.subheader("🧹 Clear Attendance")
    if st.button("🗑️ Clear Now"):
        clear_attendance()