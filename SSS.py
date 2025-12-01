import face_recognition
import cv2
import numpy as np
import os
from tkinter import simpledialog, messagebox, filedialog
import datetime
import sys
import keyboard
import json

video_capture = cv2.VideoCapture(0)

known_face_encodings = []
known_face_names = []

for file in os.listdir("./faces"):
    if file.lower().endswith((".jpg", ".jpeg", ".png")):
        path = os.path.join("./faces", file)

        image = face_recognition.load_image_file(path, mode='RGB')
        encodings = face_recognition.face_encodings(image)

        known_face_encodings.append(encodings[0])
        known_face_names.append(os.path.splitext(file)[0])

face_locations = []
face_encodings = []
face_names = []

process_this_frame = True

period_times = [datetime.time(8, 0), datetime.time(8, 50), datetime.time(9, 45), datetime.time(10, 50), datetime.time(11, 40), datetime.time(12, 35), datetime.time(13, 25), datetime.time(14, 15)]
days_pl = ["Poniedziałek", "Wtorek", "Środa", "Czwartek", "Piątek", "Sobota", "Niedziela"]

current_datetime = None
current_time = None
current_period_idx = 0

present_threshold = 0

present = []
late = []
absent = []

folder_path = ""

with open("config.json") as f:
    config = json.load(f)

present_threshold = config["present_threshold"]
folder_path = os.path.expanduser(config["path"])

def confirm_face():
    global face_names

    known_face = True

    if len(face_names) >= 5:
        face_names = face_names[-5:]

        if len(set(face_names)) == 1:
            if face_names[0] == "Nieznana twarz":
                ret, frame = video_capture.read()
                if messagebox.askyesno("Nieznana twarz", "Nie wykryto twarzy. Być może nie istnieje ona w bazie danych. Czy dodać twarz do bazy?"):
                    name = simpledialog.askstring(" ", "Podaj imię i nazwisko:")

                    for (top, right, bottom, left) in face_recognition.face_locations(rgb_small_frame):
                        top *= 4
                        right *= 4
                        bottom *= 4
                        left *= 4

                        top = max(0, top - 100)
                        right = min(frame.shape[1], right + 30)
                        bottom = min(frame.shape[0], bottom + 40)
                        left = max(0, left - 30)

                        face = frame[top:bottom, left:right]

                    cv2.imwrite(f"faces/{name}.jpg", face)

                    known_face_names.append(name)
                    known_face_encodings.append(face_encodings[0])
                    face_names[0] = name
                else:
                    known_face = False

            if known_face:
                manage_time()

                if face_names[0] not in present and face_names[0] not in late:
                    current_time_formatted = datetime.timedelta(hours=current_time.hour, minutes=current_time.minute)
                    period_time = datetime.timedelta(hours=period_times[current_period_idx].hour, minutes=period_times[current_period_idx].minute)

                    if current_time_formatted - period_time <= datetime.timedelta(minutes=present_threshold):
                        present.append(face_names[0])
                    elif current_time_formatted - period_time <= datetime.timedelta(minutes=15):
                        late.append(face_names[0])
                    else:
                        absent.append(face_names[0])

                os.makedirs(folder_path, exist_ok=True)
                with open(f"{folder_path}/{current_datetime.strftime('%d.%m')} ({days_pl[current_datetime.weekday()]}) - Lekcja {current_period_idx + 1} - Klasa {grade}.txt", "w") as file:
                    file.write("Obecni:\n\n" + "\n".join(present) + "\n\n\nSpóźnienia:\n\n" + "\n".join(late) + "\n\n\nNieobecni:\n\n" + "\n".join(absent))

            face_names.clear()

def manage_time():
    global current_datetime
    global current_time
    global current_period_idx

    current_datetime = datetime.datetime.now()
    current_time = current_datetime.time()
    current_period_idx = -1

    for i in range(len(period_times)):
        if current_time < period_times[i]:
            current_period_idx = i - 1
            break

grade = simpledialog.askstring(" ", "Podaj nazwę klasy:")

if grade in ("", None):
    sys.exit()

while True:
    ret, frame = video_capture.read()
    if not ret:
        continue

    if process_this_frame:
        small_frame = cv2.resize(frame, None, fx=0.25, fy=0.25)

        rgb_small_frame = cv2.cvtColor(small_frame, cv2.COLOR_BGR2RGB)

        face_locations = face_recognition.face_locations(rgb_small_frame)
        face_encodings = face_recognition.face_encodings(rgb_small_frame, face_locations)

        face_encodings_len = len(face_encodings)

        if face_encodings_len == 1:
            matches = face_recognition.compare_faces(known_face_encodings, face_encodings[0], 0.5)
            face_distances = face_recognition.face_distance(known_face_encodings, face_encodings[0])

            best_match_index = np.argmin(face_distances)
            name = "Nieznana twarz"

            if matches[best_match_index]:
                name = known_face_names[best_match_index]

            face_names.append(name)

    process_this_frame = not process_this_frame


    if face_locations and face_names and face_encodings_len == 1:
        (top, right, bottom, left), name = face_locations[-1], face_names[-1]

        top *= 4
        right *= 4
        bottom *= 4
        left *= 4

        cv2.rectangle(frame, (left, top), (right, bottom), (0, 0, 255), 2)
        cv2.rectangle(frame, (left, bottom), (right, bottom - 35), (0, 0, 255), cv2.FILLED)
        cv2.putText(frame, name, (left + 6, bottom - 6), cv2.FONT_HERSHEY_DUPLEX, 1.0, (255, 255, 255), 1)

        confirm_face()
    elif face_encodings_len > 1:
        cv2.putText(frame, "WYKRYTO WIECEJ NIZ JEDNA TWARZ", (30, 450), cv2.FONT_HERSHEY_DUPLEX, 1.0, (0, 0, 255), 2)

    cv2.imshow('Students Surveillance System', frame)

    key = cv2.waitKey(1)

    if keyboard.is_pressed('alt+p'):
        with open("config.json", "w") as f:
            folder_path = filedialog.askdirectory(initialdir=os.path.expanduser("~/Documents")) + "/SSS - Frekwencja"

            config["path"] = folder_path
            
            with open("config.json", "w") as f:
                json.dump(config, f, indent=4)

    if keyboard.is_pressed('alt+c'):
        new_grade = simpledialog.askstring(" ", "Podaj nazwę klasy:")

        if new_grade not in ("", None):
            grade = new_grade

    if keyboard.is_pressed('alt+t'):
        new_present_threshold = simpledialog.askinteger(" ", "Maksymalny czas uznania obecności (w minutach): ")
        
        if new_present_threshold is not None:
            present_threshold = new_present_threshold

            config["present_threshold"] = new_present_threshold

            with open("config.json", "w") as f:
                json.dump(config, f, indent=4)

        present = []
        late = []
        absent = []

        manage_time()

    if cv2.getWindowProperty('Students Surveillance System', cv2.WND_PROP_VISIBLE) == 0:
        break

video_capture.release()
cv2.destroyAllWindows()