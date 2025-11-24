import face_recognition
import cv2
import numpy as np
import os
import messagebox

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
rect_color = (0, 255, 255)

def confirm_face():
    global face_names

    if len(face_names) >= 5:
        face_names = face_names[-5:]

        if len(set(face_names)) == 1:
            if face_names[0] != "Unknown":
                if messagebox.askyesno("Wykryto twarz", f"Wykryto twarz: {face_names[0]}. Czy twarz się zgadza?"):
                    messagebox.showinfo("Obecność zatwierdzona", "Zatwierdzono obecność! Kliknij OK by zeskanować następną twarz.")
            else:
                messagebox.askyesno("Nieznana twarz", "Nie wykryto twarzy. Być może nie istnieje ona w bazie danych. Czy dodać twarz do bazy?")

            face_names.clear()

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
            matches = face_recognition.compare_faces(known_face_encodings, face_encodings[0])
            face_distances = face_recognition.face_distance(known_face_encodings, face_encodings[0])

            best_match_index = np.argmin(face_distances)
            name = "Unknown"

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

    if cv2.waitKey(1) and cv2.getWindowProperty('Students Surveillance System', cv2.WND_PROP_VISIBLE) == 0:
        break

video_capture.release()
cv2.destroyAllWindows()