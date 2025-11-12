import face_recognition
import cv2
import numpy as np
import os

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
process_this_frame = False

face_names = []

while True:
    ret, frame = video_capture.read()

    if ret and frame is not None:
        process_this_frame = True

    if process_this_frame:
        small_frame = cv2.resize(frame, (0, 0), fx=0.25, fy=0.25)

        rgb_small_frame = np.ascontiguousarray(small_frame)
        
        face_locations = face_recognition.face_locations(rgb_small_frame)
        face_encodings = face_recognition.face_encodings(rgb_small_frame, face_locations)

        for face_encoding in face_encodings:
            matches = face_recognition.compare_faces(known_face_encodings, face_encoding)
            name = "Unknown"

            face_distances = face_recognition.face_distance(known_face_encodings, face_encoding)
            best_match_index = np.argmin(face_distances)
            if matches[best_match_index]:
                name = known_face_names[best_match_index]

            face_names.append(name)

            if len(face_names) >= 5 and len(set(face_names[-5:])) == 1:
                print("Ostatnie 5 zarejestrowany twarzy to: " + face_names[-1])

    process_this_frame = not process_this_frame

    if face_locations and face_names:
        (top, right, bottom, left), name = face_locations[-1], face_names[-1]

        top *= 4
        right *= 4
        bottom *= 4
        left *= 4

        cv2.rectangle(frame, (left, top), (right, bottom), (0, 0, 255), 2)
        cv2.rectangle(frame, (left, bottom), (right, bottom - 35), (0, 0, 255), cv2.FILLED)
        cv2.putText(frame, name, (left + 6, bottom - 6), cv2.FONT_HERSHEY_DUPLEX, 1.0, (255, 255, 255), 1)

    cv2.imshow('Students Surveillance System', frame)

    if cv2.waitKey(1) and cv2.getWindowProperty('Students Surveillance System', cv2.WND_PROP_VISIBLE) == 0:
        break

video_capture.release()
cv2.destroyAllWindows()