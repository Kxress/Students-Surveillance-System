import cv2
from deepface import DeepFace
import os

KNOWN_FACES_DIR = "./faces"
frame_count = 0
label = "Initialization..."
color = (255, 255, 0)
face_location = None

video_capture = cv2.VideoCapture(0)

while True:
    ret, frame = video_capture.read()
    
    frame_count += 1
    
    if frame_count % 30 == 0:
        try:
            # First detect face location
            face_objs = DeepFace.extract_faces(img_path=frame, detector_backend='opencv', enforce_detection=False)
            
            if face_objs and len(face_objs) > 0:
                # Get the facial area (x, y, w, h)
                facial_area = face_objs[0]['facial_area']
                face_location = (facial_area['x'], facial_area['y'], facial_area['w'], facial_area['h'])
                
                # Now try to find matching face
                results = DeepFace.find(img_path=frame, db_path=KNOWN_FACES_DIR, enforce_detection=False, silent=True, model_name='VGG-Face', detector_backend='opencv')
                
                # Check if any faces were found
                if results and len(results) > 0 and len(results[0]) > 0:
                    # Get the first result DataFrame
                    first_result = results[0]
                    
                    # Get the best match (first row)
                    match_path = first_result.iloc[0]['identity']
                    distance = first_result.iloc[0]['distance']
                    
                    # Extract name from path
                    name = os.path.splitext(os.path.basename(match_path))[0]
                    
                    label = f"{name} ({distance:.2f})"
                    color = (0, 255, 0)
                else:
                    label = "Nieznana twarz"
                    color = (0, 0, 255)
            else:
                label = "Brak twarzy"
                color = (0, 0, 255)
                face_location = None
                
        except Exception as e:
            label = f"Blad: {str(e)[:30]}"
            color = (0, 0, 255)
            face_location = None
    
    # Draw rectangle around face if detected
    if face_location:
        x, y, w, h = face_location
        cv2.rectangle(frame, (x, y), (x + w, y + h), color, 2)
        
        # Draw label below the rectangle
        label_y = y + h + 20
        
        # Draw background for text (for better visibility)
        (text_width, text_height), baseline = cv2.getTextSize(label, cv2.FONT_HERSHEY_SIMPLEX, 0.7, 2)
        cv2.rectangle(frame, (x - 1, label_y - text_height - 5), (x + text_width + 10, label_y + baseline), color, -1)
        
        # Draw text
        cv2.putText(frame, label, (x + 5, label_y), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2)
    
    # Draw frame counter in corner
    cv2.putText(frame, f"Frame: {frame_count}", (30, frame.shape[0] - 20), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 1)
    
    cv2.imshow("Students Surveillance System", frame)
    
    # Check for window close
    cv2.waitKey(1)
    if cv2.getWindowProperty("Students Surveillance System", cv2.WND_PROP_VISIBLE) < 1:
        break

video_capture.release()
cv2.destroyAllWindows()
print("Program terminated")