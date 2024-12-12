import cv2
import face_recognition
import numpy as np
import os
import requests

KNOWN_FACES_DIR = "known_faces"
LOG_FILE = "notifications.log"
CAMERA_URLS = [
    "http://public-camera1-url",
    "http://public-camera2-url"
]

def load_known_faces():
    known_face_encodings = []
    known_face_names = []
    for filename in os.listdir(KNOWN_FACES_DIR):
        if filename.endswith(".jpg") or filename.endswith(".png"):
            filepath = os.path.join(KNOWN_FACES_DIR, filename)
            image = face_recognition.load_image_file(filepath)
            encodings = face_recognition.face_encodings(image)
            if encodings:  # Ensure at least one face encoding exists
                known_face_encodings.append(encodings[0])
                known_face_names.append(os.path.splitext(filename)[0])
            else:
                print(f"Warning: No face found in {filename}")
    return known_face_encodings, known_face_names

def notify_match(name):
    with open(LOG_FILE, "a") as log_file:
        log_file.write(f"Match found: {name}\n")
    print(f"Notification sent: Match found for {name}")

def process_camera(url, known_face_encodings, known_face_names):
    cap = cv2.VideoCapture(url)
    if not cap.isOpened():
        print(f"Error: Unable to access the camera stream: {url}")
        return

    try:
        while cap.isOpened():
            ret, frame = cap.read()
            if not ret:
                print(f"Error: Unable to read frame from: {url}")
                break

            rgb_frame = frame[:, :, ::-1]
            face_locations = face_recognition.face_locations(rgb_frame)
            face_encodings = face_recognition.face_encodings(rgb_frame, face_locations)

            for face_encoding, face_location in zip(face_encodings, face_locations):
                matches = face_recognition.compare_faces(known_face_encodings, face_encoding)
                name = "Unknown"

                if True in matches:
                    first_match_index = matches.index(True)
                    name = known_face_names[first_match_index]
                    notify_match(name)

                top, right, bottom, left = face_location
                cv2.rectangle(frame, (left, top), (right, bottom), (0, 255, 0), 2)
                cv2.putText(frame, name, (left, top - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 2)

            cv2.imshow(f"Live Feed - {url}", frame)
            if cv2.waitKey(1) & 0xFF == ord('q'):
                print(f"Stopping stream: {url}")
                break
    finally:
        cap.release()

def main():
    known_face_encodings, known_face_names = load_known_faces()
    print("Loaded known faces:", known_face_names)

    for url in CAMERA_URLS:
        process_camera(url, known_face_encodings, known_face_names)

    cv2.destroyAllWindows()

if __name__ == "__main__":
    main()
