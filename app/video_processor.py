 import cv2
import face_recognition
import numpy as np

def process_video_feed(video_url, known_faces, notify_callback):
    video_capture = cv2.VideoCapture(video_url)

    # Load known faces and encodings
    known_face_encodings = []
    known_face_names = []

    for name, image_path in known_faces.items():
        image = face_recognition.load_image_file(image_path)
        encoding = face_recognition.face_encodings(image)[0]
        known_face_encodings.append(encoding)
        known_face_names.append(name)

    while video_capture.isOpened():
        ret, frame = video_capture.read()
        if not ret:
            print("Failed to grab frame. Exiting...")
            break

        # Resize frame for faster processing
        small_frame = cv2.resize(frame, (0, 0), fx=0.25, fy=0.25)
        rgb_frame = small_frame[:, :, ::-1]

        # Detect faces
        face_locations = face_recognition.face_locations(rgb_frame)
        face_encodings = face_recognition.face_encodings(rgb_frame, face_locations)

        for face_encoding, face_location in zip(face_encodings, face_locations):
            matches = face_recognition.compare_faces(known_face_encodings, face_encoding)
            face_distances = face_recognition.face_distance(known_face_encodings, face_encoding)
            best_match_index = np.argmin(face_distances)

            if matches[best_match_index]:
                name = known_face_names[best_match_index]
                print(f"Match found: {name}")
                notify_callback(name, frame)

        # Display the frame (optional, for debugging)
        cv2.imshow("Video Feed", frame)
        if cv2.waitKey(1) & 0xFF == ord("q"):
            break

    video_capture.release()
    cv2.destroyAllWindows()
