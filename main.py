import cv2
import numpy as np

# Initialize the camera streams (IP camera URLs)
CAMERA_IPS = [
    "rtsp://username:password@192.168.0.62:554/stream1"  # Replace with your RTSP credentials and IP
]

# Process live video stream
def process_camera(ip):
    cap = cv2.VideoCapture(ip)  # Adjust stream path if needed
    if not cap.isOpened():
        print(f"Error: Unable to access the camera stream: {ip}")
        return

    while True:
        ret, frame = cap.read()
        if not ret:
            break

        # Process frame (example: display the video feed)
        cv2.imshow(f"Camera Feed {ip}", frame)
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

    cap.release()
    cv2.destroyAllWindows()

# Main function
def main():
    for ip in CAMERA_IPS:
        process_camera(ip)

if __name__ == "__main__":
    main()
