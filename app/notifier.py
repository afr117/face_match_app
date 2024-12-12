 
import cv2
import os

def send_notification(name, frame):
    print(f"Sending notification: Match found for {name}.")
    # Save the frame locally for reference
    if not os.path.exists("alerts"):
        os.makedirs("alerts")
    file_path = f"alerts/{name}_alert.jpg"
    cv2.imwrite(file_path, frame)
    print(f"Alert image saved to {file_path}.")
