Camera Monitoring System with Face Recognition
A complete camera monitoring application that uses your webcam to detect and recognize faces in real-time. Built with Python and Streamlit for easy use.

📋 Table of Contents
What Does This App Do?
Features
How to Install and Run Locally
How to Use the App - Page by Page Guide
Understanding the Code - File by File
Data Storage
Troubleshooting
🎯 What Does This App Do?
This is a security monitoring system that:

Connects to your webcam (or IP cameras)
Detects faces in the video feed
Recognizes people by comparing faces to saved photos
Shows real-time results with percentage match scores
Sends alerts when it recognizes someone (optional email notifications)
Perfect for:

Home security monitoring
Office access monitoring
Learning about computer vision and face recognition
Building your own custom security system
✨ Features
🎥 Live Video Streaming
Real-time webcam feed at 15-30 FPS (frames per second)
Live face detection with colored boxes:
Green box = Recognized person (match found)
Red box = Unknown person (no match)
Large, easy-to-read percentage displays showing match confidence
🧠 Face Recognition
Uses OpenCV (a computer vision library) for face detection
Template matching to identify people
Adjustable confidence threshold (how strict the matching is)
Saves reference photos of people you want to recognize
📸 Camera Management
Add multiple cameras (webcam or IP cameras)
View camera feeds as snapshots
Test camera connections
Auto-refresh option for live snapshot updates
🔔 Smart Notifications
Email alerts when faces are recognized
Debouncing - won't spam you (sends max 1 alert per minute per person)
Customizable email settings
⚙️ Settings & Controls
Adjust face recognition sensitivity
Configure email notifications
Manage reference faces
View detection history
🚀 How to Install and Run Locally
Prerequisites (What You Need)
Python 3.10 or higher installed on your computer

Download from: https://www.python.org/downloads/
During installation, check "Add Python to PATH"
A webcam (built-in laptop camera or USB webcam)

A code editor (optional but recommended):

VS Code: https://code.visualstudio.com/
Or any text editor you prefer
Step 1: Download the Code
Option A: Download from Replit

Click the 3 dots menu (⋮) in Replit
Select "Download as ZIP"
Extract the ZIP file to a folder on your computer
Option B: Clone from GitHub (if available)

git clone <repository-url>
cd camera-monitoring-system
Step 2: Install Required Libraries
Open your terminal (Command Prompt on Windows, Terminal on Mac/Linux) and navigate to the project folder:

cd path/to/CamRecognize
Then install all required Python packages:

pip install streamlit streamlit-webrtc opencv-python pillow numpy aiortc av requests
What each library does:

streamlit - Creates the web interface
streamlit-webrtc - Handles webcam streaming
opencv-python - Face detection and image processing
pillow - Image file handling
numpy - Math operations for images
aiortc - WebRTC communication protocol
av - Video processing
requests - Network requests for IP cameras
Step 3: Run the Application
In your terminal, run:

streamlit run app.py --server.port 5000
Or:

python -m streamlit run app.py --server.port 5000
Step 4: Open in Browser
After running the command, you'll see:

If app isnt being display in browser try ( http://localhost:5000 )

Local URL: http://localhost:5000
Network URL: http://192.168.x.x:5000
Open the Local URL in your web browser (Chrome or Edge recommended).

Step 5: Allow Camera Access
When you click "START" in Live Monitoring:

Your browser will ask for camera permission
Click "Allow" to grant access
Your camera feed will appear!
📖 How to Use the App - Page by Page Guide
The app has 5 main pages, accessible from the sidebar on the left.

1. 🏠 Dashboard Page
What it does: Shows an overview of your entire system

What you see:

System Statistics: Number of cameras, reference faces, and total detections
Quick Actions: Buttons to add cameras or reference faces
Recent Detections: Latest 10 face detection events with timestamps
System Status: Whether cameras are online/offline
How to use it:

When you first open the app, this is your starting page
Check your system stats to see what's configured
Use quick action buttons to set up cameras or add faces
Monitor recent activity in the detection history
What happens in the code:

Loads data from JSON files in the data/ folder
Counts cameras, faces, and detections
Displays recent detection history from detections.json
2. 📹 Camera Management Page
What it does: Add, remove, and manage your cameras

What you see:

Add Camera Section: Form to add new cameras
Camera List: All your configured cameras
Quick Add Buttons: One-click to add laptop webcam
Status Indicators: Shows if cameras are working (🟢 online / 🔴 offline)
How to use it:

To Add Your Laptop Webcam:

Click the green button: "Add Laptop Camera (Index 0)"
That's it! Your webcam is now added
To Add an IP Camera:

Enter a name (e.g., "Front Door Camera")
Enter the camera URL (e.g., http://192.168.1.100:8080/video)
Click "Add Camera"
The app will test the connection first
To Remove a Camera:

Find the camera in the list
Click the "Remove" button next to it
Confirm the deletion
What happens in the code:

camera_manager.py stores camera info in data/cameras.json
Tests camera connections before adding them
Uses OpenCV to check if webcam/IP camera is accessible
3. 👥 Face References Page
What it does: Upload and manage photos of people you want to recognize

What you see:

Upload Section: Area to upload a photo and enter a name
Reference Faces Grid: All saved reference faces with photos
Detection Tips: Helpful advice for better face detection
How to use it:

To Add a Reference Face:

Click "Browse files" and select a clear photo of someone's face
Enter the person's name (e.g., "John Smith")
Click "Save Reference Face"
The app will detect the face and save it
You'll see a success message and the face appears below
Tips for best results:

Use a well-lit photo (not too dark or bright)
Face should be clearly visible and looking at the camera
Avoid sunglasses, hats, or hands covering the face
Close-up photos work better than far-away shots
Make sure the image isn't blurry
To Remove a Reference Face:

Find the person in the grid below
Click the "Remove" button under their photo
Their face data will be deleted
What happens in the code:

face_recognition_engine.py processes the uploaded image
Uses OpenCV Haar Cascade to detect faces
Saves the original photo and a cropped face version
Stores face data in data/reference_faces.json
Face images are saved in data/reference_faces/ folder
4. 📊 Camera Feeds Page
What it does: View snapshot images from all your cameras

What you see:

Grid of Camera Snapshots: Live images from each camera
Auto-Refresh Toggle: Option to update snapshots automatically
Refresh Button: Manual refresh option
Timestamp: Shows when each snapshot was taken
How to use it:

Manual Refresh:

Click "📸 Refresh All Feeds" button
All camera snapshots update instantly
Auto-Refresh Mode:

Check the "Enable Auto-Refresh" box
Select refresh interval (5, 10, or 30 seconds)
Camera feeds will update automatically
You'll see a spinner while loading
What you see for each camera:

Camera name
Status indicator (🟢 online / 🔴 offline)
Current snapshot image
Timestamp of when the photo was taken
What happens in the code:

camera_manager.capture_frame() grabs a single image from each camera
Uses OpenCV VideoCapture to connect to camera
Converts image from BGR (OpenCV format) to RGB (display format)
Displays using Streamlit's st.image() function
5. 🎥 Live Monitoring Page
What it does: Real-time webcam streaming with live face recognition

What you see:

Video Feed: Live camera stream (15-30 FPS)
Play/Stop Controls: Start and stop the video stream
Live Detection Panel: Shows recognized faces with HUGE percentage numbers
Detection Overlays: Colored boxes drawn on faces in the video
Stream Stats: Frame count and detection count
How to use it:

To Start Live Monitoring:

Click the ▶️ START button
Your browser will ask for camera permission - click "Allow"
Wait 3-5 seconds for the connection
Your live video feed appears!
What you see on the video:

Green box around recognized faces
Shows the person's name in large text
Shows match percentage (e.g., "87.5%")
Red box around unknown faces
Shows "Unknown Face"
Shows confidence percentage
What you see in the side panel:

Green box with large percentage when someone is recognized
Red box when an unknown face is detected
Frame counter showing total frames processed
Detection count showing how many faces detected
Detection Behavior:

Face detection runs every 3 seconds (not every frame, to save processing power)
Between detections, you still see the video feed smoothly
Detection boxes stay on screen until the next detection runs
To Stop Monitoring:

Click the ⏸️ STOP button
Video feed stops
Camera turns off
What happens in the code:

Uses streamlit-webrtc library for video streaming
FaceRecognitionTransformer class processes each frame
Face detection throttled to every 3 seconds (controlled by detection_interval)
Detected faces compared to reference faces using template matching
Results drawn on video using OpenCV cv2.rectangle() and cv2.putText()
Notifications sent if someone is recognized (max 1 per minute)
6. ⚙️ Settings Page
What it does: Configure face recognition and notification settings

What you see:

Face Recognition Settings: Adjust detection sensitivity
Email Notification Settings: Configure email alerts
Test Email Button: Send a test email to verify settings
How to use it:

Face Recognition Settings:

Confidence Threshold Slider (0% to 100%)

This controls how strict face matching is
Higher percentage (e.g., 80%) = More strict, fewer false matches
Lower percentage (e.g., 50%) = More lenient, more matches but might be inaccurate
Default: 60%
Example: At 60%, a face needs to be 60% similar to match
Click "Save Settings" to apply changes

Email Notification Settings:

Enable Email Notifications: Check this box to turn on email alerts

Recipient Email: Enter the email address that will receive alerts

Example: your.email@gmail.com
SMTP Settings (for sending emails):

SMTP Server: Usually smtp.gmail.com for Gmail
SMTP Port: Usually 587
Sender Email: Your email address
Sender Password: Your email password or App Password
Click "Save Notification Settings"

Click "Send Test Email" to verify it works

Important Note for Gmail:

You need to use an App Password, not your regular password
Go to Google Account → Security → 2-Step Verification → App Passwords
Generate an app password for "Mail"
Use that password in the settings
What happens in the code:

Settings saved to data/face_settings.json and data/notification_settings.json
notification_service.py uses Python's smtplib to send emails
Email sent when faces are recognized (if enabled)
Debouncing prevents spam (max 1 email per person per minute)
🗂️ Understanding the Code - File by File
Let's break down each file and explain what every part does in simple terms.

📄 app.py - Main Application File
Purpose: This is the heart of the application. It creates the web interface and coordinates all the features.

Key Sections:
1. Imports (Lines 1-17)

import streamlit as st
import cv2
import json
...
What it does: Loads all the tools (libraries) we need

streamlit - Creates the web pages
cv2 (OpenCV) - Handles cameras and face detection
json - Saves and loads data files
camera_manager, face_recognition_engine, notification_service - Our custom code files
2. Session State Initialization (Lines 20-33)

if 'camera_manager' not in st.session_state:
    st.session_state.camera_manager = CameraManager()
What it does: Creates "memory" for the app

Session state = data that persists while you use the app
Initializes managers only once (not every time the page refreshes)
Stores: cameras, face engine, notifications, monitoring status
3. Helper Functions

a. get_cached_camera_status() (Lines 35-54)

def get_cached_camera_status(camera_url: str) -> bool:
What it does: Checks if a camera is working (online/offline)

Caching: Saves the result for 30 seconds
Why? Checking cameras is slow, so we don't check every second
Returns True if camera works, False if it doesn't
b. should_send_notification() (Lines 56-68)

def should_send_notification(match_name: str, camera_name: str) -> bool:
What it does: Prevents notification spam

Debouncing: Only allows 1 notification per person per minute
Keeps track of last notification time
Returns True if enough time has passed, False if too soon
4. FaceRecognitionTransformer Class (Lines 71-165)

class FaceRecognitionTransformer:
    def __init__(self, face_engine, notification_service, camera_name):
What it does: Processes each video frame for face recognition

Key Methods:

a. __init__() - Setup

self.detection_interval = 3  # Run face detection every 3 seconds
self.latest_matches = []
self.frame_count = 0
Sets up variables
detection_interval = 3 means detect faces every 3 seconds (not every frame)
Tracks latest matches and frame count
b. transform() - Process Each Frame

def transform(self, frame):
    img = frame.to_ndarray(format="bgr24")
Receives each video frame
Converts frame to a format OpenCV can use (numpy array)
Runs face detection every 3 seconds
Draws boxes and text on faces
Sends notifications if someone is recognized
How face detection works in this function:

Check if 3 seconds have passed since last detection
If yes: run face_engine.detect_faces_in_frame()
Save the results in self.latest_matches
Send notifications for any recognized faces
Draw boxes and text on ALL frames (using the latest results)
Drawing on frames:

cv2.rectangle(img, (x, y), (x + w, y + h), color, 4)
cv2.putText(img, name_label, ...)
cv2.rectangle() draws the colored box
cv2.putText() writes the name and percentage
Color is green (0, 255, 0) for matches, red (0, 0, 255) for unknown
5. Page Functions

Each page is a separate function. Streamlit calls the right one based on sidebar selection.

a. show_dashboard() (Lines 167-250)

Shows system overview
Displays stats (camera count, face count, detection count)
Shows recent detection history
Provides quick action buttons
b. show_camera_management() (Lines 252-350)

Displays form to add new cameras
Shows list of all cameras
Allows removing cameras
Tests camera connections
c. show_face_references() (Lines 352-450)

Upload photo form
Processes uploaded images
Detects faces in photos
Saves reference faces
Displays all saved faces in a grid
d. show_camera_feeds() (Lines 452-550)

Captures snapshots from each camera
Displays them in a grid
Auto-refresh option
Manual refresh button
e. show_live_monitoring() (Lines 552-660)

Creates WebRTC video stream
Processes frames with FaceRecognitionTransformer
Displays live detection results
Shows large percentage displays in sidebar
f. show_settings() (Lines 662-750)

Face recognition settings (confidence threshold)
Email notification settings
Test email button
Saves settings to JSON files
6. Main App Navigation (Lines 752-772)

st.sidebar.title("📹 Camera Monitoring System")
page = st.sidebar.radio("Navigation", pages)
What it does:

Creates the sidebar menu
Gets user's page selection
Calls the appropriate page function
Shows welcome message if no cameras configured
📄 camera_manager.py - Camera Management
Purpose: Handles all camera-related operations (add, remove, connect, capture).

Class: CameraManager
1. Initialization (Lines 10-13)

def __init__(self):
    self.cameras_file = "data/cameras.json"
    self.ensure_data_directory()
    self.cameras = self.load_cameras()
What it does:

Sets file path for saving camera data
Creates data folder if it doesn't exist
Loads existing cameras from JSON file
2. Data Management Methods

a. load_cameras() (Lines 19-27)

def load_cameras(self) -> List[Dict]:
What it does:

Reads cameras.json file
Returns a list of camera dictionaries
Each camera = {"name": "...", "url": "...", "status": "..."}
Returns empty list [] if file doesn't exist
b. save_cameras() (Lines 29-32)

def save_cameras(self):
What it does:

Writes the camera list to cameras.json
Uses indent=2 to make the JSON file human-readable
3. Camera Operations

a. add_camera() (Lines 34-56)

def add_camera(self, name: str, url: str) -> bool:
What it does:

Checks for duplicates: Ensures camera name/URL isn't already added
Validates connection:
If URL is a number (e.g., "0") = local webcam, skip test
If URL is an address = test the connection first
Creates camera data: Name, URL, date added, status
Saves to file
Returns: True if successful, False if failed
b. remove_camera() (Lines 58-64)

def remove_camera(self, index: int) -> bool:
What it does:

Removes camera at given position (index) in the list
Saves updated list to file
Returns True if successful
c. get_cameras() (Lines 66-68)

def get_cameras(self) -> List[Dict]:
What it does:

Simply returns the list of all cameras
Used by other parts of the app to display cameras
4. Camera Connection Methods

a. check_camera_status() (Lines 70-94)

def check_camera_status(self, url: str) -> bool:
What it does: Tests if a camera is working

Process:

For local webcam (e.g., "0"):

Converts string to integer
Opens camera with cv2.VideoCapture(0)
Tries to read one frame
Returns True if frame is captured successfully
For IP camera (e.g., "http://..."):

Opens camera with cv2.VideoCapture(url)
Tries to read one frame
If that fails, tries HTTP request as backup
Returns True if camera responds
b. capture_frame() (Lines 96-115)

def capture_frame(self, url: str) -> Optional[np.ndarray]:
What it does: Captures a single snapshot from camera

Process:

Opens camera connection
Reads one frame
Closes connection (important! releases camera)
Returns the frame as a numpy array (image data)
Returns None if capture failed
Why release the camera?

Cameras can only be accessed by one program at a time
Releasing allows other programs (or this app again) to use it
5. Helper Method

a. get_camera_by_name() (Lines 117-122)

def get_camera_by_name(self, name: str) -> Optional[Dict]:
What it does:

Searches for a camera by its name
Returns the camera dictionary if found
Returns None if not found
📄 face_recognition_engine.py - Face Detection & Recognition
Purpose: Handles all face detection and recognition logic using OpenCV.

Class: FaceRecognitionEngine
1. Initialization (Lines 11-55)

def __init__(self):
    # Initialize OpenCV face detector
    cascade_filename = 'haarcascade_frontalface_default.xml'
What it does:

Loads the Haar Cascade face detector (a pre-trained model)
Sets up file paths for storing reference faces and detections
Creates necessary directories
What is Haar Cascade?

It's a machine learning algorithm that can detect faces
Trained on thousands of face images
Looks for patterns like eyes, nose, mouth
Comes free with OpenCV
Why multiple path checks?

Different systems store OpenCV files in different locations
The code tries several common paths to find the cascade file
Falls back to cv2.data.haarcascades if all else fails
2. Reference Face Management

a. add_reference_face() (Lines 117-194)

def add_reference_face(self, image: Image.Image, name: str) -> bool:
What it does: Saves a photo as a reference for recognition

Step-by-step process:

Step 1: Convert image format

image_np = np.array(image)
if image_np.shape[2] == 4:  # RGBA
    image_cv = cv2.cvtColor(image_np, cv2.COLOR_RGBA2BGR)
PIL (Pillow) images are in RGB format
OpenCV uses BGR format (colors reversed)
Converts RGBA (with transparency) or RGB to BGR
Handles grayscale images too
Step 2: Prepare for detection

gray = cv2.cvtColor(image_cv, cv2.COLOR_BGR2GRAY)
gray = cv2.equalizeHist(gray)
Converts to grayscale (face detection works better on grayscale)
equalizeHist() improves contrast (makes faces easier to detect)
Step 3: Detect faces

faces = self.face_cascade.detectMultiScale(
    gray, 
    scaleFactor=1.1,
    minNeighbors=3,
    minSize=(30, 30)
)
detectMultiScale() finds all faces in the image
scaleFactor=1.1: How much to shrink image at each scale (smaller = slower but more accurate)
minNeighbors=3: How many detections needed to confirm a face (lower = more sensitive)
minSize=(30, 30): Minimum face size in pixels
If no faces found, try again with relaxed settings:

if len(faces) == 0:
    faces = self.face_cascade.detectMultiScale(
        gray, 
        scaleFactor=1.05,  # More sensitive
        minNeighbors=2
    )
Step 4: Extract and save the face

x, y, w, h = faces[0]  # Use first detected face
face_crop = image_cv[y:y+h, x:x+w]  # Crop to just the face
cv2.imwrite(face_path, face_crop)  # Save cropped face
Gets coordinates of the face box (x, y, width, height)
Crops the image to just the face area
Saves both the full image and cropped face
Step 5: Save metadata

self.reference_faces[name] = {
    'image_path': image_path,
    'face_path': face_path,
    'face_coordinates': (int(x), int(y), int(w), int(h)),
    'added_date': datetime.datetime.now().isoformat()
}
Stores file paths
Saves face coordinates (converted to Python int from numpy int32)
Records when the face was added
Saves to JSON file
b. remove_reference_face() (Lines 196-214)

def remove_reference_face(self, name: str) -> bool:
What it does:

Deletes both image files (full image and cropped face)
Removes entry from reference faces dictionary
Saves updated data to JSON
3. Face Detection in Live Video

a. detect_faces_in_frame() (Lines 234-308)

def detect_faces_in_frame(self, frame: np.ndarray, camera_name: str) -> List[Dict]:
What it does: Finds and recognizes faces in a video frame

Step-by-step process:

Step 1: Prepare frame

gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
gray = cv2.equalizeHist(gray)
Same as before: convert to grayscale and enhance contrast
Step 2: Detect faces

faces = self.face_cascade.detectMultiScale(gray, ...)
Finds all faces in the frame
Returns list of face coordinates
Step 3: For each detected face, compare to references

for (x, y, w, h) in faces:
    face_crop = gray[y:y+h, x:x+w]
    best_match = self.compare_face(face_crop)
Crops each detected face
Compares it to all saved reference faces
Finds the best match (if any)
Step 4: Build results

matches.append({
    'name': best_match['name'],
    'percentage': best_match['confidence'] * 100,
    'face_coordinates': (x, y, w, h),
    'camera': camera_name,
    'timestamp': datetime.datetime.now().isoformat()
})
Creates a dictionary for each detected face
Includes name, confidence percentage, location
Adds timestamp and camera name
Step 5: Log detection

If someone is recognized, adds to detection history
Saves to detections.json
Keeps record of all detections
4. Face Comparison Method

a. compare_face() (Lines 310-346)

def compare_face(self, face_image: np.ndarray) -> Dict:
What it does: Compares one face to all reference faces

Template Matching Process:

Step 1: Resize detected face to standard size

target_size = (100, 100)
face_resized = cv2.resize(face_image, target_size)
Makes all faces the same size for fair comparison
100x100 pixels is a good balance
Step 2: Compare to each reference face

for name, ref_data in self.reference_faces.items():
    ref_image = cv2.imread(ref_path, cv2.IMREAD_GRAYSCALE)
    ref_resized = cv2.resize(ref_image, target_size)
    
    result = cv2.matchTemplate(face_resized, ref_resized, cv2.TM_CCOEFF_NORMED)
    similarity = result[0][0]
What is matchTemplate()?

Compares two images pixel by pixel
Returns a similarity score from -1 to 1
cv2.TM_CCOEFF_NORMED = normalized correlation method
Higher score = more similar
Step 3: Find best match

if similarity > best_match['confidence']:
    best_match = {
        'name': name,
        'confidence': similarity
    }
Keeps track of the highest similarity score
Updates best_match if a better match is found
Step 4: Check threshold

if best_match['confidence'] < self.settings['confidence_threshold']:
    return {'name': 'No Match', 'confidence': 0}
If best match is below threshold (default 60%), return "No Match"
Threshold prevents false positives
5. Settings Management

a. load_settings() and update_settings()

Loads/saves face recognition settings
Default threshold: 0.6 (60%)
Stored in data/face_settings.json
📄 notification_service.py - Email Alerts
Purpose: Sends email notifications when faces are recognized.

Class: NotificationService
1. Initialization (Lines 10-32)

def __init__(self):
    self.settings_file = "data/notification_settings.json"
    self.settings = self.load_settings()
Default settings:

default_settings = {
    'email_enabled': False,
    'recipient_email': '',
    'smtp_server': 'smtp.gmail.com',
    'smtp_port': 587,
    'sender_email': '',
    'sender_password': ''
}
What each setting means:

email_enabled: Turn notifications on/off
recipient_email: Who receives the alerts
smtp_server: Email server address (Gmail, Outlook, etc.)
smtp_port: Connection port (587 for TLS)
sender_email: Your email address (that sends alerts)
sender_password: Your email password or app password
2. Sending Notifications

a. send_notification() (Lines 41-45)

def send_notification(self, message: str, subject: Optional[str] = None) -> bool:
What it does:

Checks if email is enabled
If yes, calls send_email_notification()
Returns True if sent successfully
b. send_email_notification() (Lines 47-91)

def send_email_notification(self, message: str, subject: Optional[str] = None) -> bool:
What it does: Sends an actual email

Step-by-step process:

Step 1: Validate settings

if not all([
    self.settings['recipient_email'],
    self.settings['sender_email'],
    self.settings['sender_password']
]):
    return False
Checks that all required fields are filled in
Returns False if anything is missing
Step 2: Create email message

msg = MIMEMultipart()
msg['From'] = self.settings['sender_email']
msg['To'] = self.settings['recipient_email']
msg['Subject'] = subject or "Camera Alert"
MIMEMultipart() creates an email object
Sets sender, recipient, and subject
Step 3: Add email body

body = f"""
Camera Monitoring Alert
{message}
Time: {datetime.datetime.now().strftime('%Y-%m-%d %H:%M:%S')}
"""
msg.attach(MIMEText(body, 'plain'))
Formats the message with timestamp
Attaches as plain text
Step 4: Connect to email server

server = smtplib.SMTP(self.settings['smtp_server'], self.settings['smtp_port'])
server.starttls()  # Enable encryption
server.login(self.settings['sender_email'], self.settings['sender_password'])
SMTP() connects to email server
starttls() encrypts the connection (secure)
login() authenticates with your credentials
Step 5: Send email

server.sendmail(sender, recipient, msg.as_string())
server.quit()  # Close connection
Sends the email
Closes the connection properly
What is SMTP?

Simple Mail Transfer Protocol
The standard way to send emails on the internet
Like the postal service for email
Why TLS (port 587)?

Transport Layer Security
Encrypts your email and password
Prevents hackers from intercepting your credentials
3. Testing

a. test_email_settings() (Lines 93-95)

def test_email_settings(self) -> bool:
    return self.send_email_notification("This is a test message", "Test Notification")
What it does:

Sends a test email to verify settings
Used by the Settings page "Test Email" button
💾 Data Storage
All data is stored in the data/ folder as JSON files (human-readable text format).

File Structure:
CamRecognize/
├── app.py
├── camera_manager.py
├── face_recognition_engine.py
├── notification_service.py
├── data/
│   ├── cameras.json              # Camera configurations
│   ├── reference_faces.json      # Reference face metadata
│   ├── detections.json           # Detection history
│   ├── face_settings.json        # Face recognition settings
│   ├── notification_settings.json # Email settings
│   └── reference_faces/          # Folder for face images
│       ├── John_Smith.jpg
│       ├── John_Smith_face.jpg
│       └── ...
└── .streamlit/
    └── config.toml               # Streamlit configuration
JSON File Examples:
cameras.json:

[
  {
    "name": "Laptop Camera",
    "url": "0",
    "added_date": "2024-10-22T10:30:00",
    "status": "active"
  }
]
reference_faces.json:

{
  "John Smith": {
    "image_path": "data/reference_faces/John_Smith.jpg",
    "face_path": "data/reference_faces/John_Smith_face.jpg",
    "face_coordinates": [487, 291, 321, 321],
    "added_date": "2024-10-22T11:15:00"
  }
}
detections.json:

[
  {
    "name": "John Smith",
    "confidence": 87.5,
    "camera": "Laptop Camera",
    "timestamp": "2024-10-22T12:00:00",
    "image_path": "data/detections/detection_12345.jpg"
  }
]
🔧 Troubleshooting
Camera Issues
Problem: Camera won't connect Solutions:

Make sure no other program is using the webcam (close Zoom, Skype, etc.)
Try index 1 or 2 instead of 0 for webcam
On Windows, try python -m streamlit run app.py instead of just streamlit run app.py
Check if your webcam has a physical privacy cover closed
Problem: "Connection is taking longer than expected" on Replit Solution: This is a Replit cloud limitation. Run the app locally on your computer for webcam access.

Face Detection Issues
Problem: Face not detected in uploaded photo Solutions:

Use a well-lit, clear photo
Face should be front-facing, not profile
Try a closer photo (face fills more of the frame)
Remove sunglasses/hats
Make sure image isn't too dark or too bright
Problem: Face recognition not accurate Solutions:

Increase confidence threshold in Settings (e.g., 70%)
Add more reference photos of the same person
Use clearer reference photos
Make sure lighting is similar between reference photo and live feed
Email Notification Issues
Problem: Email won't send Solutions:

For Gmail: Use an App Password, not your regular password
Go to: Google Account → Security → 2-Step Verification → App Passwords
Check that all email fields are filled in
Verify SMTP server and port are correct
Test your email login in a regular email client first
Installation Issues
Problem: pip install fails Solutions:

Make sure Python is added to PATH
Try: python -m pip install <package>
Update pip: python -m pip install --upgrade pip
On Windows, run Command Prompt as Administrator
Problem: Import errors when running Solution: Reinstall all packages:

pip uninstall streamlit streamlit-webrtc opencv-python
pip install streamlit streamlit-webrtc opencv-python pillow numpy aiortc av requests
🎓 Learning Resources
Want to learn more about the technologies used?

Python Basics: https://www.python.org/about/gettingstarted/
Streamlit Tutorial: https://docs.streamlit.io/library/get-started
OpenCV Face Detection: https://docs.opencv.org/4.x/db/d28/tutorial_cascade_classifier.html
Computer Vision: https://opencv.org/university/
📝 License
This project is for educational purposes. Feel free to modify and use for your own projects.

🙏 Credits
Built with:

Streamlit - Web framework
OpenCV - Computer vision
streamlit-webrtc - Video streaming
Python - Programming language
📧 Support
If you have questions or issues:

Check the Troubleshooting section above
Review the code comments in each file
Test with simpler configurations first (one camera, one reference face)
Happy monitoring!
