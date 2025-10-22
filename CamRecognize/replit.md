# Camera Monitoring System with Face Recognition

## Overview

This is a Streamlit-based camera monitoring system that combines live camera feeds with facial recognition capabilities. The application enables users to monitor multiple camera sources, detect faces in real-time, and receive notifications when specific faces are recognized. The system is designed for security monitoring applications where identifying known individuals is critical.

## User Preferences

Preferred communication style: Simple, everyday language.

## System Architecture

### Frontend Architecture
- **Framework**: Streamlit web application with multi-page navigation
- **User Interface**: Clean dashboard-driven design with sidebar navigation
- **Pages**: Dashboard, Camera Management, Face References, Live Monitoring, and Settings
- **State Management**: Uses Streamlit's session state for maintaining application state across user interactions

### Backend Architecture
- **Modular Design**: Separation of concerns through distinct manager classes
- **Camera Manager**: Handles camera configuration, connection testing, and stream management
- **Face Recognition Engine**: Processes video frames for face detection and recognition using OpenCV
- **Notification Service**: Manages alert delivery through configurable channels

### Data Storage Solutions
- **File-based Storage**: JSON files for configuration and metadata storage
- **Directory Structure**: Organized data directory containing:
  - Camera configurations (`cameras.json`)
  - Reference face data (`reference_faces.json`)
  - Detection logs (`detections.json`)
  - System settings for face recognition and notifications
- **Image Storage**: Local filesystem storage for reference face images and detection snapshots

### Face Recognition Technology
- **Detection Algorithm**: OpenCV Haar Cascade classifiers for face detection
- **Processing Pipeline**: Real-time frame analysis with configurable detection sensitivity
- **Reference System**: Stores known face images for comparison against detected faces
- **Detection Logging**: Maintains historical records of face detection events

### Notification System
- **Email Integration**: SMTP-based email notifications for security alerts
- **Configurable Settings**: User-defined notification preferences and email server configuration
- **Event-driven Alerts**: Automatic notifications triggered by face recognition events

## External Dependencies

### Core Libraries
- **Streamlit**: Web application framework for the user interface
- **OpenCV (cv2)**: Computer vision library for face detection and image processing
- **NumPy**: Numerical computing for image array manipulation
- **PIL (Pillow)**: Image processing and format conversion
- **Requests**: HTTP client for camera stream connectivity testing

### System Dependencies
- **OpenCV Haar Cascades**: Pre-trained face detection models
- **File System Access**: Local storage for configuration files and image data
- **SMTP Email Services**: External email servers for notification delivery (Gmail by default)

### Camera Integration
- **IP Camera Support**: Compatible with network cameras providing HTTP/RTSP streams
- **USB Camera Support**: Local camera device integration through OpenCV
- **Stream Validation**: Connection testing before camera registration

### Data Persistence
- **JSON Configuration Files**: Human-readable configuration storage
- **Image File Management**: Local filesystem for face reference images and detection logs
- **Cross-session State**: Persistent storage of user preferences and system settings