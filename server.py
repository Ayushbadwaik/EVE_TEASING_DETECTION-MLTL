import cv2
import requests
import time
import os

TELEGRAM_BOT_TOKEN = "8589038486:AAGwAE2apZwR2tn9ytcSu3Y7rNA2qNJpBOI"   
TELEGRAM_CHAT_ID = "5734819678" 
# Load prebuilt Haar cascades (these are simple XML files included with OpenCV)
face_cascade = cv2.CascadeClassifier(cv2.data.haarcascades + 'haarcascade_frontalface_default.xml')
upperbody_cascade = cv2.CascadeClassifier(cv2.data.haarcascades + 'haarcascade_upperbody.xml')

# Function to classify gender heuristically (simple color-based on hair)
def classify_gender(face_roi):
    # Convert to HSV for color detection
    hsv = cv2.cvtColor(face_roi, cv2.COLOR_BGR2HSV)
    # Define range for dark hair (adjust as needed; this is approximate)
    lower_hair = (0, 0, 0)  # Black/dark brown
    upper_hair = (180, 255, 50)
    mask = cv2.inRange(hsv, lower_hair, upper_hair)
    # Check top half of face for hair (simple heuristic)
    height, width = mask.shape
    top_half = mask[:height//2, :]
    hair_pixels = cv2.countNonZero(top_half)
    total_pixels = top_half.size
    if hair_pixels / total_pixels > 0.3:  # If >30% dark pixels, assume girl (long hair)
        return "girl"
    return "boy"

# Function to send Telegram message and photo
def send_telegram_alert(message, photo_path):
    url = f'https://api.telegram.org/bot{TELEGRAM_BOT_TOKEN}/sendMessage'
    data = {'chat_id': TELEGRAM_CHAT_ID, 'text': message}
    requests.post(url, data=data)
    
    # Send photo
    with open(photo_path, 'rb') as photo:
        files = {'photo': photo}
        url = f'https://api.telegram.org/bot{TELEGRAM_BOT_TOKEN}/sendPhoto'
        requests.post(url, data={'chat_id': TELEGRAM_CHAT_ID}, files=files)

# Initialize webcam
cap = cv2.VideoCapture(0)  # 0 for default camera
if not cap.isOpened():
    print("Error: Cannot access camera")
    exit()

alert_triggered = False  # To avoid spamming alerts

while True:
    ret, frame = cap.read()
    if not ret:
        break
    
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    
    # Detect faces and upper bodies
    faces = face_cascade.detectMultiScale(gray, scaleFactor=1.1, minNeighbors=5, minSize=(30, 30))
    bodies = upperbody_cascade.detectMultiScale(gray, scaleFactor=1.1, minNeighbors=5, minSize=(50, 50))
    
    # Combine detections (simple merge; in practice, associate faces with bodies)
    detections = list(faces) + list(bodies)  # Basic list; refine for accuracy
    
    girl_count = 0
    boy_count = 0
    seen_positions = []  # To avoid double-counting
    
    for (x, y, w, h) in detections:
        # Check if this detection overlaps with a seen one (simple distance check)
        center = (x + w//2, y + h//2)
        if any(abs(center[0] - p[0]) < 50 and abs(center[1] - p[1]) < 50 for p in seen_positions):
            continue
        seen_positions.append(center)
        
        # Extract face ROI if it's a face detection
        if (x, y, w, h) in faces:
            face_roi = frame[y:y+h, x:x+w]
            gender = classify_gender(face_roi)
            if gender == "girl":
                girl_count += 1
                cv2.rectangle(frame, (x, y), (x+w, y+h), (255, 0, 0), 2)  # Blue for girl
            else:
                boy_count += 1
                cv2.rectangle(frame, (x, y), (x+w, y+h), (0, 255, 0), 2)  # Green for boy
        else:
            # For body detections without face, assume boy (heuristic)
            boy_count += 1
            cv2.rectangle(frame, (x, y), (x+w, y+h), (0, 0, 255), 2)  # Red for body
    
    # Trigger condition: 1 girl and >2 boys
    if girl_count == 1 and boy_count > 2 and not alert_triggered:
        alert_triggered = True
        photo_path = f"alert_{int(time.time())}.jpg"
        cv2.imwrite(photo_path, frame)
        message = f"Alert: Potential eve teasing detected! 1 girl with {boy_count} boys."
        send_telegram_alert(message, photo_path)
        print("Alert sent!")
        # Reset after 30 seconds to allow new alerts
        time.sleep(30)
        alert_triggered = False
    
    # Display frame (optional)
    cv2.imshow('Detection', frame)
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()
