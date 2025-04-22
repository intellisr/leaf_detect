import time
import cv2
import RPi.GPIO as GPIO
from datetime import datetime
import os
import numpy as np
import tensorflow as tf

# Disable TensorFlow logging
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'

# List of plant disease classes
plant_list = ['Apple___Apple_scab', 'Apple___Black_rot', 'Apple___Cedar_apple_rust', 'Apple___healthy', 
              'Cherry_(including_sour)___Powdery_mildew', 'Cherry_(including_sour)___healthy', 
              'Corn_(maize)___Cercospora_leaf_spot Gray_leaf_spot', 'Corn_(maize)___Common_rust_', 
              'Corn_(maize)___Northern_Leaf_Blight', 'Corn_(maize)___healthy', 'Grape___Black_rot', 
              'Grape___Esca_(Black_Measles)', 'Grape___Leaf_blight_(Isariopsis_Leaf_Spot)', 'Grape___healthy',
              'Orange___Haunglongbing_(Citrus_greening)', 'Orange___healthy', 'Peach___Bacterial_spot',
              'Peach___healthy', 'Pepper,_bell___Bacterial_spot', 'Pepper,_bell___healthy',
              'Potato___Early_blight', 'Potato___Late_blight', 'Potato___healthy',
              'Strawberry___Leaf_scorch', 'Strawberry___healthy', 'Tomato___Bacterial_spot',
              'Tomato___Early_blight', 'Tomato___Late_blight', 'Tomato___Leaf_Mold',
              'Tomato___Septoria_leaf_spot', 'Tomato___Spider_mites Two-spotted_spider_mite',
              'Tomato___Target_Spot', 'Tomato___Tomato_Yellow_Leaf_Curl_Virus',
              'Tomato___Tomato_mosaic_virus', 'Tomato___healthy']


# ============== MODEL SETUP =============
model_path = 'model2.tflite'
try:
    print("Loading TFLite model...")
    interpreter = tf.lite.Interpreter(model_path=model_path)
    interpreter.allocate_tensors()
    input_details = interpreter.get_input_details()
    output_details = interpreter.get_output_details()
except Exception as e:
    print(f"Error loading TFLite model: {e}")
    exit(1)

# ============== USER CONFIGURATIONS =============
RELAY_GPIO_PIN = 18
PUMP_ON_DURATION = 5
IMAGE_SAVE_FOLDER = "./captured_images"
IMAGE_CAPTURE_INTERVAL = 5
CAMERA_INDEX = 0  # Default camera (0 for Pi Camera or USB webcam)

# ============== HARDWARE SETUP =============
try:
    print("Setting up GPIO...")
    GPIO.setmode(GPIO.BCM)
    GPIO.setup(RELAY_GPIO_PIN, GPIO.OUT, initial=GPIO.LOW)
    time.sleep(0.5)
except Exception as e:
    print(f"Error setting up GPIO: {e}")
    exit(1)

# ============== CAMERA SETUP =============
try:
    print("Setting up OpenCV camera...")
    cap = cv2.VideoCapture(CAMERA_INDEX)
    if not cap.isOpened():
        raise Exception("Failed to open camera")
    cap.set(cv2.CAP_PROP_FRAME_WIDTH, 224)
    cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 224)
    cap.set(cv2.CAP_PROP_FPS, 15)  # Lower FPS to reduce load
    time.sleep(2)  # Camera warm-up
except Exception as e:
    print(f"Error setting up camera: {e}")
    exit(1)

# ============== IMAGE PROCESSING =============
def preprocess_image(image_array):
    try:
        # Ensure RGB format (OpenCV uses BGR by default)
        image_array = cv2.cvtColor(image_array, cv2.COLOR_BGR2RGB)
        # Convert to float32 and normalize to [-1, 1] for MobileNetV2
        img_array = image_array.astype(np.float32)
        img_array = img_array / 127.5 - 1.0
        # Expand dimensions to match model input shape (1, 224, 224, 3)
        img_array = np.expand_dims(img_array, axis=0)
        return img_array
    except Exception as e:
        print(f"Error preprocessing image: {e}")
        return None

def classify_image(image_array):
    input_data = preprocess_image(image_array)
    if input_data is None:
        return None, None
    
    try:
        interpreter.set_tensor(input_details[0]['index'], input_data)
        interpreter.invoke()
        predictions = interpreter.get_tensor(output_details[0]['index'])
        predicted_class_idx = np.argmax(predictions[0])
        confidence = predictions[0][predicted_class_idx] * 100
        return plant_list[predicted_class_idx], confidence
    except Exception as e:
        print(f"Error during inference: {e}")
        return None, None

# ============== MAIN LOOP =============
def main():
    if not os.path.exists(IMAGE_SAVE_FOLDER):
        os.makedirs(IMAGE_SAVE_FOLDER)

    try:
        while True:
            print("Starting capture cycle...")
            
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            image_path = os.path.join(IMAGE_SAVE_FOLDER, f"img_{timestamp}.jpg")
            
            # Capture image
            try:
                ret, frame = cap.read()
                if not ret:
                    print("Failed to capture frame")
                    continue
                # Resize to 224x224
                frame = cv2.resize(frame, (224, 224))
                # Save image (convert BGR to RGB for saving)
                cv2.imwrite(image_path, frame)
                print(f"Captured: {image_path}")
            except Exception as e:
                print(f"Error capturing image: {e}")
                continue
            
            # Classify image
            try:
                predicted_class, confidence = classify_image(frame)
                if predicted_class is None:
                    print("Skipping due to classification error")
                    continue
                
                print(f"Prediction: {predicted_class} ({confidence:.2f}%)")
                
                if "healthy" not in predicted_class.lower():
                    print(f"Activating pump for {PUMP_ON_DURATION}s")
                    GPIO.output(RELAY_GPIO_PIN, GPIO.HIGH)
                    time.sleep(PUMP_ON_DURATION)
                    GPIO.output(RELAY_GPIO_PIN, GPIO.LOW)
                
            except Exception as e:
                print(f"Processing error: {e}")
                continue
            
            # Clear memory
            frame = None
            time.sleep(IMAGE_CAPTURE_INTERVAL)

    except KeyboardInterrupt:
        print("Exiting...")
    finally:
        cap.release()
        GPIO.cleanup()

if __name__ == "__main__":
    main()