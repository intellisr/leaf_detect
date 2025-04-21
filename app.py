import time
import RPi.GPIO as GPIO
from picamera2 import Picamera2
from datetime import datetime
import os
import numpy as np
import tensorflow as tf
from PIL import Image

# Disable TensorFlow GPU usage and logging
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
IMAGE_SAVE_FOLDER = "../captured_images"
IMAGE_CAPTURE_INTERVAL = 2

# ============== HARDWARE SETUP =============
try:
    GPIO.setmode(GPIO.BCM)
    GPIO.setup(RELAY_GPIO_PIN, GPIO.OUT, initial=GPIO.LOW)
    time.sleep(0.5)  # GPIO stabilization delay
except Exception as e:
    print(f"Error setting up GPIO: {e}")
    exit(1)

# ============== CAMERA SETUP =============
try:
    picam2 = Picamera2()
    config = picam2.create_still_configuration(
        main={"size": (224, 224)},
        buffer_count=2,
        queue=False,
        display=None
    )
    picam2.configure(config)
    picam2.set_controls({"AwbEnable": True, "FrameDurationLimits": (40000, 40000)})
    picam2.start()
    time.sleep(2)  # Camera warm-up
except Exception as e:
    print(f"Error setting up camera: {e}")
    exit(1)

# ============== IMAGE PROCESSING =============
def preprocess_image(image_path):
    try:
        # Load and resize image
        img = Image.open(image_path).resize((224, 224))
        # Convert to array and normalize to [-1, 1] for MobileNetV2
        img_array = np.array(img, dtype=np.float32)
        img_array = img_array / 127.5 - 1.0
        # Expand dimensions to match model input shape (1, 224, 224, 3)
        img_array = np.expand_dims(img_array, axis=0)
        return img_array
    except Exception as e:
        print(f"Error preprocessing image: {e}")
        return None

def classify_image(image_path):
    input_data = preprocess_image(image_path)
    if input_data is None:
        return None, None
    
    try:
        # Set the input tensor
        interpreter.set_tensor(input_details[0]['index'], input_data)
        # Run inference
        interpreter.invoke()
        # Get the output tensor
        predictions = interpreter.get_tensor(output_details[0]['index'])
        predicted_class_idx = np.argmax(predictions[0])
        confidence = predictions[0][predicted_class_idx] * 100
        return plant_list[predicted_class_idx], confidence
    except Exception as e:
        print(f"Error during inference: {e}")
        return None, None

# ============== MAIN LOOP =============
def main():
    # Create image save folder if it doesn't exist
    if not os.path.exists(IMAGE_SAVE_FOLDER):
        os.makedirs(IMAGE_SAVE_FOLDER)

    try:
        while True:
            # Generate timestamped image filename
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            image_path = os.path.join(IMAGE_SAVE_FOLDER, f"img_{timestamp}.jpg")
            
            # Capture image
            try:
                picam2.capture_file(image_path)
                print(f"Captured: {image_path}")
            except Exception as e:
                print(f"Error capturing image: {e}")
                continue
            
            # Classify image
            try:
                predicted_class, confidence = classify_image(image_path)
                if predicted_class is None:
                    print("Skipping due to classification error")
                    continue
                
                print(f"Prediction: {predicted_class} ({confidence:.2f}%)")
                
                # Activate pump if plant is not healthy
                if "healthy" not in predicted_class.lower():
                    print(f"Activating pump for {PUMP_ON_DURATION}s")
                    GPIO.output(RELAY_GPIO_PIN, GPIO.HIGH)
                    time.sleep(PUMP_ON_DURATION)
                    GPIO.output(RELAY_GPIO_PIN, GPIO.LOW)
                
            except Exception as e:
                print(f"Processing error: {e}")
                continue
            
            # Wait before next capture
            time.sleep(IMAGE_CAPTURE_INTERVAL)

    except KeyboardInterrupt:
        print("Exiting...")
    finally:
        # Cleanup resources
        picam2.close()
        GPIO.cleanup()

if __name__ == "__main__":
    main()