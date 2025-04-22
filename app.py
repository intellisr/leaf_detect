import time
import RPi.GPIO as GPIO
from picamera2 import Picamera2
from datetime import datetime
import os
import tensorflow as tf
from tensorflow.keras.applications import MobileNetV2
from tensorflow.keras.layers import Dense, GlobalAveragePooling2D
from tensorflow.keras.models import Model
from tensorflow.keras.preprocessing import image
import numpy as np

plant_list = ['Not_Detected', 'Tomato_healthy', 'Tomato_unhealthy']

# Model setup
base_model = MobileNetV2(weights=None, include_top=False, input_shape=(224, 224, 3))
x = base_model.output
x = GlobalAveragePooling2D()(x)
x = Dense(1024, activation='relu')(x)
predictions = Dense(3, activation='softmax')(x)
model = Model(inputs=base_model.input, outputs=predictions)

# Load the weights
model.load_weights('model.weights.h5')

# ============== USER CONFIGURATIONS =============
# GPIO and Relay Configuration
RELAY_GPIO_PIN = 18  # BCM pin you connected the relay to
PUMP_ON_DURATION = 10  # seconds

# Camera Configuration
IMAGE_SAVE_FOLDER = "../captured_images"
IMAGE_CAPTURE_INTERVAL = 5  # seconds (time between captures)

# ============== SETUP GPIO =============
GPIO.setmode(GPIO.BCM)
GPIO.setup(RELAY_GPIO_PIN, GPIO.OUT, initial=GPIO.LOW)

# ============== SETUP CAMERA =============
picam2 = Picamera2()
config = picam2.create_still_configuration(main={"size": (640, 480)})
picam2.configure(config)
picam2.start()

# Create folder to save images if it doesn't exist
if not os.path.exists(IMAGE_SAVE_FOLDER):
    os.makedirs(IMAGE_SAVE_FOLDER)

# ============== CLASSIFICATION FUNCTION =============
def classify_image(image_path: str) -> str:
    """    
    Return a predicted class label as string.
    """
    img = image.load_img(image_path, target_size=(224, 224))
    img_array = image.img_to_array(img)
    img_array = np.expand_dims(img_array, axis=0)
    img_array /= 255.0
    
    # Make a prediction
    predictions = model.predict(img_array)

    # Get the class label with the highest predicted probability
    predicted_class_index = np.argmax(predictions[0])
    predicted_class_label = plant_list[predicted_class_index]

    print(f"Predicted class: {predicted_class_label}")
    
    return predicted_class_label

# ============== MAIN LOOP =============
try:
    while True:
        # 1. Capture image
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        image_filename = f"img_{timestamp}.jpg"
        image_path = os.path.join(IMAGE_SAVE_FOLDER, image_filename)
        
        picam2.capture_file(image_path)
        print(f"[INFO] Captured image: {image_path}")

        # 2. Classify the image
        predicted_class = classify_image(image_path)
        print(f"[INFO] Predicted Class: {predicted_class}")

        # 3. Conditional logic to turn on the relay
        # IMPORTANT: Replace "Tomato___healthy" with the actual class you want to trigger watering
        if "unhealthy" not in predicted_class:  # Example: water if plant is not healthy
            print(f"[ACTION] {predicted_class} detected. Turning on water pump.")
            GPIO.output(RELAY_GPIO_PIN, GPIO.HIGH)
            time.sleep(PUMP_ON_DURATION)
            GPIO.output(RELAY_GPIO_PIN, GPIO.LOW)
            print("[ACTION] Water pump turned off.")
        
        # Sleep until next capture
        time.sleep(IMAGE_CAPTURE_INTERVAL)

except KeyboardInterrupt:
    print("[INFO] Exiting script.")
except Exception as e:
    print(f"[ERROR] An unexpected error occurred: {e}")
finally:
    # 4. Cleanup resources
    picam2.close()
    GPIO.cleanup()