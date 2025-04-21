import time
import RPi.GPIO as GPIO
from picamera2 import Picamera2
from datetime import datetime
import os
import numpy as np
from tflite_runtime.interpreter import Interpreter
from PIL import Image

# Disable TensorFlow GPU usage
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'

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
interpreter = Interpreter(model_path='model_quant.tflite', num_threads=4)
interpreter.allocate_tensors()
input_details = interpreter.get_input_details()
output_details = interpreter.get_output_details()

# ============== USER CONFIGURATIONS =============
RELAY_GPIO_PIN = 18
PUMP_ON_DURATION = 5
IMAGE_SAVE_FOLDER = "../captured_images"
IMAGE_CAPTURE_INTERVAL = 2

# ============== HARDWARE SETUP =============
GPIO.setmode(GPIO.BCM)
GPIO.setup(RELAY_GPIO_PIN, GPIO.OUT, initial=GPIO.LOW)
time.sleep(0.5)  # GPIO stabilization delay

# ============== CAMERA SETUP =============
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

# ============== IMAGE PROCESSING =============
def load_image(image_path):
    img = Image.open(image_path).resize((224, 224))
    img_array = np.array(img, dtype=np.uint8)
    return np.expand_dims(img_array, axis=0)

def classify_image(image_path):
    input_data = load_image(image_path)
    interpreter.set_tensor(input_details[0]['index'], input_data)
    interpreter.invoke()
    predictions = interpreter.get_tensor(output_details[0]['index'])
    return plant_list[np.argmax(predictions[0])]

# ============== MAIN LOOP =============
def main():
    if not os.path.exists(IMAGE_SAVE_FOLDER):
        os.makedirs(IMAGE_SAVE_FOLDER)

    try:
        while True:
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            image_path = os.path.join(IMAGE_SAVE_FOLDER, f"img_{timestamp}.jpg")
            
            # Capture and process
            picam2.capture_file(image_path)
            print(f"Captured: {image_path}")
            
            try:
                prediction = classify_image(image_path)
                print(f"Prediction: {prediction}")
                
                if "healthy" not in prediction:
                    print(f"Activating pump for {PUMP_ON_DURATION}s")
                    GPIO.output(RELAY_GPIO_PIN, GPIO.HIGH)
                    time.sleep(PUMP_ON_DURATION)
                    GPIO.output(RELAY_GPIO_PIN, GPIO.LOW)
                
            except Exception as e:
                print(f"Processing error: {str(e)}")
                continue
            
            time.sleep(IMAGE_CAPTURE_INTERVAL)

    except KeyboardInterrupt:
        print("Exiting...")
    finally:
        picam2.close()
        GPIO.cleanup()

if __name__ == "__main__":
    main()