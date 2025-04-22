import RPi.GPIO as GPIO
from picamera2 import Picamera2

RELAY_GPIO_PIN = 18  # BCM pin you connected the relay to
PUMP_ON_DURATION = 5  # seconds

# Camera Configuration
IMAGE_SAVE_FOLDER = "../captured_images"
IMAGE_CAPTURE_INTERVAL = 2  # seconds (time between captures)

# ============== SETUP CAMERA =============
picam2 = Picamera2()
config = picam2.create_still_configuration(main={"size": (640, 480)})
picam2.configure(config)
picam2.start()

# ============== SETUP GPIO =============
GPIO.setmode(GPIO.BCM)
GPIO.setup(RELAY_GPIO_PIN, GPIO.OUT, initial=GPIO.LOW)