import RPi.GPIO as GPIO
import time

# ============== SETUP GPIO =============
GPIO.setmode(GPIO.BCM)
GPIO.setup(18, GPIO.OUT, initial=GPIO.LOW)

while True:
    GPIO.output(18, GPIO.HIGH)
    time.sleep(30)
    GPIO.output(18, GPIO.LOW)
    print("[ACTION] Water pump turned off.")
        
        # Sleep until next capture
    time.sleep(5)