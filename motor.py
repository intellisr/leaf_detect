import RPi.GPIO as GPIO
import time

GPIO.cleanup()
# ============== SETUP GPIO =============
GPIO.setmode(GPIO.BCM)

GPIO.setup(18, GPIO.OUT)
GPIO.output(18, GPIO.LOW)

while True:
    GPIO.output(18, GPIO.HIGH)
    time.sleep(30)
    GPIO.output(18, GPIO.LOW)
    print("[ACTION] Water pump turned off.")
        
    time.sleep(5)