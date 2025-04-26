import RPi.GPIO as GPIO
import time

# Setup
GPIO.setwarnings(False)
GPIO.cleanup()
GPIO.setmode(GPIO.BCM)
GPIO.setup(18, GPIO.OUT, initial=GPIO.LOW)  # Start with relay OFF
time.sleep(1)  # Let relay stabilize

try:
    while True:
        GPIO.output(18, GPIO.HIGH)  # Relay ON → pump ON
        print("[ACTION] Water pump turned ON (GPIO 18 HIGH).")
        time.sleep(30)  # Run for 30 seconds

        GPIO.output(18, GPIO.LOW)  # Relay OFF → pump OFF
        print("[ACTION] Water pump turned OFF (GPIO 18 LOW).")
        time.sleep(5)  # Pause for 5 seconds

except KeyboardInterrupt:
    print("\nUser interrupted script.")
finally:
    GPIO.output(18, GPIO.LOW)  # Force relay OFF
    GPIO.cleanup()
    print("GPIO cleaned up. Relay is OFF.")