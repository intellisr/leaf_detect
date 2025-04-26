from gpiozero import OutputDevice
from time import sleep
import sys

relay = OutputDevice(15, active_high=False, initial_value=True)  # Starts OFF (if active-low)

try:
    print("Starting relay test... (Ctrl+C to stop)")
    while True:
        print("Turning ON")
        relay.on()  # Or relay.value = 1
        sleep(2)
        print("Turning OFF")
        relay.off()  # Or relay.value = 0
        sleep(2)
except KeyboardInterrupt:
    print("\nExiting...")
    relay.off()  # Force OFF
    sys.exit(0)