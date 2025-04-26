from gpiozero import OutputDevice
from time import sleep
import atexit

# ===== Configuration =====
RELAY_PIN = 18            # GPIO pin connected to the relay
PUMP_ON_TIME = 30         # Seconds to keep the pump ON
PUMP_OFF_TIME = 5         # Seconds to keep the pump OFF

# ===== Setup =====
relay = OutputDevice(RELAY_PIN, active_high=True, initial_value=False)

# ===== Safety Cleanup =====
def shutdown():
    """Turn off the relay and cleanup on exit."""
    print("Script stopped! Ensuring relay is OFF.")
    relay.off()

atexit.register(shutdown)  # Run on normal/forced exit

# ===== Main Loop =====
try:
    print("Starting water pump control script...")
    while True:
        relay.on()  # Activate relay → pump ON
        print(f"Pump ON for {PUMP_ON_TIME} seconds")
        sleep(PUMP_ON_TIME)

        relay.off()  # Deactivate relay → pump OFF
        print(f"Pump OFF for {PUMP_OFF_TIME} seconds")
        sleep(PUMP_OFF_TIME)

except KeyboardInterrupt:
    print("User stopped the script")
except Exception as e:
    print(f"CRITICAL ERROR: {e}")
finally:
    shutdown()  # Ensure relay is OFF