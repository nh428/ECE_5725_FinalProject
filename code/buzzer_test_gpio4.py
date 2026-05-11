#!/usr/bin/env python3

import time
import RPi.GPIO as GPIO

BUZZER_PIN = 4      # BCM GPIO 4
FREQUENCY = 2000    # Hz

GPIO.setmode(GPIO.BCM)
GPIO.setwarnings(False)
GPIO.setup(BUZZER_PIN, GPIO.OUT)

pwm = GPIO.PWM(BUZZER_PIN, FREQUENCY)

try:
    print("PWM buzzer test on GPIO 4")
    print("Press Ctrl+C to stop.")
    print()

    while True:
        print("Buzzer ON")
        pwm.start(50)   # 50% duty cycle
        time.sleep(0.5)

        print("Buzzer OFF")
        pwm.stop()
        time.sleep(0.5)

except KeyboardInterrupt:
    print("\nStopping buzzer test...")

finally:
    pwm.stop()
    GPIO.cleanup()
    print("GPIO cleaned up.")