"""Dummy infinite loop for nsys profiling."""
import time

if __name__ == "__main__":
    print("Running dummy loop for profiling... (Ctrl+C to stop)")
    while True:
        time.sleep(0.1)
