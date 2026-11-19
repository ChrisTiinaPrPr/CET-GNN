"""Bounded retry for Windows readers briefly blocking atomic file replacement."""
import os
import time


def replace_with_retry(source, destination, attempts=10):
    for attempt in range(attempts):
        try:
            os.replace(source, destination)
            return
        except PermissionError as error:
            if getattr(error, 'winerror', None) not in (5, 32, 33) or attempt == attempts-1:
                raise
            time.sleep(min(.05 * 2**attempt, .5))
