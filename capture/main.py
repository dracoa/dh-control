import mss
import mss.tools
import time
from datetime import datetime


def capture():
    monitor_number = 1
    with mss.mss() as sct:
        mon = sct.monitors[monitor_number]
        monitor = {
            "top": mon["top"],  # 100px from the top
            "left": mon["left"],  # 100px from the left
            "width": mon["width"],
            "height": mon["height"],
            "mon": monitor_number,
        }
        while True:
            now = datetime.now()  # current date and time
            filename = sct.shot(mon=-1, output='{0}.png'.format(now.strftime("%m_%d_%Y_%H%M%S")))
            print(filename)
            time.sleep(60)


if __name__ == "__main__":
    capture()
