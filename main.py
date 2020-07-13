from pynput import mouse
from pynput.mouse import Button, Controller as MouseController
import threading
import mss
import mss.tools
from PIL import Image
import time
from detection import is_ready
from pynput.keyboard import Key, Controller as KeyboardController
from datetime import datetime

active = False
last = time.time()
delay = 0.5
monitor_number = 1
width, height = 56, 56
y = 1001
skills = [
    {"name": "s1", "trigger": -1, "action": "1", "x": 631, "y": y},
    {"name": "s2", "trigger": -1, "action": "2", "x": 698, "y": y},
    {"name": "s3", "trigger": -1, "action": "3", "x": 764, "y": y},
    {"name": "s4", "trigger": -1, "action": "4", "x": 831, "y": y},
    {"name": "s5", "trigger": 1, "action": Button.left, "x": 900, "y": y},
    {"name": "s6", "trigger": 0, "action": Button.right, "x": 966, "y": y},
]
mouse_ctrl = MouseController()
keyboard_ctrl = KeyboardController()


def on_click(x, y, button, pressed):
    global active, last
    if button == Button.right:
        active = pressed
        if active:
            last = time.time()
        print(active, last)


def fire(skill):
    action = skill["action"]
    print('fire {0}'.format(action))
    if type(action) is str:
        keyboard_ctrl.type(action)
    else:
        keyboard_ctrl.press(Key.shift_l)
        mouse_ctrl.click(action, 1)
        keyboard_ctrl.release(Key.shift_l)
    skill["last_fire"] = time.time()


def fire_at_interval(skill):
    diff = time.time() - skill["last_fire"]
    if diff > skill["trigger"]:
        fire(skill)


def fire_when_ready(raw, skill):
    box = (skill["x"], skill["y"], skill["x"] + width, skill["y"] + height)
    simg = raw.crop(box)
    if is_ready(simg):
        fire(skill)


def loop():
    global active
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
            if active and time.time() - last > delay:
                sct_img = sct.grab(monitor)
                raw = Image.frombytes("RGB", sct_img.size, sct_img.bgra, "raw", "BGRX")
                for skill in skills:
                    if skill["trigger"] != 0:
                        if skill["trigger"] > 0:
                            fire_at_interval(skill)
                        else:
                            fire_when_ready(raw, skill)


def capture():
    global active
    with mss.mss() as sct:
        mon = sct.monitors[monitor_number]
        monitor = {
            "top": mon["top"],  # 100px from the top
            "left": mon["left"],  # 100px from the left
            "width": mon["width"],
            "height": mon["height"],
            "mon": monitor_number,
        }
        skill = skills[0]
        box = (skill["x"], skill["y"], skill["x"] + width, skill["y"] + height)
        while True:
            if active:
                sct_img = sct.grab(monitor)
                raw = Image.frombytes("RGB", sct_img.size, sct_img.bgra, "raw", "BGRX")
                simg = raw.crop(box)
                simg.save('images/temp/{0}.jpg'.format(datetime.now().strftime("%Y%m%d_%H%M%S.%f")))
                time.sleep(0.1)


def main():
    th = threading.Thread(target=loop)
    th.start()
    with mouse.Listener(on_click=on_click) as listener:
        listener.join()


if __name__ == "__main__":
    for s in skills:
        s["last_fire"] = time.time()
    main()
