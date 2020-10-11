from pynput import mouse
from pynput.mouse import Button, Controller as MouseController
import threading
import mss
import mss.tools
from PIL import Image
import time
from detection import is_ready
from pynput.keyboard import Key, Controller as KeyboardController

from skills_detector import detect_skill_loc, classify_skills

active = False
last = time.time()
delay = 0.5
monitor_number = 1
width, height = 56, 56
x = 786
y = 1304
skills = [
    {"name": "s1", "trigger": -1, "action": "1"},
    {"name": "s2", "trigger": -1, "action": "2"},
    {"name": "s3", "trigger": -1, "action": "3"},
    {"name": "s4", "trigger": -1, "action": "4"},
    {"name": "s5", "trigger": 1, "action": Button.left},
    {"name": "s6", "trigger": 0, "action": Button.right},
]
skill_loc = []

mouse_ctrl = MouseController()
keyboard_ctrl = KeyboardController()


def on_click(x, y, button, pressed):
    global active, last
    if button == Button.right:
        active = pressed
        if active:
            last = time.time()


def fire(skill):
    action = skill["action"]
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
    global active, skill_loc
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
                if len(skill_loc) < 6:
                    print('detect skills location')
                    sct_img = sct.grab(monitor)
                    raw = Image.frombytes("RGB", sct_img.size, sct_img.bgra, "raw", "BGRX")
                    skill_loc = detect_skill_loc(raw)

                sct_img = sct.grab(monitor)
                raw = Image.frombytes("RGB", sct_img.size, sct_img.bgra, "raw", "BGRX")

                skill_images = []
                for loc in skill_loc:
                    skill_images.append(raw.crop(loc))

                skill_state = classify_skills(skill_images)
                print(skill_state)
                for i in range(len(skills)):
                    skill = skills[i]
                    if skill["trigger"] != 0:
                        if skill["trigger"] > 0:
                            fire_at_interval(skill)
                        elif skill_state[i] == 0:
                            fire(skill)


def main():
    th = threading.Thread(target=loop)
    th.start()
    print("detection start")
    with mouse.Listener(on_click=on_click) as listener:
        listener.join()


if __name__ == "__main__":
    for i in range(len(skills)):
        skills[i]["last_fire"] = time.time()
    main()
