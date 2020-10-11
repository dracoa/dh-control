import mss
import numpy as np
import onnxruntime
from PIL import Image

from utils.general import detect_onnx

bar_session = onnxruntime.InferenceSession('./onnx/bar.onnx')
skill_session = onnxruntime.InferenceSession('./onnx/skill_loc.onnx')
class_session = onnxruntime.InferenceSession('./onnx/fire_six.onnx')
climg_size_w = class_session.get_inputs()[0].shape[2]
climg_size_h = class_session.get_inputs()[0].shape[3]


def detect_skill_loc(image_src):
    bar_result = detect_onnx(image_src, bar_session)
    if bar_result is not None:
        bar_result = bar_result[0]
        confs = bar_result[:, 4:, ]
        m = np.argmax(confs)
        x1, y1, x2, y2, conf, _ = bar_result[m]
        w = (x2 - x1) * 3
        bar = image_src.crop((x1, y1, x1 + w, y2))
        bar.save('./outputs/bar.jpg')

        skill_result = detect_onnx(bar, skill_session)[0]
        skill_result = skill_result[skill_result[:, 0].argsort()]
        skills = []
        idx = 1
        for i in range(len(skill_result)):
            sx1, sy1, sx2, sy2, _, _ = skill_result[i]
            w, h = (sx2 - sx1, sy2 - sy1)
            coord = [sx1 + x1, sy1 + y1, sx2 + x1, sy2 + y1]
            skills.append(coord)
            image_src.crop(coord).save('./outputs/skill_{}.jpg'.format(idx))
            print(idx, w, h, w / h)
            idx += 1
        return skills


def classify_skills(skills):
    inputs = []
    for img in skills:
        resized = img.resize((climg_size_w, climg_size_h))
        img_in = np.transpose(resized, (2, 0, 1)).astype(np.float32)  # HWC -> CHW
        img_in /= 255.0
        inputs.append(img_in)
    inputs = np.array(inputs)
    input_name = class_session.get_inputs()[0].name
    outputs = class_session.run(None, {input_name: inputs})
    outputs = outputs[0]
    return outputs.argmax(axis=1).flatten()


if __name__ == "__main__":
    with mss.mss() as sct:
        mon = sct.monitors[1]
        monitor = {
            "top": mon["top"],  # 100px from the top
            "left": mon["left"],  # 100px from the left
            "width": mon["width"],
            "height": mon["height"],
            "mon": 1,
        }
        sct_img = sct.grab(monitor)
        raw = Image.frombytes("RGB", sct_img.size, sct_img.bgra, "raw", "BGRX")

    raw = Image.open('./images/4.jpg')
    skills = detect_skill_loc(raw)
    print(skills)
