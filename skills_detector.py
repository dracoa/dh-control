import torch
import numpy as np
import onnxruntime
from PIL import Image
from utils.general import non_max_suppression, letterbox_image, scale_coords

bar_session = onnxruntime.InferenceSession('./onnx/bar.onnx')
skill_session = onnxruntime.InferenceSession('./onnx/skill.onnx')
class_session = onnxruntime.InferenceSession('./onnx/fire_six.onnx')
climg_size_w = class_session.get_inputs()[0].shape[2]
climg_size_h = class_session.get_inputs()[0].shape[3]


def detect_onnx(image_src, session):
    img_size_h = session.get_inputs()[0].shape[2]
    img_size_w = session.get_inputs()[0].shape[3]
    resized = letterbox_image(image_src, (img_size_w, img_size_h))

    img_in = np.transpose(resized, (2, 0, 1)).astype(np.float32)  # HWC -> CHW
    img_in = np.expand_dims(img_in, axis=0)
    img_in /= 255.0

    # inference
    input_name = session.get_inputs()[0].name
    outputs = session.run(None, {input_name: img_in})

    batch_detections = torch.from_numpy(np.array(outputs[0]))
    batch_detections = non_max_suppression(batch_detections, conf_thres=0.4, iou_thres=0.5, agnostic=False)
    if batch_detections[0] is not None:
        boxs = batch_detections[0][..., :4]
        confs = batch_detections[0][..., 4]
        w, h = image_src.size
        boxs[:, :] = scale_coords((img_size_w, img_size_h), boxs[:, :], (h, w)).round()
        return np.append(boxs.numpy(), confs.numpy().reshape(-1, 1), axis=1)

    return None


def detect_skill_loc(image_src):
    bar_result = detect_onnx(image_src, bar_session)
    if bar_result is not None:
        print(bar_result)
        confs = bar_result[:, 4:, ]
        m = np.argmax(confs)
        x1, y1, x2, y2, conf = bar_result[m]
        w = (x2 - x1) * 3
        bar = image_src.crop((x1, y1, x1 + w, y2))
        bar.save('./outputs/bar.jpg')

        skill_result = detect_onnx(bar, skill_session)
        skill_result = skill_result[skill_result[:, 0].argsort()]
        skills = []
        idx = 1
        for i in range(len(skill_result)):
            sx1, sy1, sx2, sy2, _ = skill_result[i]
            w, h = (sx2 - sx1, sy2 - sy1)
            if w / h < 1.1:
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
    return outputs.argmax( axis=1).flatten()


if __name__ == "__main__":
    img = Image.open('./images/4.jpg')
    skills = detect_skill_loc(img)
    print(skills)
