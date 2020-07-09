import onnxruntime
import numpy as np
from PIL import Image
from torchvision import transforms

transformations = transforms.Compose(
    [transforms.Resize((72, 72)),
     transforms.ToTensor(),
     transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))])

ort_session = onnxruntime.InferenceSession("skill.onnx")


def image_loader(image_name):
    """load image, returns cuda tensor"""
    image = Image.open(image_name)
    image = transformations(image).float()
    image.unsqueeze_(0)
    return image


def to_numpy(tensor):
    return tensor.detach().cpu().numpy() if tensor.requires_grad else tensor.cpu().numpy()


def is_ready(img):
    image = transformations(img).float()
    image = image.unsqueeze_(0)
    ort_inputs = {ort_session.get_inputs()[0].name: to_numpy(image)}
    ort_outs = ort_session.run(None, ort_inputs)
    x = ort_outs[0][0]
    print(x)
    smax = np.exp(x) / sum(np.exp(x))
    return smax[0] > 0.85
