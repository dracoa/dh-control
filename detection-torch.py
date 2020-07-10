import onnxruntime
import numpy as np
from PIL import Image
from torchvision import transforms

label = ['active', 'non_skill', 'not_ready', 'ready']

transformations = transforms.Compose(
    [transforms.Resize((72, 72)),
     transforms.ToTensor(),
     transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))])

ort_session = onnxruntime.InferenceSession("skill.onnx")


def to_numpy(tensor):
    return tensor.detach().cpu().numpy() if tensor.requires_grad else tensor.cpu().numpy()


def is_ready(img):
    result = is_ready_torch(img)
    x = result[0][0]
    smax = np.exp(x) / sum(np.exp(x))
    print(smax, label[np.argmax(smax)])
    return smax[3] > 0.85


def is_ready_torch(img):
    image = transformations(img).float()
    image = image.unsqueeze_(0)
    ort_inputs = {ort_session.get_inputs()[0].name: to_numpy(image)}
    return ort_session.run(None, ort_inputs)


img = Image.open("debug.jpg")
print(is_ready(img))
