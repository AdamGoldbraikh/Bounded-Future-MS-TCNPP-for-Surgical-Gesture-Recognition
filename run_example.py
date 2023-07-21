from utils.efficientnetV2 import EfficientnetV2
from utils.transforms import GroupNormalize, GroupScale, GroupCenterCrop
from model import MST_TCN2
import torch
from torchvision import transforms
from FrameGenrator import FrameGenerator
from project import run

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# load model
num_classes_list = [6]
model = MST_TCN2(10, 10, 3, 128, 1280, num_classes_list,
                 dropout=0.5, w_max=3, offline_mode=False)
model.load_state_dict(torch.load(
    "examples/3_seconds_delay/MS-TCN2_gestures.model"))
model.eval()
model = model.to(device)

# load extractor
path = "examples/extractor.pth"
extractor = EfficientnetV2(
    size="m", num_classes=6, pretrained=False)  # load extractor
extractor.load_state_dict(torch.load(path))
extractor = extractor.eval()

# define the normaliztion and the augmentations
mean, std = extractor.input_mean, extractor.input_std
frame_gen = FrameGenerator("examples/frames")
normalize = GroupNormalize(mean, std)
val_augmentation = transforms.Compose([GroupScale(int(256)),
                                       GroupCenterCrop(224)])
shape = (224, 224)

# runner is a generator
runner = run(frame_gen, model, extractor, normalize,
             val_augmentation, use_extractions=True, shape=shape)

outs = []
for i, output in enumerate(runner):
    outs.append(output)
    # `output` is the output of the model at time `i`

# convert to tensor, this is the exact output of the model
outputs = []
for i in range(len(num_classes_list)):
    predictions = torch.vstack([o[i] for o in outs])
    outputs.append(predictions)
# outputs will be the exact output of the model
