import argparse
import os


def str2bool(v):
    return v.lower() in ("yes", "true", "t", "1")


data = "/data/"


parser = argparse.ArgumentParser(
    description="Train model for video-based surgical gesture recognition.")
parser.register('type', 'bool', str2bool)

# Experiment
parser.add_argument('--exp', type=str, default="experiment",
                    help="Name (description) of the experiment to run.")
parser.add_argument('--video_suffix', type=str,
                    choices=['_side', '_top'], default='_side')
parser.add_argument('--image_tmpl', default='img_{:05d}.jpg')

# Data
parser.add_argument('--epoch_size', type=int, default=2400,
                    help="number of samples in a epoch ")

parser.add_argument('--data_path', type=str, default=os.path.join(data, 'VTS', "frames"),
                    help="Path to data folder, which contains the extracted images for each video. "
                         "One subfolder per video.")
parser.add_argument('--transcriptions_dir', type=str, default=os.path.join(data, "VTS", "transcriptions_gestures"),
                    help="Path to folder containing the transcription files (gesture annotations). One file per video.")
parser.add_argument('--video_sampling_step', type=int, default=6,
                    help="Describes how the available video data has been downsampled from the original temporal "
                         "resolution (by taking every <video_sampling_step>th frame).")
parser.add_argument('--do_horizontal_flip', type='bool', default=False,
                    help="Whether data augmentation should include a random horizontal flip.")
parser.add_argument('--corner_cropping', type='bool', default=True,
                    help="Whether data augmentation should include corner cropping.")

# Model
parser.add_argument('--num_classes', type=int,
                    default=6, help="Number of classes.")

parser.add_argument('--arch', type=str, default="EfficientnetV2", choices=['3D-ResNet-18', '3D-ResNet-50', "2D-ResNet-18", "EfficientnetV2"],
                    help="Network architecture.")
parser.add_argument('--use_resnet_shortcut_type_B', type='bool', default=False,
                    help="Whether to use shortcut connections of type B.")
parser.add_argument('--input_size', type=int, default=224,
                    help="Target size (width/ height) of each frame.")
# Training
parser.add_argument('--resume_exp', type=str, default=None,
                    help="Path to results of former experiment that shall be resumed (untested).")
parser.add_argument('-j', '--workers', type=int, default=16,
                    help="Number of threads used for data loading.")
parser.add_argument('--epochs', type=int, default=100,
                    help="Number of epochs to train.")
parser.add_argument('-b', '--batch-size', type=int,
                    default=32, help="Batch size.")
parser.add_argument('--lr', type=float, default=0.00025, help="Learning rate.")
parser.add_argument('--use_scheduler', type=bool, default=True,
                    help="Whether to use the learning rate scheduler.")
# parser.add_argument('--loss_weighting', type=bool, default=True,
#                     help="Whether to apply weights to loss calculation so that errors in more current predictions "
#                          "weigh more heavily.")
parser.add_argument('--eval_freq', '-ef', type=int, default=3,
                    help="Validate model every <eval_freq> epochs.")
parser.add_argument('--out', type=str, default="output",
                    help="Path to output folder, where all models and results will be stored.")
