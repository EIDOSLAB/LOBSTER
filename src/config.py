from torch import nn

# ArgumentParser configs
AVAILABLE_MODELS = ["lenet300", "lenet5", "resnet32", "resnet18", "resnet101", "alexnet", "vgg", "isic-classification", "segnet", "unet"]
AVAILABLE_DATASETS = ["mnist", "fashion-mnist", "cifar10", "imagenet", "cifar100", "isic-classification", "isic-segmentation"]
AVAILABLE_SENSITIVITIES = ["", "lobster"]

# Logs root directory
LOGS_ROOT = "logs"

# Layers considered during regularization and pruning
LAYERS = (nn.modules.Linear, nn.modules.Conv2d, nn.modules.BatchNorm2d)
