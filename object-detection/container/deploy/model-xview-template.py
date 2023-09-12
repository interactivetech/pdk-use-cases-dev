from torchvision.models.detection.faster_rcnn import FasterRCNN
from torchvision.models.detection.backbone_utils import resnet_fpn_backbone

class FRCNNObjectDetector(FasterRCNN):
    # Template class that we will inject the number of classes for torch-model-archiver
    def __init__(self, num_classes={}, **kwargs):
        backbone = resnet_fpn_backbone('resnet50', True)
        super(FRCNNObjectDetector, self).__init__(backbone, num_classes, **kwargs)