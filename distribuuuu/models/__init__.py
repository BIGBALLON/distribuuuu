from distribuuuu.models.alexnet import alexnet  # noqa
from distribuuuu.models.densenet import *  # noqa
from distribuuuu.models.resnet import *  # noqa


def build_model(arch, **kwargs):
    return globals()[arch](**kwargs)
