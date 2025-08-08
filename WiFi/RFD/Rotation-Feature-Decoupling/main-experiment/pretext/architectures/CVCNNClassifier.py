from torch import nn

class CVCNNClassifier(nn.Module):
    def __init__(self, opt):
        super(CVCNNClassifier, self).__init__()
        self.fc_classifier = nn.LazyLinear(opt['num_classes'])

    def forward(self, x):
        x = self.fc_classifier(x)
        return x


def create_model(opt):
    return CVCNNClassifier(opt)
