import torch
import torch.nn as nn
import torch.nn.functional as F
import torchvision.transforms as T
import torchvision.models as models

device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

class Model(nn.Module):

    def __init__(self, name):
        super(Model, self).__init__()

        normalize = T.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
        self.model = self._init_model(name)
        self.transforms = T.Compose([T.ToTensor(),
                            normalize])

    def _init_model(self, name):

        if name == 'resnet18':
            model = models.resnet18(pretrained=True).eval()
        if name == 'resnet50':
            model = models.resnet18(pretrained=True).eval()
        
        model.fc = nn.Sequential()
        return model.to(device)

    def forward(self, x):
        """
        Args:
            x: patches with type numpy.ndarray(window, window, 3)

        Returns:
            x: extracted feature by CNN with type numpy.ndarray(feature_length, )
        """
        with torch.no_grad():
            x = self.transforms(x).unsqueeze(0).to(device)
            x = self.model(x)
            x = F.normalize(x)
            x = x.cpu().detach().squeeze().numpy()
        return x