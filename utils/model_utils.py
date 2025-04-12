import torch
from torchvision import models, transforms
import torch.nn as nn

class ResNet50Unskip(nn.Module):
    def __init__(self):
        super(ResNet50Unskip, self).__init__()
        base = models.resnet50(weights=None)
        self.features = nn.Sequential(*list(base.children())[:-2])
        self.pool = nn.AdaptiveAvgPool2d((1, 1))
        self.fc = nn.Linear(2048, 7)

    def forward(self, x):
        x = self.features(x)
        x = self.pool(x)
        x = torch.flatten(x, 1)
        x = self.fc(x)
        return x

labels = ['akiec', 'bcc', 'bkl', 'df', 'nv', 'vasc', 'mel']

def load_model(path):
    model = ResNet50Unskip()
    model.load_state_dict(torch.load(path, map_location=torch.device('cpu')))
    model.eval()
    return model

def predict_image(model, image):
    transform = transforms.Compose([
        transforms.Resize((224, 224)),
        transforms.ToTensor(),
        transforms.Normalize([0.5]*3, [0.5]*3)
    ])
    img_tensor = transform(image).unsqueeze(0)
    with torch.no_grad():
        outputs = model(img_tensor)
        _, pred = torch.max(outputs, 1)
    return pred.item(), labels[pred.item()]
