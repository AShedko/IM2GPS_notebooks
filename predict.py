import torch
import torchvision.transforms as transforms
import sys
model = torch.load("./models/resnet50_trained").to(torch.device("cpu"))
tf = transforms.Compose([
        transforms.Resize(256),
        transforms.CenterCrop(224),
        transforms.ToTensor(),
        transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
    ])

def predict(img):
    img = tf(img).unsqueeze(0)
    return model(img).max(1)[1].numpy()[0]
                    # argmax then [number from tensor]
