#Importing Dependencies
import os
import json
import torch
import torchvision
import torchvision.models as models
import time
import os
import pip

def install(package):
    pip.main(['install', package])

#referencing example here: https://sagemaker-examples.readthedocs.io/en/latest/frameworks/pytorch/get_started_mnist_deploy.html
def net():
    '''
    TODO: Complete this function that initializes your model
          Remember to use a pretrained model
    '''
    install('smdebug')
    
    #Looking  at 'Deep Learning Topics in CV' lesson, 'Training a Neural Network' page examples and sagemaker debugger and profiling solution examples
    model=models.resnet18(pretrained=True)
    
    for param in model.parameters():
        param.requires_grad= False
        
    num_features=model.fc.in_features
    model.fc = nn.Sequential(
        nn.Linear(num_features, 133) #num of dog classes in dataset
    )
    
    return model

#referncing model deployement example files from 'Deep Learning, Deployment' Module and troublshooting in 'tech-help' slack channel
def model_fn(model_dir):
    model = net()
    with open(os.path.join(model_dir, "model.pth"), "rb") as f:
        model.load_state_dict(torch.load(f))
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    model.to(device).eval()
    return model

def input_fn(request_body, request_content_type):
    assert request_content_type=='application/json'
    data = json.loads(request_body)['inputs']
    data = torch.tensor(data, dtype=torch.float32, device=device)
    return data

def predict_fn(input_object, model):
    with torch.no_grad():
        prediction = model(input_object)
    return prediction

def output_fn(prediction, content_type):
    assert content_type == 'application/json'
    res = prediction.cpu().numpy().tolist()
    return json.dumps(res)
