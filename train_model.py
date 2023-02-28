#TODO: Import your dependencies.
#For instance, below are some dependencies you might need if you are using Pytorch
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader #referencing examples here: https://pytorch.org/tutorials/beginner/basics/data_tutorial.html
import torchvision
import torchvision.models as models
import torchvision.transforms as transforms
import argparse
import time
import os
import pip

#TODO: Import dependencies for Debugging andd Profiling
#import smdebug.pytorch as smd
#from smdebug.pytorch import get_hook

from tqdm import tqdm
from PIL import ImageFile
ImageFile.LOAD_TRUNCATED_IMAGES = True

#referencing 'tech-help' slack channel solution for resnet18 smd bug
import pip

def install(package):
    pip.main(['install', package])
    
#Reference Transfer learning documentation: https://pytorch.org/tutorials/beginner/transfer_learning_tutorial.html
#Reference Fine Tuning TorchVision Models documentation: https://pytorch.org/tutorials/beginner/finetuning_torchvision_models_tutorial.html

def test(model, test_loader, hook):
    import smdebug.pytorch as smd
    from smdebug.pytorch import get_hook
    
    hook = get_hook(create_if_not_exists=True)
    
    '''
    TODO: Complete this function that can take a model and a 
          testing data loader and will get the test accuray/loss of the model
          Remember to include any debugging/profiling hooks that you might need
    '''
    #Looking  at 'Deep Learning Topics in CV' lesson, 'Training a Neural Network' page examples and sagemaker debugger and profiling solution examples
    model.eval()
    #hook for debugging
    hook.set_mode(smd.modes.EVAL)
    
    #get test accuracy/loss
    total = 0
    correct = 0
    
    #from Introduction to Neural Networks lesson, Training a Neural Network example
    with torch.no_grad():
        
        for data, target in test_loader:
            output=model(data)
            _, pred= torch.max(output.data, 1)
            total += target.size(0)
            correct += (pred == target).sum().item()
            
            hook.save_scalar("accuracy", correct/total) #accuracy points should equal number of (image/files)/batch size
            
        print(f"'accuracy': {100 * correct/total}%")
        
    return hook

def train(model, train_loader, criterion, optimizer, epochs, hook): 
    import smdebug.pytorch as smd
    from smdebug.pytorch import get_hook
    
    hook = get_hook(create_if_not_exists=True)
    
    '''
    TODO: Complete this function that can take a model and
          data loaders for training and will get train the model
          Remember to include any debugging/profiling hooks that you might need
    '''
    model.train()
    #hook for debugging
    hook.set_mode(smd.modes.TRAIN)
    hook.register_loss(criterion)
    
    #from sagemaker debugger example files
    #for e in range(epochs):
    running_loss=0
    correct=0
        #hook.set_mode(smd.modes.TRAIN)
        #hook.register_loss(criterion)
        
    for i, (data, target) in enumerate(train_loader):
        hook.set_mode(smd.modes.TRAIN)
        optimizer.zero_grad()
        pred = model(data)             
        loss = criterion(pred, target)
        running_loss+=loss
        loss.backward()
        optimizer.step()
        pred=pred.argmax(dim=1, keepdim=True)
        correct += pred.eq(target.view_as(pred)).sum().item()
            
        hook.save_scalar("accuracy", (correct/len(train_loader.dataset)))
        hook.save_scalar("loss", loss)
            
    print(f"Epoch {epochs}: Loss {running_loss/len(train_loader.dataset)}, \
                Accuracy {100*(correct/len(train_loader.dataset))}%")
        
        
    return model, hook
    
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

def create_data_loaders(data, batch_size):
    '''
    This is an optional function that you may or may not need to implement
    depending on whether you need to use data loaders or not
    '''
    #referencing 'introduction to deep learning' lesson in 'training a neural network' example and session lead notes from connect session 8 (2/10/23)
    data = args.data
    test_data = args.test_data
    test_batch_size = args.test_batch_size

    train_transform = transforms.Compose([
        transforms.RandomResizedCrop((224, 224)),
        transforms.RandomHorizontalFlip(),
        transforms.ToTensor(),
        ])

    test_transform = transforms.Compose([
        transforms.Resize((224, 224)),
        transforms.ToTensor(),
        ])

    train_data = torchvision.datasets.ImageFolder(root=data, transform=train_transform)
    train_data_loader = torch.utils.data.DataLoader(train_data, batch_size=batch_size, shuffle=True)

    test_data = torchvision.datasets.ImageFolder(root=test_data, transform=test_transform)
    test_data_loader  = torch.utils.data.DataLoader(test_data, batch_size=test_batch_size, shuffle=True)

    
    return train_data_loader, test_data_loader

#referncing model deployement example files from 'Deep Learning, Deployment' Module and troublshooting in 'tech-help' slack channel
def model_fn(model_dir):
    model = net()
    with open(os.path.join(model_dir, "model.pth"), "rb") as f:
        model.load_state_dict(torch.load(f))
    return model


#def save_model(model, model_dir):
#    logger.info("Saving the model.")
#    path = os.path.join(model_dir, "model.pth")
#    torch.save(model.cpu().state_dict(), path)

def main(args):
    '''
    TODO: Initialize a model by calling the net function
    '''
    #for testing purposes
    data = args.data
    batch_size = args.batch_size
    epochs = args.epochs
    test_data = args.test_data
    test_batch_size = args.test_batch_size
    
    #referencing operationalizing ml model examples
    train_loader, test_loader=create_data_loaders(args.data, args.batch_size)
    
    model=net()
    
    import smdebug.pytorch as smd
    from smdebug.pytorch import get_hook
    
    #initializing hook
    #hook = smd.Hook(out_dir=) #Here
    hook = smd.Hook.create_from_json_file()
    #hook.register_module(model)
    '''
    TODO: Create your loss and optimizer
    '''
    loss_criterion = nn.CrossEntropyLoss()
    optimizer = torch.optim.Adam(model.parameters(), lr =args.lr)
    
    hook.register_module(model)
    hook.register_loss(loss_criterion)
    
    '''
    TODO: Call the train function to start training your model
    Remember that you will need to set up a way to get training data from S3
    '''
    #train(model, train_loader, loss_criterion, optimizer, epochs, hook)
    
    '''
    TODO: Test the model to see its accuracy
    '''
    #test(model, test_loader, hook)
    
    for epochs in range(1, args.epochs):
        train(model, train_loader, loss_criterion, optimizer, epochs, hook)
        test(model, test_loader, hook)
    
    '''
    TODO: Save the trained model
    '''
    #path = "model.pt"
    #referencing examples in 'tech-help' slack channel from course and operationalizing machine learing lesson examples
    #torch.save(model.cpu().state_dict(), os.path.join(args.model_dir, path))
    
    with open(os.path.join(args.model_dir, 'model.pth'), 'wb') as path:
        torch.save(model.cpu().state_dict(), path)

if __name__=='__main__':
    parser=argparse.ArgumentParser()
    '''
    TODO: Specify any training args that you might need
    '''
    parser.add_argument(
        "--batch_size",
        type=int,
        default=128,
    )
    parser.add_argument(
        "--test_batch_size",
        type=int,
        default=100,
    )
    parser.add_argument(
        "--epochs",
        type=int,
        default=3,
    )
    parser.add_argument(
        "--lr", type=float, default=0.001,
    )
    #referencing examples in 'tech-help' slack channel from course and operationalizing machine learing lesson examples
    parser.add_argument('--data', type=str, default=os.environ['SM_CHANNEL_TRAIN'])
    parser.add_argument('--test_data', type=str, default=os.environ['SM_CHANNEL_TEST'])
    parser.add_argument('--model_dir', type=str, default=os.environ['SM_MODEL_DIR'])
    #parser.add_argument('--output_dir', type=str, default=os.environ['SM_OUTPUT_DATA_DIR'])
    
    args=parser.parse_args()
    
    main(args)
