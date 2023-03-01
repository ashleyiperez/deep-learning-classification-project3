#TODO: Import your dependencies.
#For instance, below are some dependencies you might need if you are using Pytorch
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F #from tutorial here: https://resbyte.github.io/posts/2017/08/pytorch-tutorial/
import torchvision
import torchvision.models as models
import torchvision.transforms as transforms
import smdebug.pytorch as smd # from Deploy Deep Learning Models example for Sagemaker Debugger
from smdebug.pytorch import get_hook

import argparse
import os
#referencing examples from classmates in 'tech-help' slack channel
from tqdm import tqdm
from PIL import ImageFile
ImageFile.LOAD_TRUNCATED_IMAGES = True

#referencing pytorch_mnist.py from 'hyperparameter tuning example files' provided in course

def test(model, test_loader, hook):
    hook = get_hook(create_if_not_exists=True)
    '''
    TODO: Complete this function that can take a model and a 
          testing data loader and will get the test accuray/loss of the model
          Remember to include any debugging/profiling hooks that you might need
    '''
    #referencing mnist.py from 'model deployment example files'
    #model.to("cpu")
    model.eval()
    
    hook.set_mode(smd.modes.EVAL)
    
    correct = 0
    total = 0
    
    #from Introduction to Neural Networks lesson, Training a Neural Network example
    with torch.no_grad():
        for data, target in test_loader:
            output=model(data)
            _, pred= torch.max(output.data, 1)
            total += target.size(0)
            correct += (pred == target).sum().item()
            
            hook.save_scalar("accuracy", correct/total)
            
        print(f"'accuracy': {100 * correct/total}%")
        
    
    return hook
    

def train(model, train_loader, criterion, optimizer, epochs, hook):
    hook = get_hook(create_if_not_exists=True)
    '''
    TODO: Complete this function that can take a model and
          data loaders for training and will get train the model
          Remember to include any debugging/profiling hooks that you might need
    '''
    
    model.train()
    
    hook.set_mode(smd.modes.TRAIN)
    hook.register_loss(criterion)
    
    #referencing example for common architechtural types and fine-tuning, Fine-Tuning a CNN Model Example
    for e in range(epochs + 1):
        running_loss=0
        correct=0
        for i, (data, target) in enumerate(train_loader):
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
            
        print(f"Epoch {e}: Loss {running_loss/len(train_loader.dataset)}, \
                Accuracy {100*(correct/len(train_loader.dataset))}%")
        #referencing example from here: https://docs.aws.amazon.com/sagemaker/latest/dg/debugger-modify-script-pytorch.html
        hook.save_scalar("loss", loss)
        
    return hook
    
def net():
    '''
    TODO: Complete this function that initializes your model
          Remember to use a pretrained model
    '''
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

def main(args):
    '''
    TODO: Initialize a model by calling the net function
    '''
    #referencing operationalizing ml model examples
    train_loader, test_loader=create_data_loaders(args.data, args.batch_size)
    
    model=net()
    
    #initializing hook
    hook = smd.Hook(out_dir="s3://sagemaker-us-east-1-425636437011/project-hpo-output/") #Here
    #hook = smd.Hook.create_from_json_file()
    hook.register_hook(model)
    '''
    TODO: Create your loss and optimizer
    '''
    loss_criterion = nn.CrossEntropyLoss()
    optimizer = torch.optim.Adam(model.parameters(), lr =args.lr)
    
    #for testing purposes
    data = args.data
    batch_size = args.batch_size
    epochs = args.epochs
    test_data = args.test_data
    test_batch_size = args.test_batch_size
    
    '''
    TODO: Call the train function to start training your model
    Remember that you will need to set up a way to get training data from S3
    '''
    #train_loader = create_data_loaders(data, batch_size)
    train(model, train_loader, loss_criterion, optimizer, epochs, hook)
    
    '''
    TODO: Test the model to see its accuracy
    '''
    #test_loader = create_data_loaders(test_data, test_batch_size)
    test(model, test_loader, hook)
    
    #for epochs in range(1, args.epochs + 1):
    #    train(model, train_loader, loss_criterion, optimizer, epochs, hook)
    #    test(model, test_loader, hook)
    
    '''
    TODO: Save the trained model
    '''
    path = "dog_classifier_hpo.pt"
    #referencing examples in 'tech-help' slack channel from course and operationalizing machine learing lesson examples
    torch.save(model.cpu().state_dict(), os.path.join(args.model_dir, path))

if __name__=='__main__':
    parser=argparse.ArgumentParser()
    '''
    TODO: Specify all the hyperparameters you need to use to train your model.
    '''
    parser.add_argument(
        "--batch_size",
        type=int,
        default=7,
    )
    parser.add_argument(
        "--test_batch_size",
        type=int,
        default=7,
    )
    parser.add_argument(
        "--epochs",
        type=int,
        default=1,
    )
    parser.add_argument(
        "--lr", type=float, default=0.1,
    )
    #referencing examples in 'tech-help' slack channel from course and operationalizing machine learing lesson examples
    parser.add_argument('--data', type=str, default=os.environ['SM_CHANNEL_TRAINING'])
    parser.add_argument('--test_data', type=str, default=os.environ['SM_CHANNEL_TEST'])
    parser.add_argument('--model_dir', type=str, default=os.environ['SM_MODEL_DIR'])
    parser.add_argument('--output_dir', type=str, default=os.environ['SM_OUTPUT_DATA_DIR'])
    
    args=parser.parse_args()
    
    main(args)
