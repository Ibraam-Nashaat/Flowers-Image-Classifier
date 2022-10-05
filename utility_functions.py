import argparse
from torchvision import datasets, transforms
import torch
import numpy as np
import json
from PIL import Image


def get_input_args():
   
    # Create Parse using ArgumentParser
    parser= argparse.ArgumentParser()
    parser.add_argument('--data_dir',type=str,default='flowers',help='path to the data')
    parser.add_argument('--learning_rate',type=str,default='0.0001',help='learning rate of the optimizer')
    parser.add_argument('--archi',type=str,default='vgg16',choices=['vgg16', 'vgg19'],help='model architecture')
    parser.add_argument('--save_dir',type=str,default='checkpoint',help='directory to save the model')
    parser.add_argument('--epochs',type=str,default='10',help='No. of epochs')
    parser.add_argument('--gpu',type=str,default='True',choices=['true', 'false'],help='use gpu or not')
    parser.add_argument('--hidden_units',type=str,default='4096',help='no of hidden units')
    parser.add_argument('--top_k',type=str,default='5',help='number of top classes displayed')
    parser.add_argument('--image_path',type=str,default='flowers/test/13/image_05775.jpg',help='the path of image to predict')
    parser.add_argument('--category_names',type=str,default='cat_to_name.json',help='mapping the category to name')
    parser.add_argument('--load_dir',type=str,default='vgg16_checkpoint.pth',choices=['vgg16_checkpoint.pth', 'vgg16_checkpoint19.pth'],help='directory to load the model')

    return parser.parse_args()


def load_datasets(train_dir, valid_dir, test_dir ):
    print("\nLoading Datasets")
    train_transforms= transforms.Compose([transforms.RandomRotation(30),
                                          transforms.RandomResizedCrop(224),
                                          transforms.RandomHorizontalFlip(),
                                          transforms.ToTensor(),
                                          transforms.Normalize([0.485, 0.456, 0.406],
                                                               [0.229, 0.224, 0.225])])
    validation_transforms = transforms.Compose([transforms.Resize(255),
                                                transforms.CenterCrop(224),
                                                transforms.ToTensor(),
                                                transforms.Normalize([0.485, 0.456, 0.406],
                                                               [0.229, 0.224, 0.225])])
    test_transforms= validation_transforms

    # Load the datasets with ImageFolder
    train_dataset = datasets.ImageFolder(train_dir,transform=train_transforms)
    validation_dataset= datasets.ImageFolder(valid_dir,transform= validation_transforms)
    test_dataset= datasets.ImageFolder(test_dir,transform= test_transforms)

    # Using the image datasets and the trainforms, define the dataloaders
    train_loader = torch.utils.data.DataLoader(train_dataset,batch_size=64,shuffle=True)
    test_loader=torch.utils.data.DataLoader(test_dataset,batch_size=64,shuffle=True)
    validation_loader=torch.utils.data.DataLoader(validation_dataset,batch_size=64,shuffle=True)
    
    return train_loader, validation_loader, test_loader, train_dataset,validation_dataset


def process_image(image):
    ''' Scales, crops, and normalizes a PIL image for a PyTorch model,
        returns an Numpy array
    '''
    
    # Process a PIL image for use in a PyTorch model
    image= image.resize((256,256))
    crop= transforms.CenterCrop(224)
    image=crop(image)
    image=np.array(image)/255
    mean=np.array([0.485, 0.456, 0.406])
    std=np.array([0.229, 0.224, 0.225])
    image= (image-mean)/std
    image=image.transpose((2, 0, 1))
    return image

def testing_arg_parser(arg):
    print( "\nData Directory: "+ arg.data_dir +
            "\nLearning Rate: "+ arg.learning_rate +
            "\nModel architecture: "+ arg.archi +
            "\nSave Directory: "+arg.save_dir +
            "\nEpochs: "+arg.epochs +
            "\nGpu Status: "+arg.gpu +
            "\nHidden Units: "+arg.hidden_units)
    
def label_mapping(map_path, top_class):
    print("\nLabel Mapping")
    with open(map_path, 'r') as f:
        cat_to_name = json.load(f)
    
    names=[]
    for i in top_class:
        names.append(cat_to_name[str(i)])
        
    return names

def print_predictions(top_p,names):
    print("\nPrinting Predictions: ")
    for i in range(len(top_p)):
        print(f"top {i+1} :   flower name : {names[i]} , probability: {top_p[i]:.5f}")