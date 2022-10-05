import torch
from torch import nn
from torchvision import models
from torch import optim
from utility_functions import process_image
from PIL import Image
import numpy as np

def create_model(hidden_units, archi):
    print("\nCreating the Model")
    model_archi= {"vgg16":models.vgg16(pretrained=True), "vgg19": models.vgg19(pretrained=True)}
    
    model= model_archi[archi]
    for param in model.parameters():
        param.requires_grad = False
    
    model.classifier = nn.Sequential(nn.Linear(25088, hidden_units),
                                 nn.ReLU(),
                                 nn.Linear(hidden_units, hidden_units),
                                 nn.ReLU(),
                                 nn.Linear(hidden_units, 102),
                                 nn.LogSoftmax(dim=1))
    return model


def prepare_train(gpu, learning_rate, model):
    print("\nPreparing the Model for Training")
    device = torch.device("cuda" if torch.cuda.is_available() and gpu.lower()=="true" else "cpu")
    criterion= nn.NLLLoss()
    optimizer= optim.Adam(model.classifier.parameters(),lr= float(learning_rate))
    model.to(device)
    return model, device, criterion, optimizer


def train_model(model,epochs,train_loader,validation_loader,device,criterion, optimizer):
    print("\nTraining the Model")
    for epoch in range(epochs):
        epoch_loss=0
        train_accuracy=0
        for images,labels in train_loader:
            images, labels= images.to(device), labels.to(device)
            logps= model(images)
            loss= criterion(logps,labels)

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            epoch_loss+=loss.item()
            ps=torch.exp(logps)
            _,top_class= ps.topk(1,dim=1)
            equals= top_class==labels.view(*top_class.shape)
            train_accuracy += torch.mean(equals.type(torch.FloatTensor)).item()

        else:
            valid_loss=0
            valid_accuracy=0
            model.eval()
            with torch.no_grad():
                for images,labels in validation_loader:
                    images,labels = images.to(device),labels.to(device)
                    logps= model(images)
                    loss= criterion(logps,labels)
                    valid_loss+= loss.item()

                    ps=torch.exp(logps)
                    _,top_class= ps.topk(1,dim=1)
                    equals= top_class==labels.view(*top_class.shape)
                    valid_accuracy += torch.mean(equals.type(torch.FloatTensor)).item()


            print(f"Epoch {epoch+1}/{epochs}.. "
                      f"Train loss: {epoch_loss/len(train_loader):.3f}.. "
                      f"Validation loss: {valid_loss/len(validation_loader):.3f}.. "
                      f"Training accuracy: {train_accuracy/len(train_loader):.3f}.."
                      f"Validation accuracy: {valid_accuracy/len(validation_loader):.3f}")
            model.train()
            
            
    return model

        
def test_model(model, device, criterion, test_loader):
    print("\nTesting the Model")
    test_loss=0
    accuracy=0
    model.to(device)
    model.eval()
    with torch.no_grad():
        for images,labels in test_loader:
            images,labels = images.to(device),labels.to(device)
            logps= model(images)
            loss= criterion(logps,labels)
            test_loss+= loss.item()

            ps=torch.exp(logps)
            _,top_class= ps.topk(1,dim=1)
            equals= top_class==labels.view(*top_class.shape)
            accuracy += torch.mean(equals.type(torch.FloatTensor)).item()

    print(f"\nTest loss: {test_loss/len(test_loader):.3f}.. Test accuracy: {accuracy/len(test_loader):.3f}")

def save_model(model, checkpoint_path, archi, train_dataset,optimizer):
    checkpoint = {'input_size': model.classifier[0].in_features,
                  'output_size': 102,
                  'archi': archi,
                  'training_epochs': 10,
                  'class_to_idx':train_dataset.class_to_idx,
                  'hidden_layers': [ each.out_features for each in model.classifier if type(each)==torch.nn.modules.linear.Linear],
                  'state_dict': model.state_dict()}

    torch.save(checkpoint, checkpoint_path)
    print("\nModel Saved Successfully")
    
    
def load_checkpoint(filepath, gpu):
    map_location= "cuda" if torch.cuda.is_available() and gpu.lower()=="true" else "cpu"
    checkpoint = torch.load(filepath, map_location= map_location)
    
    archi={'vgg16':models.vgg16(pretrained=True), "vgg19": models.vgg19(pretrained=True)}
    model= archi[checkpoint['archi']]
    hidden_layers= checkpoint['hidden_layers']
    classifier= nn.ModuleList([nn.Linear(checkpoint['input_size'], hidden_layers[0]), nn.ReLU()])
    layer_sizes = zip(hidden_layers[:-1], hidden_layers[1:-1])
    for h1, h2 in layer_sizes:
        classifier.extend([nn.Linear(h1, h2),nn.ReLU()])
    classifier.extend([nn.Linear(hidden_layers[-2], checkpoint['output_size']),nn.LogSoftmax(dim=1)])
    model.classifier= nn.Sequential(*classifier)
    model.load_state_dict(checkpoint['state_dict'])
    model.epochs= checkpoint['training_epochs']
    model.class_to_idx= checkpoint['class_to_idx']
    print("\nModel Loaded Successfully")
    return model

def predict(image_path, model, gpu, topk=5):
    ''' Predict the class (or classes) of an image using a trained deep learning model.
    '''
    
    device = torch.device("cuda" if torch.cuda.is_available() and gpu.lower()=="true" else "cpu")
    # Implement the code to predict the class from an image file
    image = Image.open(image_path)
    print(image_path)
    image= process_image(image)
    image= np.expand_dims(image, axis=0)
    image= torch.from_numpy(image).float()
    model.to(device)
    image= image.to(device)
    model.eval()
    with torch.no_grad():
        logps= model(image)
    ps=torch.exp(logps)
    top_p,top_class= ps.topk(topk,dim=1)
    keys= list(model.class_to_idx.keys())
    values= list(model.class_to_idx.values())
    t_class=[]
    for i in top_class[0].cpu().numpy():
        t_class.append(int(keys[values.index(i)]))
    
    top_p= top_p[0].cpu().numpy()
    print("\nPrediction Completed")
    return top_p,t_class