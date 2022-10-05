import torch
from torch import nn
from utility_functions import*
from model_functions import*

arg= get_input_args()
testing_arg_parser(arg)

train_dir = arg.data_dir + '/train'
valid_dir = arg.data_dir + '/valid'
test_dir = arg.data_dir + '/test'

train_loader, validation_loader, test_loader, train_dataset,validation_dataset= load_datasets(train_dir,valid_dir,test_dir)

model= create_model(int(arg.hidden_units), arg.archi)

model, device, criterion, optimizer= prepare_train( arg.gpu, arg.learning_rate, model)

model= train_model(model, int(arg.epochs), train_loader, validation_loader, device, criterion, optimizer)

test_model(model, device, criterion, test_loader)

checkpoint= arg.archi+"_"+arg.save_dir+".pth" 

save_model(model,checkpoint,arg.archi,train_dataset,optimizer)


