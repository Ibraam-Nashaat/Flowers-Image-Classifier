import torch
from torch import nn
from utility_functions import*
from model_functions import*

arg= get_input_args()

checkpoint= arg.load_dir

model = load_checkpoint(checkpoint, arg.gpu)

top_p,top_class= predict(arg.image_path, model,arg.gpu, int(arg.top_k))

names= label_mapping(arg.category_names, top_class)

print_predictions(top_p,names)