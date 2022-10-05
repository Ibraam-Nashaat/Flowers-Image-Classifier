# Flowers-Image-Classifier
- The project represents **Flowers Image Classifier** that can classify 102 different types of flowers from their images using PyTorch and transfer learning.
- The dataset used in this project can be found [here](https://s3.amazonaws.com/content.udacity-data.com/nd089/flower_data.tar.gz).
- This project was done as a part of Udacity's **AI Programming with Python Nanodegree**.

## Parts of the project
- The project can be separated into two parts:
  1. Jupyter Notebook:
       - We will implement the model in a jupyter notebook to experiment different models `.ipynb`.
         - Flowers Image Classifier.ipynb
      
  2. Python Command Line Application: 
       - We will convert the code to python application that run from the command line to be embedded in other applications `.py`.
          - train.py
          - predict.py
          - model_functions.py
          - utility_functions.py
  
  **Note**: `cat_to_name.json` contains a dictionary mapping the integer encoded categories to the actual names of the flowers
       
## Dependencies
- PyTorch 0.4.0 
- Torchvision 0.2.1
- Numpy
- MatPlotLib
- PIL
- json

**Note** : Using the values of hyperparameters and learning rate written in this project with VGG16 model will result in 92% accuracy.
