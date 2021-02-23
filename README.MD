# Transfer Learning with PyTorch, Model Interpretability with Captum, and Interactive Dashboard with Dash

This repository contains code that
1. Builds an image classifier on the flowers dataset using PyTorch. Two notebooks are included, 1 for finetuning ResNet (ImageRetraining_FineTuning.ipynb) and 1 for using ResNet as feature extractor (ImageRetraining_FeatureExtraction.ipynb). The resulting trained models are saved locally and used for the next 2 steps.
2. Interpret the saved model above using Captum (Captum.ipynb). This notebooks covers a few methods to understand which features the CNN is looking to generate the classification. 
3. An interactive dashboard (dash_classifier.py) that allows an user to upload an image, and generate the classification of the image generated in step 1. 

## Download the data 
1. 

## Replicate environment
You can clone the environment using Conda from the included environment.yml file:
```
conda env create -f environment.yml
```

## Deploy as a Dashboard


## Author
- Jiangwei Yao

## License 
Apache License 2.0

