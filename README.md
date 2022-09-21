# Interpretable Machine Learning approach to predict discharge eighteen-item Functional Independence Measure (FIM) scores for Stroke Rehabilitation

**Background and Purpose**
Stroke is the leading cause of disability in the United States. Rehabilitation is vital in stroke for recovery. Functional Independence Measure (FIM) is a validated survey instrument comprising of an eighteen-item, seven-level ordinal scale measured at the time of admission and discharge from the rehabilitation center. Predicting all individual 18 items at the time of admission to the rehabilitation center, although difficult, can help plan a better personalized rehabilitation program and answer the expectations of patients and their families. Explaining the individual item predictions at the patient level can help identify the primary outcome predictors and further individualize the rehabilitation plan. 

**Methods**
The study population consisted of retrospectively collected data from 803 patients (52% male, 45% Caucasian, 18% African American, 79% ischemic stroke) admitted to Memorial Hermann Comprehensive Stroke Center, Houston, Texas, USA. FIM score comprises of 18 items containing ordinal values making it a multioutput regression problem. Popular machine learning and deep learning models like chained Bayesian Ridge Regression, XGBoost, Lightgbm, Random Forest, TabNet were developed. The models were tuned using ree-structured Parzen Estimator algorithm.  SHAP (Shapley Additive explanations) values were obtained to explain the predictions.

**Results**
Predictions for all 18 individual items in FIM were obtained. The best-performing model was a chained regression model using Bayesian ridge regression. The uniform mean absolute error for all 18 items was 0.80. Patient-level and population-level interpretability was obtained with the help of SHAP values.

**Conclusion**
Our findings strongly suggest that although predicting individual items in the FIM instrument is challenging, it can be done  using state-of-the-art machine learning models. The predictions, along with the explanations, can help develop a personalized rehabilitation plan.  

# Code
Due to HIPAA rules for patient data, no code or data can be shared. 

# Packages
- [scikit-learn](https://scikit-learn.org/stable/) (Machine learning library to develop chained Bayesian ridge regression )
- [PyTorch](https://pytorch.org/) (An open source deep learning platform) 
- [Optuna](https://github.com/optuna/optuna/tree/5000dbe185aed6c65a7a07dff41a4c9f000ec52a) (Automatic hyperparameter optimization software framework)
- [SHAP](https://shap.readthedocs.io/en/latest/index.html) (SHAP (SHapley Additive exPlanations) is a game theoretic approach to explain the output of any machine learning model.)




**So, here’s a simple pytorch template that help you get into your main project faster and just focus on your core (Model Architecture, Training Flow, etc)**

In order to decrease repeated stuff, we recommend to use a high-level library. You can write your own high-level library or you can just use some third-part libraries such as [ignite](https://github.com/pytorch/ignite), [fastai](https://github.com/fastai/fastai), [mmcv](https://github.com/open-mmlab/mmcv) … etc. This can help you write compact but full-featured training loops in a few lines of code. Here we use ignite to train mnist as an example.

# Requirements
- [yacs](https://github.com/rbgirshick/yacs) (Yet Another Configuration System)
- [PyTorch](https://pytorch.org/) (An open source deep learning platform) 
- [ignite](https://github.com/pytorch/ignite) (High-level library to help with training neural networks in PyTorch)

# Table Of Contents
-  [In a Nutshell](#in-a-nutshell)
-  [In Details](#in-details)
-  [Future Work](#future-work)
-  [Contributing](#contributing)
-  [Acknowledgments](#acknowledgments)

# In a Nutshell   
In a nutshell here's how to use this template, so **for example** assume you want to implement ResNet-18 to train mnist, so you should do the following:
- In `modeling`  folder create a python file named whatever you like, here we named it `example_model.py` . In `modeling/__init__.py` file, you can build a function named `build_model` to call your model

```python
from .example_model import ResNet18

def build_model(cfg):
    model = ResNet18(cfg.MODEL.NUM_CLASSES)
    return model
``` 

   
- In `engine`  folder create a model trainer function and inference function. In trainer function, you need to write the logic of the training process, you can use some third-party library to decrease the repeated stuff.

```python
# trainer
def do_train(cfg, model, train_loader, val_loader, optimizer, scheduler, loss_fn):
 """
 implement the logic of epoch:
 -loop on the number of iterations in the config and call the train step
 -add any summaries you want using the summary
 """
pass

# inference
def inference(cfg, model, val_loader):
"""
implement the logic of the train step
- run the tensorflow session
- return any metrics you need to summarize
 """
pass
```

- In `tools`  folder, you create the `train.py` .  In this file, you need to get the instances of the following objects "Model",  "DataLoader”, “Optimizer”, and config
```python
# create instance of the model you want
model = build_model(cfg)

# create your data generator
train_loader = make_data_loader(cfg, is_train=True)
val_loader = make_data_loader(cfg, is_train=False)

# create your model optimizer
optimizer = make_optimizer(cfg, model)
```

- Pass the all these objects to the function `do_train` , and start your training
```python
# here you train your model
do_train(cfg, model, train_loader, val_loader, optimizer, None, F.cross_entropy)
```

**You will find a template file and a simple example in the model and trainer folder that shows you how to try your first model simply.**


# In Details
```
├──  config
│    └── defaults.py  - here's the default config file.
│
│
├──  configs  
│    └── train_mnist_softmax.yml  - here's the specific config file for specific model or dataset.
│ 
│
├──  data  
│    └── datasets  - here's the datasets folder that is responsible for all data handling.
│    └── transforms  - here's the data preprocess folder that is responsible for all data augmentation.
│    └── build.py  		   - here's the file to make dataloader.
│    └── collate_batch.py   - here's the file that is responsible for merges a list of samples to form a mini-batch.
│
│
├──  engine
│   ├── trainer.py     - this file contains the train loops.
│   └── inference.py   - this file contains the inference process.
│
│
├── layers              - this folder contains any customed layers of your project.
│   └── conv_layer.py
│
│
├── modeling            - this folder contains any model of your project.
│   └── example_model.py
│
│
├── solver             - this folder contains optimizer of your project.
│   └── build.py
│   └── lr_scheduler.py
│   
│ 
├──  tools                - here's the train/test model of your project.
│    └── train_net.py  - here's an example of train model that is responsible for the whole pipeline.
│ 
│ 
└── utils
│    ├── logger.py
│    └── any_other_utils_you_need
│ 
│ 
└── tests					- this foler contains unit test of your project.
     ├── test_data_sampler.py
```


# Future Work

# Contributing
Any kind of enhancement or contribution is welcomed.


# Acknowledgments



