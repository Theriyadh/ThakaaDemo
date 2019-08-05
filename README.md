# ThakaaDemo
A simple custom demo in video classification and it is intended to purpose a addetional document to recruitment process and it does not serve or evaluate the technical and managerial ability.  

## Getting Started
 IMPORTANT DOWNLOAD FULL FOLDER HERE: https://mega.nz/#F!FTJgnYYA!FnijsKnSEHXCHacDrgKE7w
 
The video classification are using Keras (Tensorflow Backend) with some of applying Neural network architectures. The demo was inspired by (Large-scale Video Classification with Convolutional Neural Networks Research by Google and Stanford Univeristy) Hoever implementation is quite different (Paper https://cs.stanford.edu/people/karpathy/deepvideo/)


## Introduction

1. The demo is combination by human activity classification (Football, Basketball) and Leagues and clubs classification. 
2. The dataset was personally built on previous videos and frames were extracted (~3500 pictures) 
3. The dataset was split into 4 Different Classes;
```
 A. Serie A: Ac Milan 
 B. Serie B: Inter Milan 
 C. Saudi League: Alnasser 
 D. NBA: Golden State Worriers
 ```
4. The methodology used was quite simple:
```
 A. for all the frames in a video a loop is made
 B. Then for each frame, the photo is passed to CNN
 C. CNN will make a prediction
 D. a list of prediction will be maintained 
 E. Compute the average of the predictions of each class and label the highest number
 ```
5.  A fine-tuned classifier (ResNet50) for recognizing scenes and clubs (However would be intersting to see resullts by LSTMs.
6. The model has reached ~88-89% accuracy after fine-tuning ResNet50 on the dataset.
7. To try the model please dont forget to change the path when cloning the repository


## Prerequisites

Libraried needed to be installed...

```
1. Tensorflow
2.keras.models 
3. numpy
4. argparse
5. pickle
6. cv2
7. collections
```


## Running the tests

Please simply run the Demo.py (after installing and changing the path if necessary) 


## Conclusion 

Let me know at anytime if you needed help with installtion or running the model as I have provided a result sample

## Versioning

0.1

## Authors

* **Riyadh Alkhanin** - BEng MIET CBE



