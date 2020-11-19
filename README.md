# Identifying Melanoma in Lesion Images

#### Solution to the Kaggle 2020 [**SIIM-ISIC Melanoma Classification**](https://www.kaggle.com/c/siim-isic-melanoma-classification) Competition

<br>
<a href="https://githubtocolab.com/reyvaz/SIIM-ISIC-Melanoma-Identification-2020/blob/master/Melanoma_Identification_2020.ipynb" 
rel="see html report">
<img src="media/colab.png" alt="Drawing" width = "110">
</a>

[![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://githubtocolab.com/reyvaz/SIIM-ISIC-Melanoma-Identification-2020/blob/master/Melanoma_Identification_2020.ipynb)

<br>

This repo contains my solution to the [SIIM-ISIC Melanoma Classification](https://www.kaggle.com/c/siim-isic-melanoma-classification) competition hosted by [Kaggle](https://www.kaggle.com/), The Society for Imaging Informatics in Medicine ([SIIM](https://siim.org/)), and The International Skin Imaging Collaboration ([ISIC](https://www.isic-archive.com/)) that ended on August 17, 2020. The goal of the competition was to identify melanoma in images of skin lesions.

The evaluation metric used is the [area under the ROC curve](https://en.wikipedia.org/wiki/Receiver_operating_characteristic) between the predicted probability and the observed target. 

## Model

My final submission, which ranked top 11 percent in the competition with a private test AUC of 0.9363, consisted on an ensemble of 15 CNN classifiers as follows.

- 5 cross-validation splits were used to train 3 different EfficientNet based models. EfficientNet B3, B4, and B6.
- All classifiers were trained using Tensorflow Keras 2.x on Google Colab TPUs
- All used either imagenet or noisy-student pre-trained weights.
- Learning rate schedule with warm-up, constant, and exponential decay.
- All classifiers trained with either Rectified-Adam (Radam) or Adam with Nesterov Momentum (Nadam) optimizers. 
- Binary Cross-Entropy loss with 5% label smoothing was used throughout.


## Data

For training on TPU I used data (images, labels) contained in TFRecords created and made available by Chris Deotte. The training dataset consisted of the [official 2020 competition data](https://www.kaggle.com/cdeotte/melanoma-1024x1024) and I expanded it with the data from the [2018 competition](https://www.kaggle.com/cdeotte/isic2019-768x768). The data in these TFRecs was distributed so that:

- All images from a single patient are contained within a TFRecord.
- All TFRecords contain similar proportions of benign/malignant cases.
- Image counts, and image counts per patient are balanced across TFRecords. 

The TFRecords' data distribution features carried-over into the cross-validation splits.

TFRecords were processed using Tensorflow Dataset API to support TPU training. 

- Images were resized to match the original corresponding EfficientNet version size.
- Images were randomly augmented during training. Augmentations consisted of:
    - Zoom-in, rotate, shear, horizontal and vertical flip, hue, brightness, saturation, and contrast. 
- Since the dataset is heavily unbalanced i.e. malignant cases make up less than 2 percent of images, I oversampled malignant cases by a factor of 4. 
- Data was divided in 5 cross-validation folds. The within-fold (i.e. training) part of each fold was complemented with the 2018 data, while the out-of-fold (OOF) was not. 

## Inference and Post-Processing
Inference was made with un-augmented images as well as with up to 30 random test time augmentations (TTA) with each of the 15 classifiers. 

- Predictions across augmentations were weighted equally within each classifier. 
- Test predictions across EfficientNet versions for models trained within each fold were weighted according to their performance on their corresponding OOF (validation) data. This yielded 5 predictions, each corresponding to 1 of the 5 cross-validation folds.
- The 5 test predictions above were then weighted equally (i.e. simple-averaged) to yield a single prediction. 


## Credits:
- Thanks to **The Society for Imaging Informatics in Medicine**, **The International Skin Imaging Collaboration**, and **Kaggle** for hosting this competition and for making this data available. Thanks to the **Kaggle community** for great insights and very valuable guidance.

- Pavel Yakubovskiy's (qubvel) [EfficientNet library](https://github.com/qubvel/efficientnet) was used to set up all classifiers in this notebook. The pre-trained imagenet and noisy-student weights also come from his library. 

- Wei Hao Khoong's Notebook [[SIIM-ISIC] Multiple Model Training + Stacking](https://www.kaggle.com/khoongweihao/siim-isic-multiple-model-training-stacking) provided me great insight into TPU training, TensorFlow 2.x, Dataset API, as well as model stacking. 

- The TFRecords used for training come from Chris Deotte's Kaggle datasets [Melanoma TFRecords 1024x1024](https://www.kaggle.com/cdeotte/melanoma-1024x1024) for the 2020 data, and [ISIC 2019 TFRecords 768x768](https://www.kaggle.com/cdeotte/isic2019-768x768) for the 2018 data.

- Chris Deotte's notebook [Triple Stratified KFold with TFRecords](https://www.kaggle.com/cdeotte/triple-stratified-kfold-with-tfrecords/). Not only it was an inspiration, but it provided great guidance into the developing of this notebook. Also, the augmentation functions for image shear and rotate used here were adapted from his notebook. 


To run the notebook follow the instructions found in the Reproduce Notebook section just below the introduction in the Jupyter notebook.


<br>
