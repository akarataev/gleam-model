<h1 align="center">
  <img src="Docs/Icon.png" width="150" alt="icon">
  <br>Gleam model<br>
  <p align="center">
    <img src="https://img.shields.io/badge/Language-Python-blue.svg">
    <a href="LICENSE.md"><img src="https://img.shields.io/badge/License-MIT-brightgreen.svg"></a>
    <img src="https://img.shields.io/badge/Event-VK Hackathon 2018-orange.svg">
  </p>
</h1>
<p align="center">Skin cancer screening nerual network model</p>
<br><br>
<p align="center"><img src="Docs/Mockup.png" width="1000"></p>

## About Gleam

Skin cancer is in third place in terms of the incidence of oncology detection in Russian men and second in women. The five-year threshold is experienced by only half of the patients. Even with the second stage of melanoma and with proper treatment. Therefore, an early production diagnosis is practically the only way to get a favorable prognosis for treatment. The solution could be a mobile application, with which you can make an early diagnosis and promptly seek treatment.


## Dataset

To train this model the data to use is a set of images from the International Skin Imaging Collaboration: [Mellanoma Project ISIC](https://isic-archive.com).

### Preprocessing

The following preprocessing tasks are developed for each image:

1. Visual inspection to detect images with low quality
1. Image resizing: Transform images to 255x256x3
1. Crop images

## Final result

#### AUC/ROC curve chart VGG16

###### For transfer learning model:

<img src="Models/Plots/auc_roc_ft_vgg16.png">


#### Training and validation loss VGG16

###### For transfer learning (30 epochs):
<img src="Models/Plots/vl_tl_history.png">

###### For fine tuning (10 epochs):
<img src="Models/Plots/vl_ft_history.png">


#### Training and validation accuracy VGG16

###### For transfer learning (30 epochs):
<img src="Models/Plots/va_tl_history.png">

###### For fine tuning (10 epochs):
<img src="Models/Plots/va_ft_history.png">
