# Knowledge-Enriched-Distributional-Model-Inversion-Attacks

This repository is dedicated to the first assignment of the "Data Protection Techniques" course. In this assignment, I will delve into a comprehensive study and conduct numerous experiments based on the paper: [Knowledge-Enriched Distributional Model Inversion Attacks]((https://openaccess.thecvf.com/content/ICCV2021/papers/Chen_Knowledge-Enriched_Distributional_Model_Inversion_Attacks_ICCV_2021_paper.pdf)\] ), which can also be accessed on [arXiv](https://arxiv.org/abs/2010.04092).

## Contents

- [Download Data](#download-data)
- [Training Classifier](#training-classifier)
- [Training Inversion-Specific GAN](#training-inversion-specific-gan)
- [Attacking The Model](#attacking-the-model)
- [Calculating Frechet Inception Distance (FID)](#calculating-fréchet-inception-distance)
- [Reference](#reference)


## Requirements

- Python 3.6
- PyTorch
- CUDA
- TensorBoardX
-  OpenCV
- PIL

## Download Data

This project uses the CelebA dataset, which originally labels images with attributes like 'smiling' or 'glasses.' We are not concerned with those labels, instead, we’re interested in the identity of datapoints. So, we work with 1,000 unique identities. For these identities, we have 27,018 training images and 3,009 test images. You can find the images and their corresponding labels in ./data/trainset.txt for training data and ./data/testset.txt for test data.

Furthermore, there is a distinct dataset available in ./data/ganset.txt which is public and is utilized to pretrain the GAN. It's crucial to note that this dataset does not have class overlap with the private data designated for training the classifier, ensuring that the identities in the GAN pretraining set are exclusive from those in the classifier’s training and test sets."

To download the CelebA dataset, you can run the provided Python script as follows:

```sh
python data_downloader.py
```

> ### NOTE
> There are occasions when downloading the CelebA dataset using torchvision
> results in a Google Drive error. This is due to the dataset being hosted on
> Google Drive, which sometimes restricts the number of download requests.
>
> In such cases, an alternative is to download the CelebA dataset directly 
> from Kaggle using the following link:
> [CelebA Dataset on Kaggle](https://www.kaggle.com/datasets/jessicali9530/celeba-dataset)


## Training Classifier
The paper is based on attacking a classifier model and extracting its private training data. To train such a classifier, we take some famous well-known model architectures that have been trained on the ImageNet dataset and fine-tune them to classify our celebrity identities. Follow the steps below to train a classifier.

1. To train a `FaceNet` or `IR152` based classifier, make sure to download model backbones from [here](https://drive.google.com/drive/folders/1ZTTrRJr-2HOgfyxndP8a9R2Hb_UOgV6J).
2. Run this in your command line: 

```sh
python train_classifier.py
```

**NOTES**:

- Training parameters can be accessed and modified from `./config/classify.json`.
- Pretrained checkpoints of these classifiers can also be found [here](https://drive.google.com/drive/folders/1ZTTrRJr-2HOgfyxndP8a9R2Hb_UOgV6J).
- `--model_name` parameter indicates the backbone architecrure of classifier.

The table below shows the details of pretrained backbone models.
| Model       | Size (MB) | Parameters | Depth | Time (ms) per inference step (CPU) | Time (ms) per inference step (GPU) |
|-------------|-----------|------------|-------|------------------------------------|------------------------------------|
| Resnet152   | 232       | 60.4M      | 311   | 127.4                              | 6.5                                |
| VGG16       | 528       | 138.4M     | 16    | 69.5                               | 4.2                                |
| FaceNet64   | 98        | 25.6M      | 22    | 58.2                               | 4.6             


## Training Inversion-Specific GAN
The assigned paper is focused on proposing a novel GAN architecture named Inversion-Specific GAN. To train this GAN, follow the steps below:

1. For each of the models, make sure the classifier is present at `./target_model/target_ckp`. This classifier can be downloaded from [here](https://drive.google.com/drive/folders/1ZTTrRJr-2HOgfyxndP8a9R2Hb_UOgV6J) or trained at the previous stage.
2. For training a GAN against VGG16, make sure the target model is named `VGG16_86.30_allclass.tar`.
3. For training a GAN against FaceNet64, make sure the target model is named `FaceNet64_88.50.tar`.
4. For training a GAN against IR152, make sure the target model is named `IR152_91.16.tar`.
5. Run this in your commandline.
```sh
python k+1_gan.py --model_name_T "VGG16"
```

**NOTES**:

- `--model_name_T` specifies the target model being attacked.
- training parameters can be accessed from ./config/celeba.json
- Trained GANs and generated images are in `./improvedGAN`.
- Pretrained checkpoints of these GANs can be accessed [here](https://drive.google.com/drive/folders/1eCuJXdpKlrIAf9jIYxQ1cHviCQ4hxL--?usp=sharing).
- A general binary GAN can be trained by running this in commandline.

```sh
python binary_gan.py
```

## Attacking The Model

This section is where we perform the actual attack. Here, we put the target model and our trained GAN face-to-face and execute the attack, calculating accuracy and top-5 accuracy. Simply forwarding images from the target model would not work due to the high bias of this approach. Since the GAN is trained on the target model, it might generate poor, random patterns of pixels but still optimize the target model. To calculate accuracy, we need another model named the Evaluation Classifier to work as an oracle. In this case, we use `FaceNet_95.88.tar`, which can be downloaded [here](https://drive.google.com/drive/folders/1ZTTrRJr-2HOgfyxndP8a9R2Hb_UOgV6J) as our Evaluation Classifier. Follow these steps:

1. Download the Evaluation Classifier `FaceNet_95.88.tar` from [here](https://drive.google.com/drive/folders/1ZTTrRJr-2HOgfyxndP8a9R2Hb_UOgV6J) and place it at `./target_model/target_ckp`.
2. Make sure you have the Classifiers as explained in the previous stage with the correct names and correct locations.
3. Place the Inversion-Specific GAN or GMI in the `./improvedGAN` location. You can train these GANs or download them from [here](https://drive.google.com/drive/folders/1eCuJXdpKlrIAf9jIYxQ1cHviCQ4hxL--).
4. The names of these GAN generators should be as follows: `improved_celeba_G`, `improved_celeba_G_facenet`, and `improved_celeba_G_IR152` for 'VGG16', 'FaceNet', and `IR152`, respectively. For GMI, the name should be `celeba_G.tar`.
5. Ensure you also have discriminators; their names should be exactly as the generator name but with 'G' replaced by 'D'. They can also be downloaded from [here](https://drive.google.com/drive/folders/1eCuJXdpKlrIAf9jIYxQ1cHviCQ4hxL--).
6. Run the following command in the command line with your specified arguments:
```sh
python recovery.py
```

**NOTES**:

- The `improved_flag parameter` indicates whether the 'Inversion-Specific GAN,' as mentioned in the paper, is being used for the attack.
- The `dist_flag parameter` signifies whether distribution recovery is employed or not.

By setting both improved_flag and dist_flag be False, we are simply using the method proposed in [[1]](#1).

## Calculating Fréchet Inception Distance
**FID (Fréchet Inception Distance)** is a classic metric employed to quantify the performance of Generative Adversarial Networks (GANs). It has emerged as a widely adopted measure for assessing the quality and realism of images generated by GANs. The fundamental concept behind FID involves comparing the statistical distributions of generated images and real images within a feature space derived from a pre-trained deep neural network, typically an Inception Network.

In this project, since the original code provided by the authors lacked a module to calculate FID, I implemented it separately. By executing the provided code, we will be able to compute the FID for both the GMI model and the Inversion-Specific model, enabling a comprehensive analysis of their performances. To calculate FID, follow the steps below:

1. Ensure that the generator for the model you wish to evaluate is located in `./improvedGAN` and is named `improved_celeba_G.tar`.

2. For calculating the FID of the GMI model, ensure that the GMI model is in `./improvedGAN` and named `celeba_G.tar`.

3. Run the following command in your command line:
```sh
python calculate_FID.py
```
**NOTES**:

To calculate FID on the improved generator, use the `--improved_flag`. Omitting it will result in calculating FID for the GMI baseline model.

---
>**Important Note:** If you wish to execute these lines and reproduce my experiments, simply run the code provided for this assignment in the `Knowledge Enriched DMI.ipynb` file. Be sure to consult my documentation if you encounter any issues.
>
>Happy experimenting :)

## Reference
<a id="1">[1]</a> 
Zhang, Yuheng, et al. "The secret revealer: Generative model-inversion attacks against deep neural networks." Proceedings of the IEEE/CVF Conference on Computer Vision and Pattern Recognition. 2020.
