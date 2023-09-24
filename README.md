# Knowledge-Enriched-Distributional-Model-Inversion-Attacks

This repository is dedicated to the first assignment of the "Data Protection Techniques" course. In this assignment, I will delve into a comprehensive study and conduct numerous experiments based on the paper: [Knowledge-Enriched Distributional Model Inversion Attacks]((https://openaccess.thecvf.com/content/ICCV2021/papers/Chen_Knowledge-Enriched_Distributional_Model_Inversion_Attacks_ICCV_2021_paper.pdf)\] ), which can also be accessed on [arXiv](https://arxiv.org/abs/2010.04092).

## Sections


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
- Trained GANs and generated images are in `./improvedGAN`.
- Pretrained checkpoints of these GANs can be accessed [here](https://drive.google.com/drive/folders/1eCuJXdpKlrIAf9jIYxQ1cHviCQ4hxL--?usp=sharing).


## Getting Started
* Install required packages.
* Download relevant datasets including Celeba, MNIST, CIFAR10.
* Get target model prepared or run our code
    python train_classifier.py <br>
    Note that this code only provides three model architectures: VGG16, IR152, Facenet. And pretrained checkpoints for the three models can be downloaded at https://drive.google.com/drive/folders/1U4gekn72UX_n1pHdm9GQUQwwYVDvpTfN?usp=sharing.

## Build a inversion-specific GAN
* Modify the configuration in 'celeba.json'.
* Modify the target model path in 'k+1_gan.py' to your customized path.
* Run
    python k+1_gan.py.
* Model checkpoints and generated image results are saved in folder ’improvedGAN‘.
* A general GAN can be obtained as a baseline by running
    python binary_gan.py.
* Pretrained binary GAN and inversion-specific GAN can be downloaded at https://drive.google.com/drive/folders/1L3frX-CE4j36pe5vVWuy3SgKGS9kkA70?usp=sharing.


## Distributional Recovery
Run
    python recovery.py
    
* --model chooses the target model to attack.
* --improved_flag indicates if an inversion-specfic GAN is used. If False, then a general GAN will be applied.
* --dist_flag indicates if distributional recovery is performed. If False, then optimization is simply applied on a single sample instead of a distribution.
* By setting both improved_flag and dist_flag be False, we are simply using the method proposed in [[1]](#1).


## Reference
<a id="1">[1]</a> 
Zhang, Yuheng, et al. "The secret revealer: Generative model-inversion attacks against deep neural networks." Proceedings of the IEEE/CVF Conference on Computer Vision and Pattern Recognition. 2020.
