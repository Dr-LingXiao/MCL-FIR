# A Multihead Continual Learning Framework for Fine-Grained Fashion Image Retrieval with Contrastive Learning and Exponential Moving Average Distillation


![Framework Overview](assets/MCL-FIR.jpg)




### Download Data

#### Dataset Split

We supply our dataset split and some descriptions of the datasets with a bunch of meta files. Download them by the following script.

```sh
wget -c -P data/ https://drive.google.com/file/d/1_Cyo-IkHYU977bneTXaMC_f63e3vLfSA/view?usp=sharing
cd data/
tar -zxvf meta_data.tar.gz
```

#### FashionAI Dataset

As the full FashionAI has not been publicly released, we utilize its early version for the [FashionAI Global Challenge 2018](https://tianchi.aliyun.com/competition/entrance/231671/introduction?spm=5176.12281949.1003.9.493e3eafCXLQGm). You can first sign up and download the data. Once done, you should uncompress them into the `FashionAI` directory:

```sh
unzip fashionAI_attributes_train1.zip fashionAI_attributes_train2.zip -d {your_project_path}/data/FashionAI
```

#### DARN Dataset

As some images’ URLs have been broken, only 214,619 images are obtained for our experiments. We provide with a series of [URLs](https://drive.google.com/file/d/10jpHsFI2njzEGl7kdACXbvstz6tXyE0R/view?usp=sharing) for the images.

#### DeepFashion Dataset

[DeepFashion](https://www.cv-foundation.org/openaccess/content_cvpr_2016/papers/Liu_DeepFashion_Powering_Robust_CVPR_2016_paper.pdf) is a large dataset which consists of four benchmarks for various tasks in the field of clothing including [category and attribute prediction](http://mmlab.ie.cuhk.edu.hk/projects/DeepFashion.html) which we use for our experiments, in-shop clothes retrieval, fashion landmark detection and consumer-to-shop clothes retrieval.

#### Zappos50k Dataset

We utilize identical split provided by [Conditional SImilarity Network](https://arxiv.org/abs/1603.07810). To download the Zappos50k dataset and their triplet list, please refer to [their repository](https://github.com/andreasveit/conditional-similarity-networks).



## Getting Started

All data prepared, you can simply train the model with
```sh
python mainFashionAI_3datasets.py
```
## Testing

As training terminates, two snapshots are saved for testing. One is the model that has the highest performance on validation set and the other is the one of the latest epoch. You can load any of them and test on the test set.
```sh
python mainFashionAI_test_3datasets.py
```


If you find this work useful in your research, please consider citing:
```bibtex
@article{Xiao_TMM,
  title   = {A Multihead Continual Learning Framework for Fine-Grained Fashion Image Retrieval
             with Contrastive Learning and Exponential Moving Average Distillation},
  author  = {Xiao, Ling and Yamasaki, Toshihiko},
  journal = {IEEE Transactions on Multimedia},
  year    = {2025}
}
