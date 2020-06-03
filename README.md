## Few Short, Zero Shot Learning Research

The objective of the repository is working on a few shot, and zero-shot learning problems and also to write readable, clean, and tested code. This includes the implementation of a few-shot image classification problems, using algorithms such as Prototypical Networks, etc.

## Important Blogs and Paper

1. Generalizing from a Few Examples: A Survey on Few-Shot Learning [(QUANMING Y et al. (2020))](https://arxiv.org/pdf/1904.05046.pdf)
2. Prototypical Networks for Few-shot Learning [(J. Snellet al. (2017))](https://arxiv.org/pdf/1703.05175.pdf)
3. Matching Networks for One Shot Learning [(Vinyals et al. (2017))](https://arxiv.org/pdf/1606.04080.pdf)
4. Model-Agnostic Meta-Learning for Fast Adaptation of Deep Networks [(Finn et al. (2017))](https://arxiv.org/pdf/1703.03400v3.pdf)
5. Learning to Compare: Relation Network for Few-Shot Learning [(Sung F et al. (2018))](https://arxiv.org/pdf/1711.06025v2.pdf)
6. Optimization as a Model For Few-Shot Learning [(Ravi. S et al. (2017))](https://openreview.net/pdf?id=rJY0-Kcll)
7. How To Train Your MAML [(Antreas A et al. (2017))](https://arxiv.org/pdf/1810.09502.pdf)
8. [Theory and Concepts](https://towardsdatascience.com/advances-in-few-shot-learning-a-guided-tour-36bc10a68b77)
9. [Implementation in PyTorch](https://towardsdatascience.com/advances-in-few-shot-learning-reproducing-results-in-pytorch-aba70dee541d)
10. [Few Shot Learning in CVPR 2019](https://towardsdatascience.com/few-shot-learning-in-cvpr19-6c6892fc8c5)

## 1. Introduction

### What is Few Shot Learning?
With the advancement of machine learning mainly in computational resources, and has been highly successful in data-intensive application but often slows down when the data is small. Recently, few-shot learning (FSL) is proposed to tackle this problem. Using prior knowledge, FSL can generalize to new tasks containing few samples with supervision. Based on how prior knowledge can be used to handle this core issue, FSL methods categorize into three perspectives: (i) data, which uses prior knowledge to augment the supervised experience (ii) model, which uses prior knowledge to reduce the size of the hypothesis
space and (iii) algorithm, which uses prior knowledge to alter the search for the best hypothesis in the given hypothesis space.

### 1.1 Notation and Terminology
Consider a learning task T , FSL deals with a data set D = {Dtrain,Dtest} consisting of a training set Dtrain = {(xi,yi)} i = 1 to I where I is small, and a testing set Dtest = {xtest}. Let p(x,y) be the ground-truth joint probability distribution of input x and output y, and ˆh be the optimal hypothesis from x to y. FSL learns to discover ˆh by fitting Drain and testing on Dtest. To approximate ˆh, the FSL model determines a hypothesis space H of hypotheses h(θ) where θ denotes all the parameters used by h. Here, a parametric h is used, as a nonparametric model often requires large data sets, and thus not suitable for FSL. The below Figure, illustrates a different perspective of FSL method to solve the problems.

![](https://github.com/Shandilya21/few_shot_research/raw/master/images/FSL_methods.jpg)

## 2. Theory
### Prototypical Networks
To achieve optimal few shot performance [(Snell et.al)](https://arxiv.org/pdf/1703.05175.pdf) apply compelling inductive bias in class prototype form. The assumption made to consider an embedding in which samples from each class cluster around the **prototypical representation** which is nothing but the mean of each sample. However, In the n-shot classification problem, where n > 1, it performed by taking a class to the closest prototype. With this, the paper, has a strong theoretical proof on using euclidean distance over cosine distance which also represents the class mean of prototypical representations. Prototypical Networks also work for **Zero-Shot Learning**, which can learn from rich attributes or natural language descriptions. For eg. "color", "master category", "season", and "product display name", etc.

![](https://github.com/Shandilya21/few_shot_research/raw/master/images/proto_nets_diagram.png)

## 3. Data Set

Download the datasets from here (small-version) [(Download)](https://www.kaggle.com/paramaggarwal/fashion-product-images-small), (full-version) [(Download)](https://www.kaggle.com/paramaggarwal/fashion-product-images-dataset/version/1) Download any version of the dataset as per your requirement. Preferable to perform test how models works on smaller version and then build on full version of the dataset.

**DataSet Description**

- id: Images id as in images_folder/.
- gender: Gender wise fashion items (M/W), etc. 
- masterCategory: Categories contains type of fashion items such as Apparel, Accessories, etc.
- SubCategory: Categories contains the specific fashion item category collections, such as Footwear, Watch etc.  
- articleType: Categories contains the items specifc such as Topwear -> Tshirts, Shirts, Shoes --> Casual, etc.  
- baseColour: Color of the articleType items such as NavyBlue, Black, Grey, etc. 
- season: fashion items specific to seasons (fall/winter/summer).
- usage: Fashion items for specific purposes, such as casual, ethnic, etc.
- displayName: Name displayed on items with specific attributes. 


| id | gender| masterCategory| SubCategory| articleType| baseColour| season| usage | productDisplayName        | 
|----|-------|---------------|------------|------------|-----------|-------|-------|---------------------------|
|1163| Male  | Apparel       | TopWear    | Shirt      | NavyBlue  | Fall  | Ethnic| Turtle Men Navy Blue Shirt|
|1165| Female| Apparal       | BottomWear | Jeans      | Black     | Summer| Casual| Levis Female Black Jeans  |
|2152| Female| Accessories   | Watches    | Watches    | Silver	   | Winter| Formal| Titan Women Silver Watch  |
|1455| Girl  | Apparel       | TopWeat    | Tshirt     | Grey	   | Summer| Casual| Gini Jony Girls Knit Top  |


## 3.1 Experiment Setup
### 3.1.1. Requirements

Use virtualenv (preferable). you can find link for virtualenv setup as [(virtualenv1)](https://stackoverflow.com/questions/52816156/how-to-create-virtual-environment-for-python-3-7-0) and [(virtualenv2)](https://stackoverflow.com/questions/43069780/how-to-create-virtual-env-with-python3/43070301).

Clone the Repository
```
git clone https://github.com/Shandilya21/few_shot_research.git
```
Listed in "requirements.txt" Install neccesssary supporting packages or libraries for reproducing similar results.

```
pip install -r requirements.txt
```
Download Data from the Link, and put inside data folder. Extract the zip files contains Images, and csv file for fashion product items descriptions or other details. refer to **Data Descriptions" section for an overview.

Edit DATA_PATH in config.py and replace with the location where you stored the fashionNet dataset.

#### 3.1.2 Data Preperations for fashionNet Dataset

###### (i). Data Preprocessing Approach:
 * Preprocess (.csv) file with basic utils such as [NaN, empty rows or columns, incomplete data], eithier by removing or replacing or augmenting.
 * naming convention has changed to class name of each product, for e.g: images/1163.jpg --> images/"Topwear__Casual Shirts.jpg" for distributed the meta training and testing set.
 * Split the Meta training and testing on the basis subcateogry classes, for eg:, cufflinks ---> background classes, Shirts, Tie, etc ---> evaluation classes. 
 * Move the images to the right location
 * Please refer code ```script/prepare_fashionNet.py``` for further details.

```Meta Training Set or testing set class distribution may have changed if we add more classes into it. The Proto-nets paper also notes using a larger “way”, i.e. more classes during training may help. you can test with more meta data selecting from articleType (.csv) to ```data/fashionNet/Meta```.

Follow the following instructions to prepare the fashionNet dataset

```
python script/prepare_fashionNet.py
```
After acquiring the data and running the setup scripts your folder structure would look like

```
DATA_PATH/
    fashionNet/
        images_background/
        images_evaluation/
```

If you want to reproduce the results on fashionNet DataSet, use the preprocessed data. [(Download)](https://drive.google.com/open?id=1QcggrlCX5H7Q_FfXjko8rrlbVh1CtL8P) the files and place inside ```DATA_PATH/fashionNet/```.


###### (ii) DataLoader or n_shot_preprocessing
* raw_images ('RGB') of ~ [60 X 80] ------> CenterCrop(56), and Resize to (28, 28).
* class_name: image_name, for example: let image name is ```Topwear__Casual Shirts.jpg``` ---> [Topwear (subCategory) + Casual Shirts (articleType)] is a class name.
* In ```core.py/prepare_n_shot```, you may find the n shot task label. return tensor [q*K, ]. Nshotsampler for training and evaluation is wrapper Sampler subclass that generates batches of n-shot, k-way, q-query tasks. the wrapper function return the batch tensors of support and query sets. The support and query set are disjoint i.e. do not contain overlapping samples.


## 4. Prototypical Networks

Run `experiments/proto_nets.py` to reproduce results using Prototypical Networks. Refer to Descriptions section in detailed.

**Arguments**
- dataset: {'fashionNet'}.
- distance: {'l2', 'cosine'}. Which distance metric to use
- n-train: Support samples per class for training tasks
- n-test: Support samples per class for validation tasks
- k-train: Number of classes in training tasks
- k-test: Number of classes in validation tasks
- q-train: Query samples per class for training tasks
- q-test: Query samples per class for validation tasks

In the main paper of Prototypical network, the author present strong arguments of euclidean distance over cosine distance which also represents the class mean of prototypical representations which we reciprocate in the experiments.

Run the command to run the experiments for protonets. 
```
bash chmod +x experiments/running.sh
```
**Note: Please change the hparams accordingly.** 



