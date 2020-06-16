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

![](https://github.com/Shandilya21/few_shot_research/raw/master/images/proto_nets_diagram.png)


To achieve optimal few shot performance [(Snell et.al)](https://arxiv.org/pdf/1703.05175.pdf) apply compelling inductive bias in class prototype form. The assumption made to consider an embedding in which samples from each class cluster around the **prototypical representation** which is nothing but the mean of each sample. However, In the n-shot classification problem, where n > 1, it performed by taking a class to the closest prototype. With this, the paper, has a strong theoretical proof on using euclidean distance over cosine distance which also represents the class mean of prototypical representations. Prototypical Networks also work for **Zero-Shot Learning**, which can learn from rich attributes or natural language descriptions. For eg. "color", "master category", "season", and "product display name", etc.


### Meta Agnostic Meta Learning (MAML)

![](https://github.com/Shandilya21/few_shot_research/raw/master/images/maml_diagram.png)


The objective of meta-learning algorithms is to optimize meta parameters. Precisely, we have algorithms that access to the training loss and some meta parameters and output some optimal or learned parameters. Likewise, Meta Agnostic Meta-Learning short for MAML is an optimization algorithm compatible with the model that learns through gradient descent. The meta parameters was a point of initialization for the SGD algorithms shared between all the independent task. Since the SGD update is differentiable, one can compute the gradients concerning meta parameters simply through backpropagation.

## 1. Data Set

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


#### 1.2 Data Preperations for fashionNet Dataset

###### (i). Data Preprocessing Approach:
 * Preprocess (.csv) file with basic utils such as [NaN, empty rows or columns, incomplete data], eithier by removing or replacing or augmenting.
 * naming convention has changed to class name of each product, for e.g: images/7100.jpg --> images/"Duffel Bag__7100.jpg".
 * Split the Meta training and testing on the basis articleType, for eg:, cufflinks ---> background classes, Shirts, Tie, etc ---> evaluation classes. 
 * Moved the images to the right location i.e, to the correct classes.
 * Please refer code ```script/prepare_fashionNet.py``` in details.

How we split the background (support) and (query) samples are based on the set of specific classes which is under the **data/fashionNet/Meta**  The Proto-nets paper also notes using a larger “way”, i.e. more classes during training may help for better performance.

###### (ii) DataLoader or n_shot_preprocessing
* raw_images ('RGB') of ~ [60 X 80] ------> CenterCrop(56), and Resize to (28, 28).
* class_name: image_name, for example: let image name is ```Duffel Bag__7100.jpg``` ---> [Casual Shirts (articleType) + (image_id)] is a class name.
* In ```core.py/prepare_n_shot```, you may find the n shot task label. return tensor [q_queries * K_shots, ]. Nshotsampler for training and evaluation is wrapper Sampler subclass that generates batches of n-shot, k-way, q-query tasks. this wrapper function return the batch tensors of support and query sets. The support and query set are disjoint i.e. do not contain overlapping samples.

Follow the below instructions to prepare the fashionNet dataset

```
python script/prepare_fashionNet.py
```
After acquiring the data and running the setup scripts your folder structure would look like

```
DATA_PATH/
    fashionNet/
        images_background/
        images_evaluation/
        refac_images/
```
images_background: contains support classes
images_evaluation: contains query classes
refac_images : Images after renamed (based on class name + image_id)

##### Checkpoints (.pth) and Preprocessed Data Set
If you want to reproduce the results on fashionNet DataSet, use the preprocessed data and Checkpoints.
[***(Download)***](https://drive.google.com/drive/folders/1jTHGsISd44RkwBcWP-LTRhGbfYbBpBZG?usp=sharing) the ```data.tar.gz``` files and place inside ```DATA_PATH/fashionNet/```. Extract the files and run the code.


## 2 Experiment Setup
### 2.1 Requirements

Use virtualenv (preferable).
Clone the Repository
```
git clone https://github.com/Shandilya21/few_shot_research.git
```
Listed in "requirements.txt" Install neccesssary supporting libraries for reproducing results.

```
pip install -r requirements.txt
```
Download Data from the Link, and put inside data folder refer ```data/fashionNet/README.md``` in details. Extract the zip files contains Images, and csv file for fashion product items descriptions or other details.

Edit DATA_PATH in ```config.py``` and replace with the fashionNet dataset location.

Run the command to run all the experiments. 
```
bash chmod +x experiments/run.sh
./run.sh
```

## 3. Networks
#### 3.1 ProtoTypical Networks

Run `experiments/proto_nets.py` to reproduce results using Prototypical Networks. (Refer the Theory section for details).

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

|               |Fashion |       |       |
|---------------|--------|-------|-------|
|k - ways       | 2      | 3     | 5     |
|n - shots      | 2      | 4     | 5     |
|This Repo (l2) | 80.2   | 77.5  | 84.74 |
|This Repo (Cos)| 72.5   | 73.88 | 77.68 |


#### 3.2 Meta Agnostic Meta Learning (MAML)

Run `experiments/maml.py` to reproduce results using MAML Networks. (Refer the Theory section for details).

**Arguments**

- dataset: {'omniglot', 'miniImageNet'}. Whether to use the Omniglot or miniImagenet dataset
- distance: {'l2', 'cosine'}. Which distance metric to use
- n: Support samples per class for few-shot tasks
- k: Number of classes in training tasks
- q: Query samples per class for training tasks
- inner-train-steps: Number of inner-loop updates to perform on training tasks
- inner-val-steps: Number of inner-loop updates to perform on validation tasks
- inner-lr: Learning rate to use for inner-loop updates
- meta-lr: Learning rate to use when updating the meta-learner weights
- meta-batch-size: Number of tasks per meta-batch
- order: Whether to use 1st or 2nd order MAML
- epochs: Number of training epochs
- epoch-len: Meta-batches per epoch
- eval-batches: Number of meta-batches to use when evaluating the model after each epoch


|           | Order  |Fashion|       |       |
|-----------|--------|-------|-------|-------|
|k - ways   |        | 2     | 5     | 5     |
|n - shots  |        | 1     | 3     | 5     |
|This Repo  | 1      | 92.67 | 90.65 | 93.23 |


#### 4. Future Work and Approaches
##### 4.1 Multimodal Few Shot Classification

We can extend the given problem in many possible ways, however, multimodal few shot classification one of the potential approaches that have an active research in dialogue, and question answering systems and shows significant results. A multi-modal approach facilitates bridging the information gap by means of meaningful joint embeddings. Similar previous research includes [(Chen Xing et.al)](https://papers.nips.cc/paper/8731-adaptive-cross-modal-few-shot-learning.pdf), [(Frederik Pahde et.al)](https://openreview.net/pdf?id=HJB8ntJPG) open a new frontier of research in the respective areas. In given problem we extend to use ```productDisplay``` (refer dataset-table), which have unique description for every fashion products as intutive inference for the images to build an effective classifier with less number of samples for each classes. We can build a end to end multimodal classifier, which has both the modalities (text and images) while training and only images during test time.

![](https://github.com/Shandilya21/few_shot_research/raw/master/images/multimodal.png)

In the above figure the product desciption has feeded with Bi-LSTM for extracting textual semantics, whereas the images features is extracted using pretrained ResNet trained on ImageNet instance. The cross attention is incorporated to attend the features of textual attributes with respect to images and vice versa. The further context is passed with fully connected layer and trained on the cross entropy loss.

##### 4.2 Zero Shot Learning
We can extend the module and develop the zero shot learning approach for the task of image classofication on fashion dataset. Zero-Shot Learning is the  type of learning that able to predict classes that has not been seen while training the model. It resembles our ability to generalize and identify new things without explicit supervision.

<!-- CONTRIBUTING -->
#### Contributing

Contributions are what make the project such an amazing place to be learn, inspire, and create. Any contributions you make are **greatly appreciated**.

1. Fork the Project
2. Create your Feature Branch (`git build -b build/newfeature`)
3. Commit your Changes (`git commit -m 'Add some newfeature'`)
4. Push to the Branch (`git push origin build/newfeature`)
5. Open a Pull Request

#### Acknowledgment: 
[(oscarknagg)](https://github.com/oscarknagg/few-shot) I would like to thanks for sharing the code and supporting references.