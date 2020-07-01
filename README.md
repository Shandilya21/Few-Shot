<a href="https://github.com/Shandilya21/Few-Shot/issues"><img alt="GitHub issues" src="https://img.shields.io/github/issues/Shandilya21/Few-Shot"></a> 
<a href="https://github.com/Shandilya21/Few-Shot/network"><img alt="GitHub forks" src="https://img.shields.io/github/forks/Shandilya21/Few-Shot"></a>
<a href="https://github.com/Shandilya21/Few-Shot/stargazers"><img alt="GitHub stars" src="https://img.shields.io/github/stars/Shandilya21/Few-Shot"></a>
<a href="https://github.com/Shandilya21/Few-Shot/blob/master/LICENSE"><img alt="GitHub license" src="https://img.shields.io/github/license/Shandilya21/Few-Shot"></a>

## Few Shot, Zero Shot and Meta Learning Research

The objective of the repository is working on a few shot, zero-shot, and meta learning problems and also to write readable, clean, and tested code. Below is the implementation of a few-shot algorithms for image classification.

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

## Introduction

### What is Few Shot Learning?
With the advancement of machine learning mainly in computational resources, and has been highly successful in data-intensive application but often slows down when the data is small. Recently, few-shot learning (FSL) is proposed to tackle this problem. Using prior knowledge, FSL can generalize to new tasks containing few samples with supervision. Based on how prior knowledge can be used to handle this core issue, FSL methods categorize into three perspectives: (i) data, which uses prior knowledge to augment the supervised experience (ii) model, which uses prior knowledge to reduce the size of the hypothesis
space and (iii) algorithm, which uses prior knowledge to alter the search for the best hypothesis in the given hypothesis space.

### 1.1 Notation and Terminology
Consider a learning task T , FSL deals with a data set D = {Dtrain,Dtest} consisting of a training set Dtrain = {(xi,yi)} i = 1 to I where I is small, and a testing set Dtest = {xtest}. Let p(x,y) be the ground-truth joint probability distribution of input x and output y, and ˆh be the optimal hypothesis from x to y. FSL learns to discover ˆh by fitting Drain and testing on Dtest. To approximate ˆh, the FSL model determines a hypothesis space H of hypotheses h(θ) where θ denotes all the parameters used by h. Here, a parametric h is used, as a nonparametric model often requires large data sets, and thus not suitable for FSL. The below Figure, illustrates a different perspective of FSL method to solve the problems.

![](https://github.com/Shandilya21/few_shot_research/raw/master/images/FSL_methods.jpg)

## Theory
### Prototypical Networks

![](https://github.com/Shandilya21/few_shot_research/raw/master/images/proto_nets_diagram.png)


To achieve optimal few shot performance [(Snell et.al)](https://arxiv.org/pdf/1703.05175.pdf) apply compelling inductive bias in class prototype form. The assumption made to consider an embedding in which samples from each class cluster around the **prototypical representation** which is nothing but the mean of each sample. However, In the n-shot classification problem, where n > 1, it performed by taking a class to the closest prototype. With this, the paper, has a strong theoretical proof on using euclidean distance over cosine distance which also represents the class mean of prototypical representations. Prototypical Networks also work for **Zero-Shot Learning**, which can learn from rich attributes or natural language descriptions. For eg. "color", "master category", "season", and "product display name", etc.


### Meta Agnostic Meta Learning (MAML)

![](https://github.com/Shandilya21/few_shot_research/raw/master/images/maml_diagram.png)


The objective of meta-learning algorithms is to optimize meta parameters. Precisely, we have algorithms that access to the training loss and some meta parameters and output some optimal or learned parameters. Likewise, Meta Agnostic Meta-Learning short for MAML is an optimization algorithm compatible with the model that learns through gradient descent. The meta parameters was a point of initialization for the SGD algorithms shared between all the independent task. Since the SGD update is differentiable, one can compute the gradients concerning meta parameters simply through backpropagation.


## Setup
### Requirements

This codebase requires Python 3.5 (or higher). We recommend using Anaconda or Miniconda for setting up the virtual environment. Here's a walk through for the installation and setup.

Clone the Repository
```
git clone https://github.com/Shandilya21/few_shot_research.git
cd Few-Shot
conda create -n few_shot python=3.5
conda activate few_shot
```
Install all supporting libraries and packages in "requirements.txt".
```
pip install -r requirements.txt
```
Download the data, and place inside data folder. Extract the zip files to continue.

Edit DATA_PATH in ```config.py``` and replace with appropriate __dataset_path__. <br />

Kindly go through below instructions for fashionNet dataset preperation

```
python script/prepare_fashionNet.py
```

To know the dataset in details, kindly refer ```data/fashionNet/README.md```.

#### Training
```
bash chmod +x experiments/run.sh
./run.sh
```

### Checkpoints (.pth) and Preprocessed Data Set
To reproduce the results on fashionNet DataSet, download the preprocessed data and Checkpoints.
[***(Download)***](https://drive.google.com/drive/folders/1jTHGsISd44RkwBcWP-LTRhGbfYbBpBZG?usp=sharing) place the files inside ```DATA_PATH/fashionNet/```.

## Approach
#### ProtoTypical Networks

```Run `experiments/proto_nets.py` to reproduce results using Prototypical Networks```.

**Arguments**
- ```dataset```: {'fashionNet'}.
- ```distance```: {'l2', 'cosine'}. Which distance metric to use
- ```n-train```: Support samples per class for training tasks
- ```n-test```: Support samples per class for validation tasks
- ```k-train```: Number of classes in training tasks
- ```k-test```: Number of classes in validation tasks
- ```q-train```: Query samples per class for training tasks
- ```q-test```: Query samples per class for validation tasks

In the main paper of Prototypical network, the author present strong arguments of euclidean distance over cosine distance which also represents the class mean of prototypical representations which we reciprocate in the experiments.

| Small version |    1   |   2   |  3    |
|---------------|--------|-------|-------|
|k - ways       | 2      | 3     | 5     |
|n - shots      | 2      | 4     | 5     |
|This Repo (l2) | 80.2   | 77.5  | 84.74 |
|This Repo (Cos)| 72.5   | 73.88 | 77.68 |


#### Meta Agnostic Meta Learning (MAML)

```Run `experiments/maml.py` to reproduce results using MAML Networks. (Refer the Theory section for details)```.

**Arguments**

- ```dataset```: {'omniglot', 'miniImageNet'}. Whether to use the Omniglot or miniImagenet dataset
- ```distance```: {'l2', 'cosine'}. Which distance metric to use
- ```n```: Support samples per class for few-shot tasks
- ```k```: Number of classes in training tasks
- ```q```: Query samples per class for training tasks
- ```inner-train-steps```: Number of inner-loop updates to perform on training tasks
- ```inner-val-steps```: Number of inner-loop updates to perform on validation tasks
- ```inner-lr```: Learning rate to use for inner-loop updates
- ```meta-lr```: Learning rate to use when updating the meta-learner weights
- ```meta-batch-size```: Number of tasks per meta-batch
- ```order```: Whether to use 1st or 2nd order MAML
- ```epochs```: Number of training epochs
- ```epoch-len```: Meta-batches per epoch
- ```eval-batches```: Number of meta-batches to use when evaluating the model after each epoch


| Small version | Order  |   1   |   2   |   3   |
|---------------|--------|-------|-------|-------|
|k - ways       |        | 2     | 5     | 5     |
|n - shots      |        | 1     | 3     | 5     |
|This Repo      | 1      | 92.67 | 90.65 | 93.23 |


### TODO
* Multimodal Few Shot Classification.
* Zero Shot Image Classification.

<!-- CONTRIBUTING -->
## Contributing

Contributions are very welcome. If you know how to make this code better, please open an issue. If you want to submit a pull request, please open an issue first.

## Implementation References
* [(oscarknagg)](https://github.com/oscarknagg/few-shot) for implementation (code in PyTorch).
