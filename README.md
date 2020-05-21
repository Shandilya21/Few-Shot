## Few Short, Zero Shot Learning Research

The objective of this repository is to working in few shot, zero shot learning research and contains tested and clean code. This is an implementation of
few-shot image classification in fashion domain, using state of the art Prototypical Networks, and other FSL network, etc.

### Important Blogs and Paper
1. Generalizing from a Few Examples: A Survey on Few-Shot Learning [QUANMING Y et al. (2020)](https://arxiv.org/pdf/1904.05046.pdf)
2. Prototypical Networks for Few-shot Learning [J. Snellet al. (2017)](https://arxiv.org/pdf/1703.05175.pdf)
3. Matching Networks for One Shot Learning [Vinyals et al. (2017)](https://arxiv.org/pdf/1606.04080.pdf)
4. Model-Agnostic Meta-Learning for Fast Adaptation of Deep Networks [Finn et al. (2017)](https://arxiv.org/pdf/1703.03400v3.pdf)
5. Learning to Compare: Relation Network for Few-Shot Learning [Sung F et al. (2018)](https://arxiv.org/pdf/1711.06025v2.pdf)
6. Optimization as a Model For Few-Shot Learning [Ravi. S et al. (2017)](https://openreview.net/pdf?id=rJY0-Kcll)
7. How To Train Your MAML [Antreas A et al. (2017)](https://arxiv.org/pdf/1810.09502.pdf)
8. [Theory and Concepts](https://towardsdatascience.com/advances-in-few-shot-learning-a-guided-tour-36bc10a68b77)
9. [Implementation in PyTorch](https://towardsdatascience.com/advances-in-few-shot-learning-reproducing-results-in-pytorch-aba70dee541d)
10. [Few Shot Learning in CVPR 2019]((https://towardsdatascience.com/few-shot-learning-in-cvpr19-6c6892fc8c5))

### What is Few Shot Learning?
With advancement of machine learning due to computational resources, and has been highly successful in data intensive application but often slow down when the data is small. Recently, few shot learning (FSL) is proposed to tackle this problem. Using prior knowledge, FSL can generalize to new tasks containing few samples with supervision. Based on how prior knowledge can be used to handle this core issue, FSL methods categorises into three perspectives: (i) data, which uses prior knowledge to augment the supervised experience; (ii) model, which uses prior knowledge to reduce the size of the hypothesis
space; and (iii) algorithm, which uses prior knowledge to alter the search for the best hypothesis in the given hypothesis space. 

### Notation and Terminology
Consider a learning task T , FSL deals with a data set D = {Dtrain,Dtest} consisting of a training set Dtrain = {(xi,yi)} i=1 to I where I is small, and a testing set Dtest = {xtest}. Let p(x,y) be the ground-truth joint probability distribution of input x and output y, and ˆh be the optimal hypothesis from x to y. FSL learns to discover ˆh by fitting Dtrain and testing on Dtest. To approximate ˆh, the FSL model determines a hypothesis space H of hypotheses h(θ)’ where θ denotes all the parameters used by h. Here, a parametric h is used, as a nonparametric model often requires large data sets, and thus not suitable for FSL. Below Figure, illustrates about different perspective of FSL methods to solve problem.

![](https://github.com/Shandilya21/few_shot_research/raw/master/images/FSL_methods.jpg)

## Data Set
Download the datasets from here (small)[Download](https://www.kaggle.com/paramaggarwal/fashion-product-images-small), (full)[Download](https://www.kaggle.com/paramaggarwal/fashion-product-images-dataset/version/1) Download any version of the dataset as per your requirement. I suggest to test how it works on small version and then build on complete version of the dataset.

**DataSet Description**
- id: Image id map with fashion images.
- gender: Gender wise fashion items (M/W), etc. 
- masterCategory: categories contains type of fashion items such as Apparel, Accessories, etc.
- SubCategory: categories contains the specific fashion item category collections, such as Footwear, Watches, topwear, etc.  
- articleType: categories contains the items specifc such as Topwear -> Tshirts, Shirts, Shoes --> Casual, Bags --> Handbags, etc.  
- baseColour: Color of the articleType items such as NavyBlue, Black, Grey, etc. 
- season: cloth for specific suitable based on seasons (fall/winter/summer)
- usage: item dedicated to specific purpose, such as wedding, ethinic, casual, formal, etc.
- displayName: Name displayed on items with specific features attributes 


| id  | gender | masterCategory | SubCategory | articleType | baseColour | season | usage  | productDisplayName                 | 
|-----|--------|----------------|-------------|-------------|------------|--------|--------|------------------------------------|
|1163 | Male   | Apparel        | TopWear     | Shirt       | NavyBlue	 | Fall   | Ethnic | Turtle Check Men Navy Blue Shirt   |
|1165 | Female | Apparal        | BottomWear  | Jeans       | Black      | Summer | Casual | Peter England Female Party Jeans   |
|2152 | Female | Accessories    | Watches     | Watches     | Silver	 | Winter | Formal | Titan Women Silver Watch           |
|1455 | Girl   | Apparel        | TopWeat     | Tshirt      | Grey		 | Summer | Casual | Gini and Jony Girls Knit White Top |