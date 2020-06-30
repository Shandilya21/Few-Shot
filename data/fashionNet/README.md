## 1. Data Set

Download the datasets (small-version) [***(Download)***](https://www.kaggle.com/paramaggarwal/fashion-product-images-small), (full-version) [***(Download)***](https://www.kaggle.com/paramaggarwal/fashion-product-images-dataset/version/1). 

**Dataset Description**

- ```id```: Images id as in images_folder.
- ```gender```: Gender wise fashion items (M/W), etc. 
- ```masterCategory```: Categories contains type of fashion items such as Apparel, Accessories, etc.
- ```SubCategory```: Categories contains the specific fashion item category collections, such as Footwear, Watch etc.  
- ```articleType```: Categories contains the items specifc such as Topwear -> Tshirts, Shirts, Shoes --> Casual, etc.  
- ```baseColour```: Color of the articleType items such as NavyBlue, Black, Grey, etc. 
- ```season```: fashion items specific to seasons (fall/winter/summer).
- ```usage```: Fashion items for specific purposes, such as casual, ethnic, etc.
- ```displayName```: Name displayed on items with specific attributes. 


| id | gender| masterCategory| SubCategory| articleType| baseColour| season| usage | productDisplayName        | 
|----|-------|---------------|------------|------------|-----------|-------|-------|---------------------------|
|1163| Male  | Apparel       | TopWear    | Shirt      | NavyBlue  | Fall  | Ethnic| Turtle Men Navy Blue Shirt|
|1165| Female| Apparal       | BottomWear | Jeans      | Black     | Summer| Casual| Levis Female Black Jeans  |
|2152| Female| Accessories   | Watches    | Watches    | Silver	   | Winter| Formal| Titan Women Silver Watch  |
|1455| Girl  | Apparel       | TopWeat    | Tshirt     | Grey	   | Summer| Casual| Gini Jony Girls Knit Top  |



#### 1.1 Data Preperations for fashionNet Dataset

###### (i). Data Preprocessing Approach:
 * Preprocess (.csv) file by removing [NaN, empty rows or columns, incomplete data], eithier by removing or augmentation.
 * Rename Images to their class name with image id, for e.g: images/7100.jpg --> images/"Duffel Bag__7100.jpg".
 * Split the Meta training (support) and testing (query), for eg:, cufflinks ---> background classes, Shirts, Tie, etc ---> evaluation classes. 
 * Moved the images to their respective class name folder.
 * FOr more details, refer code ```script/prepare_fashionNet.py```.

###### (ii) DataLoader or n_shot_preprocessing
* Images ('RGB') of ~ [60 X 80] ------> CenterCrop(56), and Resize to (28, 28).
* In ```core.py/prepare_n_shot```, you may find the n shot task label. return tensor [q_queries * K_shots, ]. Nshotsampler is wrapper class that generates batches of n-shot, k-way, q-query tasks. Also, The support and query set are disjoint i.e. do not contain overlapping samples.

```
DATA_PATH/
    fashionNet/
        images_background/
        images_evaluation/
        refac_images/
```
After acquiring the data and running the setup scripts your folder structure would look like

```images_background```: contains support classes <br />
```images_evaluation```: contains query classes <br />
```refac_images```: Images after renamed (based on class name + image_id) <br />