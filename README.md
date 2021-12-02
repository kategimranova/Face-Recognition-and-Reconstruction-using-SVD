# Face Recognition and Reconstruction using SVD
In this project
The method used for Face Recognition was described in the paper [Facial Recognition with Singular Value Decomposition](http://link.springer.com/chapter/10.1007/978-1-4020-6264-3_26)
# Face recognition
## Architecture
The picture shows the architecture (diagram) of the implemeted model
![architecture](https://github.com/kategimranova/Face-Recognition-and-Reconstruction-using-SVD/blob/main/images/architecture.png?raw=true)
## Data
To implement SVD-based face recognition we have chosen a widely used face recognition dataset called Labeled Faces in the Wild. This database consists of more than 13000 centered labeled face images of more than 5 thousands famous people, so we were able to pick several distinct photos for each person to train and validate SVD for face recognition on them. Fortunately, the dataset is obtainable from datasets module of sklearn where you can specify the minimum number of face images per person. We chose this parameter as 20 and loaded 3023 different pictures of 62 people. Every image from dataset consists of 62 x 47 pixels.
![dataset example](https://github.com/kategimranova/Face-Recognition-and-Reconstruction-using-SVD/blob/main/images/dataset.png?raw=true)

## Algorithm
### Step 1:
#### Obtain a training set S with N face images of known individuals. 
We divided our data pictures into train and test subsets, where train set covers 90% of all data (stratified sampling). In our case N = 2720. Every image from dataset consists of 62 x 47 pixels , so it can be represented as matrices with 62 rows and 47 columns. Then every matrix to 1-dimensional matrix or 62x47 column vector . These vectors convert to matrix called trainig set.
![step1](https://github.com/kategimranova/Face-Recognition-and-Reconstruction-using-SVD/blob/main/images/step1.png?raw=true)

### Step 2: 
**Compute the mean face f of set S by the following formula:**

$\overline{f} = \frac{1}{N}\sum_{n=1}^{N}f_i$

![step2](https://github.com/kategimranova/Face-Recognition-and-Reconstruction-using-SVD/blob/main/images/step2.png?raw=true)
