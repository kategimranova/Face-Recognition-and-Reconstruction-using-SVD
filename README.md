# Face Recognition and Reconstruction using SVD
The main goal of this project was to implement face recognition algorithm and face reconstruction from face projection to illustrate capabilities of SVD-based algorithms.
The method used for Face Recognition was described in the paper [Facial Recognition with Singular Value Decomposition](http://link.springer.com/chapter/10.1007/978-1-4020-6264-3_26).
# Face recognition
## Architecture
The picture shows the architecture (diagram) of the implemeted model
![architecture](https://github.com/kategimranova/Face-Recognition-and-Reconstruction-using-SVD/blob/main/images/architecture.png?raw=true)
## Code
File `Face_Recognition_and_Reconstruction.ipynb` contains the step-by-step implementation of algorithm.
## Data
To implement SVD-based face recognition we have chosen a widely used face recognition dataset called Labeled Faces in the Wild. This database consists of more than 13000 centered labeled face images of more than 5 thousands famous people, so we were able to pick several distinct photos for each person to train and validate SVD for face recognition on them. Fortunately, the dataset is obtainable from datasets module of sklearn where you can specify the minimum number of face images per person. We chose this parameter as 20 and loaded 3023 different pictures of 62 people. Every image from dataset consists of 62 x 47 pixels.
![dataset example](https://github.com/kategimranova/Face-Recognition-and-Reconstruction-using-SVD/blob/main/images/dataset.png?raw=true)

## Algorithm
### Step 1:
##### *Obtain a training set S with N face images of known individuals*. 
We divided our data pictures into train and test subsets, where train set covers 90% of all data (stratified sampling). In our case N = 2720. Every image from dataset consists of 62 x 47 pixels , so it can be represented as matrices with 62 rows and 47 columns. Then every matrix to 1-dimensional matrix or 62x47 column vector . These vectors convert to matrix called trainig set.

<img src="https://github.com/kategimranova/Face-Recognition-and-Reconstruction-using-SVD/blob/main/images/step1.png?raw=true" width="400" />


### Step 2: 
##### *Compute the mean face f of set S by the following formula*:

<img src="https://latex.codecogs.com/svg.image?\overline{f}&space;=&space;\frac{1}{N}\sum_{n=1}^{N}f_i" title="\overline{f} = \frac{1}{N}\sum_{n=1}^{N}f_i" />

<img src="https://github.com/kategimranova/Face-Recognition-and-Reconstruction-using-SVD/blob/main/images/step2.png?raw=true" width="400" />

### Step 3: 
##### *Calculate the SVD*
We need to form matrix A by substracting mean vector *f* from every vector from training matrix. Assume the rank of *A* is *r*.  It can be proved, that the obtained matrix has the following Single Value Decomposition (details in the picture below).

<img src="https://github.com/kategimranova/Face-Recognition-and-Reconstruction-using-SVD/blob/main/images/step3.png?raw=true" width="700" />

### Step 4:
##### *Find eigen faces*
It can be proved that <img src="https://latex.codecogs.com/svg.image?\{u_1,u_2,...,u_r\}" title="\{u_1,u_2,...,u_r\}" /> form orthonormal basis for the system, called ‘face subspace’ or 'Eigen faces'. These basic features will be used to perform each individual face.
<img src="https://github.com/kategimranova/Face-Recognition-and-Reconstruction-using-SVD/blob/main/images/step4.png?raw=true" width="700" />

Some of obtained Eigen-faces are presented below:
![eigenfaces](https://github.com/kategimranova/Face-Recognition-and-Reconstruction-using-SVD/blob/main/images/eigenfaces.png?raw=true)


### Step 5:
##### *For each known individual, compute the coordinate vector (a projection onto base-faces)*:
<img src="https://latex.codecogs.com/svg.image?x_i&space;=&space;{\[u_1,u_2,...,u_r\]}^T(f_i&space;-&space;\overline{f})" title="x_i = {\[u_1,u_2,...,u_r\]}^T(f_i - \overline{f})"/>

This obtained coordinate vector <img src="https://latex.codecogs.com/svg.image?x_i" title="x_i" />  is used then to find which of the training faces best describes the input face f.

### Step 6: 
##### *Finding the closest face*
To find the closest face from training set, first we have to compute projections onto base faces for each face in a train, second, compute the distance of input projection to each training face projection, then simply obtain the index of the minimal distance and predict the input with the label corresponding to the face with this index.



## Testing the application
![test](https://github.com/kategimranova/Face-Recognition-and-Reconstruction-using-SVD/blob/main/images/test.png?raw=true)

# Face reconstruction
The images in our data is not in high resolution so we did not have any problems with the time or memory complexity, so we did not need to compress them, but compact SVD as was already demonstrated on the previous lessons, can help to reduce the size of an image while maintaining the resolution. And here we just wanted to show how does it work practically with different ranks chosen for the decomposition. So, in case of high resolution face images, we could use compact SVD for face recognition.

![reconstruction](https://github.com/kategimranova/Face-Recognition-and-Reconstruction-using-SVD/blob/main/images/reconstruction.png?raw=true)

# Results
The main idea of our algorithm is to project the face that we should recognize on our face-basis and then just find the face from our dataset which projection is the closest one to the projection of face which we want to recognize.

We implemented such face recognition algorithm and face reconstruction from face projection to illustrate capabilities of SVD-based algorithms.

# References

 http://link.springer.com/chapter/10.1007/978-1-4020-6264-3_26
