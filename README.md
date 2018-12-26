# **Traffic Sign Classifier** 

---
### Overview
In this project, deep neural networks and convolutional neural networks will be used to classify traffic signs. Specifically, models will be trained to classify traffic signs from the [German Traffic Sign Dataset.](http://benchmark.ini.rub.de/?section=gtsrb&subsection=dataset) The trained model will then be used to predict traffic signs found on the web. This process can be extended to other Traffic Sign Datasets and explored with local traffic signs if sufficient training dataset can be collected and lablled.

[//]: # (Image References)

[image1]: output_images/data_distribution_comparision.png "data_distribution_comparision"
[image2]: output_images/pre-processed_original_data_set_edited.png "pre-processed_original_data_set"
[image3]: output_images/transformation_testing_7.png "transformation_testing_7"
[image4]: output_images/aug_y_train_2_distribution_comparision.png "aug_y_train_2_distribution_comparision"
[image5]: output_images/aug_y_valid_1_distribution_comparision.png "aug_y_valid_1_distribution_comparision"
[image6]: output_images/sample_unedited_test_images.png "sample_unedited_test_images"
[image7]: output_images/sample_resized_test_images.png "sample_resized_test_images"
[image8]: output_images/sample_labelled_test_images.png "sample_labelled_test_images"
[image9]: output_images/sample_pre-processed_test_image.png "sample_pre-processed_test_image"
[image10]: output_images/softmax_probabilities_visualization.png "softmax_probabilities_visualization"
[image11]: output_images/original_LeNet_model_architecture.png
[image12]: output_images/modifiedLeNet.jpeg 


---
## 1. Files Submitted

The required files can be referenced at :
* [Jupyter Notebook with Code](Traffic_Sign_Classifier.ipynb)
* [HTML output of the code](Traffic_Sign_Classifier_Final.html)
* [Helpers.py](helpers.py)
* [Dataset](../data)
* [Augmented Dataset](../aug_data)
* [Test Images from Web](../my_test_data)


---
## 2. Dataset Exploration

### 2.1 Dataset Summary

* Number of training examples = 34799
* Number of validation examples = 4410
* Number of testing examples = 12630
* Image data shape = (32, 32, 3)
* Number of classes = 43

### 2.2 Exploratory Visualization

From the plot below it is very evident that there is an uneven distribution of data across different classes in the training, validation and testing dataset. This extreme variation in the number of images in the datasets can result in the model being trained to develop a bias towards the classes with propotionately higher data.

![alt text][image1]

---
## 3. Design and Test a Model Architecture

### 3.1 Preprocessing

The Preprocessing of dataset involved the following two steps:

* **Converting to grayscale**
    * Each traffic sign has a very unique geometric shapes and outlines which them distinguishable even in grayscale.
    * This aspect could help the model better study the traffic sign irrespective of the colors and lighting conditions.
    * Also, reducing the images to grayscale reduces the training time which allows more time for experimentation.
    * Infact, this worked very well for Sermanet and LeCun as described in their traffic sign classification [article.](https://www.researchgate.net/publication/224260345_Traffic_sign_recognition_with_multi-scale_Convolutional_Networks)
    

* **Normalizing the data to the range (-1,1)**
    * The dataset was normalized so that it will bring the mean and variance of all images between a common range of (-1,1) preferably centered about 0. This helps the optimizer in converging with the intial random weights that are choosen from the normal distribution using a single learning rate.

    * **Before Normalization:**
        * Training data mean:  82.677589037
        * Validation data mean: 83.5564273756
        * Testing data mean: 82.1484603612

    * **After Normalization:**
        * Training data mean:  -0.354081335648
        * Validation data mean: -0.347215411128
        * Testing data mean: -0.358215153428

![alt text][image2]


### 3.2 Model Architecture
	
I firstly implemented the same architecture from the LeNet Lab as shown below, with no changes expect for the Softmax output layers to 43. To furhter enhance the model I modified the layers taking inspiration from the Sermanet/LeCun model as shown below from their traffic sign classifier paper. 
**My final model consisted of the following layers:**

| Layer         		|     Description	        					| 
|:---------------------:|:---------------------------------------------:| 
| Input         		| 32x32x1 Gray image   							| 
| Convolution 5x5      	| 1x1 stride, VALID padding, outputs 28x28x48 	|
| RELU					|												|
| Max pooling 2x2	    | 2x2 stride, VALID padding,  outputs 14x14x48 	|
| Convolution 5x5	    | 1x1 stride, VALID padding, outputs 10x10x96  	|
| RELU					|												|
| Max pooling 2x2	    | 2x2 stride, VALID padding,  outputs 5x5x96 	|
| Convolution 3x3	    | 1x1 stride, VALID padding, outputs 3x3x172  	|
| RELU					|												|
| Max pooling 2x2	    | 1x1 stride, VALID padding,  outputs 2x2x172 	|
| Flatten       	    | outputs 1x688                             	|
| Fully connected		| outputs 84       								|
| RELU					|												|
| Softmax				| outputs 43        							|


**Original LeNet Model Architecture**
![alt text][image11]

**Modified LeNet Model Architecture**
Adapted from Sermanet/LeCunn traffic sign classification journal article
![alt text][image12]


### 3.3 Model Training
	
#### 3.3.1 Data Augmentation

As mentioned earlier due to the disproportion of data amongst the various labels in the dataset, the model could have developed a bias towards the labels with significantly higher number of images. To reduce the variation in the size of the dataset per label, new data was generated by means of augmenting the existing data from the respective labels.

As noted earlier, few label in the dataset have fewer samples than others. To overcome this problem of limited quantity and limited diversity of data, we generate(manufacture) our own data with the existing data which we have. This methodology of generating our own data is known as data augmentation.

**The following function was created to augment and store the generated data as a pickle file.**

---
```python
def augment_data(data:dict, labels_data:list, data_set_no:int, data_category:['train','valid'], \
                 limit:['mean','max']):
    ...
    return
```
---

The following section of code from the above mentioned function `augment_data()`is very unique. 
It runs the images through mainly four functions namely - `rand_translate()`, `rand_scale()`, `rand_brighten(img)` and `rand_warp(img)`. Each of these functions transforms the images slightly by randomly chosing a very small value for transformation matrix. This is to ensure that the content of the image still stays intact.

**An Example of how an image gets transformed by each function:**
![alt text][image3]


The following code adds an extra level of randomness by scanning through a randomnly generated list - `sequence` which not only decides which transformations will occur but also decides the order of transformations.

---
```python
# Generate hot-encoded label with 1 having higher probability like [0,1,1,1] or [1,1,0,0]..etc.
translate,scale,brighten,warp = np.random.choice([0,1], 4, p=[0.4, 0.6])
sequence = [('translate',translate),('scale',scale),('brighten',brighten),('warp',warp)]

# Randomizing the Sequence of Augmentation
random.shuffle(sequence)

for k in range(4):
    if sequence[k][0] == 'translate' and sequence[k][1] == 1:
        img = rand_translate(img)
    if sequence[k][0] == 'scale' and sequence[k][1] == 1:
        img = rand_scale(img)
    if sequence[k][0] == 'brighten' and sequence[k][1] == 1:
        img = rand_brighten(img)
    if sequence[k][0] == 'warp' and sequence[k][1] == 1:
        img = rand_warp(img)
```
---

**The following graphs show how the Augmented Data generated reduced the extremity in the size of the dataset**
![alt text][image4]
![alt text][image5]

#### 3.3.2 Parameter Tuning
The Hyperparameter values were influenced from the orginal LeNet Architecture but the number of Epochs were gradually increased to 100 when a continuous increase in validation accuracy could be observed with increase in the number of Epochs. Also the Adam optimizer was used as a part of the original LeNet Architecture.

| Hyper Parameter | Value | 
|:---------------:|:-----:| 
| Batch Size      | 128	  | 
| Epochs      	  | 100	  |
| Learning Rate   |	0.001 |
| mu	          | 0 	  |
| Sigma	          | 0.1   |
| keep_prob	      | 0.5   |

| Dataset        | Accuracy | 
|:--------------:|:--------:| 
| Validation Set | 96.66%   |
| Test Set       | 95.5%    |

### 3.4 Solution Approach

I firstly implemented the same architecture from the LeNet Lab, with no changes expect for the Softmax output layers to 43. This model with my few trials of augmented dataset gave a test accuracy of about 93.47%. To furhter enhance the model I modified the layers taking inspiration from the Sermanet/LeCun model from their traffic sign classifier paper. I realized the original LeNet Architecture had a depth of only 6 in the first Convolution layer which might not be enough to segment and group several unique combination of pixels that would be needed for the model to distinguish the complex features in a very small image size. It would have been very time consuming to try several different configurations of layers due to the limitation of GPU, I decided to benchmark the layer configurations from the Sermanet/LeCun model and it proved to be helpful in modifying the orignial LeNet architecture and further improving the test accuracy to **95.5%** 

It has been observed in practice that when using a larger batch there is a significant degradation in the quality of the model, as measured by its ability to generalize. The lack of generalization ability is due to the fact that large-batch methods tend to converge to sharp minimizers of the training function. Hence I kept the batch size to 128 and a batch size smaller was giving out too many simlar images as the number of classes is very high and number of repetitions even after shuffling is extremely high. This repetition could prevent generalization of the model. 

The validation accuracy gradually rises across all the 100 epochs and infact it continued to rise till the 100th epoch as seen in the html code output. This clearly indicates the model wasn't overfitting atleast not yet, it could overfit beyond 100 epochs when the accuracy approaches 100% and it becomes harder for optimizer to find convergence and hence tends to memorize the training data. Unfortunately I couldn't capture a graph while I was training it and now I am out of time to do so again. In anycase since I augmented the data very well before training the model, the model didn't overfit and did quite well. Also with the Adam optimizer being adaptive and fast, model continued to gradully learn without fluctuations in the validation accuracy.

---
## 4. Test a Model on New Images


### 4.1 Acquiring New Images
	
The following 8 images were downloaded from the internet to test the trained model. Since all the images were of different sizes some pre-processing had to be done as under. Since the images get very pixalted and are already little warped, and much brighter and vivid they might be a bit challenging for the model to predict. In addition, the GTSRB dataset states that the images "contain a border of 10 % around the actual traffic sign (at least 5 pixels) to allow for edge-based approaches". Hence i cropped the images to exclude such a border. This could be another source of confusion for the model.

The following perspectives can be considered for the chosen images:

* The Angle of the image
    * The first and the 4th image is a little warpped and tilted. 
        * This is more than the tilt and warpping done in the augmentation of the data. Hence it should provide a challenge for the model to identify.
* The Brightness of the image
    * The first and the last images are comparatively much brighter than any image in the data set.
* The Contrast of the image
    * The contrast of almost all the images is a lot higher than any image in the dataset which could provide a challenge to the model in predicting the changing gradients.
* The jitteriness of the image
    * Almost all the images are very jittered this can cause a lot of confusion to the model specially while convoluting the smaller details.
* Are there any background objects?
    * 4,5,6 imagess have some partial background in them which cause a sudden graident change. Again this can create boundary recognition confusion.

>_**Traffic signs with shadows and things obstructing them partially would be the most challenging images to test the accuracy of the model.**_ 



**Loading Sample Test Images I saved from the web**
![alt text][image6]

**Resizing Images to 32x32x3**
![alt text][image7]

**Labelling the Images**
![alt text][image8]

**Pre-Processing the Images**
![alt text][image9]

### 4.2 Performance on New Images

The model predicted the sample signs downloaded from the internet with 100% accuracy which is better than the validation and test accuracy achieved earlier on the acquired dataset. This accuracy could reduce with increase in the size of the sample dataset but it would be wise to notice that if the model predicts similar easily distinguishable real-world data as the eight new images downloaded then the accuracy could continue to stay very high.

### 4.3 Model Certainty - Softmax Probabilities

The top five softmax probabilities of the predictions on the captured images are outputted and the plot clearly shows the model is always 100% certain of its prediction. Also, it can be observed that the other 4 guesses for each of the 8 images visually appear very similar to the first guess but still it successfully picks the correct one, which means the model is very well trained for real-world prediction.

![alt text][image10]