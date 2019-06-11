---
layout: post
title: Beautiful or not?
---

To begin with, I must say that the results of this project are not that impressive. Nevertheless there are some interesting approaches in this blog post that are fun to implement. For me, at least.

## Purpose of the project

The goal of this project was measuring image aesthetics with deep learning. I started out with replicating a research paper and afterwards also implemented some simpler solutions as a sanity check. As I work as a data scientist in a company that offers machine learning solutions, the main motivation behind doing this research is to use knowledge from it in a recommendation system for some service that relies on user-generated image base and where showing beautiful photos of items increase the number of transactions.

I aimed to measure the aesthetics of images, mainly by scoring how aesthetically pleasing the image is.

## Introduction to the AVA Dataset

Dataset used for this was Aesthetic Visual Analysis (AVA) dataset [[^fn1]] with ~250 000 images, which additionally contains semantic labels in more than 60 categories, aesthetic scores for each image, and information about the styles of photography of the images. The subset of data that I used, contained only images and their overall ratings. The smallest rating is 1, and the largest is 10. The ratings are provided as a distribution of ratings, describing the amount of times an image received each rating.

## The architecture of neural network

As I was highly inspired by the NIMA research [[^fn2]] paper, I loosely mimicked their network architecture. They used a convolutional neural network previously trained on ImageNet dataset, from where th top layer had been removed and replaced with a new fully connected layer on top of it. I tried out multiple versions of prediction layers, including the one used in the research paper. I freezed the imported model weights and only trained the weights that I had introduced to the model. I added a global average pooling layer between base model and the prediction layer. I used Adam optimizer.

Above described architecture was used in all the tasks described below, with changes to hyperparameters, losses and metrics, and to the number of nodes in the last fully connected layer.

To be more specific - the MobileNetV2 pretrained network was used to initialize the model weights. For implementation I used Python3.6 and TensorFlow2.

### Data preprocessing

The images were preprocessed to match the input to the MobileNetV2 network, thus the RGB channels were normalized to [-1, 1] range. They were also rescaled into shape 256\*256 and afterwards augmented by cropping a 224\*224 piece of it, in order to avoid overfitting to the frames that some photos had. Here is an example of an image and after preprocessing.

![Original photo](/assets/original.png){:width="40%"}
![Processed photo](/assets/processed.png)


## Techniques used in this project
### 1. Multiclass task with EMD loss
#### NIMA paper and method description

![Model architecture in NIMA paper](/assets/architecture.png)

As mentioned above, I have taken some ideas from the NIMA research paper. In that paper, they treated this problem as a classification task where each rating is a separate class and the important part is, that these classes can be ordered. They argued, that formalizing the problem in such way, will yield better results so I decided to test it out. They used Earth Mover's Distance (EMD) as their loss function. It is a loss function that penalizes the distance of misclassification. To do this, the sum of all predictions (mass of distributions) has to be 1, thus they used softmax activation to achieve it.

#### Earth Mover's Distance loss function

![EMD loss](/assets/emd.png){:width="70%"}

The figure above shows the EMD loss, where CDF signifies cumulative distribution function as sums of consecutive *p*-s and *r=2*. The main motivation behind predicting the whole distribution instead of the mean value is that the variance should give additional information about the beauty of the photo. Some images are conventionally beautiful and for some images, people don't really agree on its beauty - these images have a higher variance in ratings.

#### Results

I had hard time getting the model to converge properly. Looking at the loss history of the model on the image below we can see that the mean squared error on the validation set is not above baseline. It means that the model isn't really better than just using simple methods. The baseline that I have used in this is the probability distribution of average ratings. For each rating I found the average and then divided it by the sum of averages. Then I found the MSE between that and the predicted distribution. The model was trained on 2000 images.

![Loss history](/assets/emdloss.png){:width="70%"}

Adding more data made the validation error more unstable. Decreasing the learning rate made the loss decrease more monotonously, however I could not beat predicting the baseline on validation data.

An example of an image, its true probability distribution and predicted distribution.

![Example](/assets/drop.png){:width="40%"}

True mean +- variance : 4.767 +- 1.374

Predicted mean +- variance : 5.428 +- 0.612

<style>
.tablelines table, .tablelines td, .tablelines th {
        border: 1px solid black;
        }
</style>
|                               	| 1    	| 2     	| 3     	| 4     	| 5     	| 6     	| 7     	| 8     	| 9   	| 10    	|
|-------------------------------	|------	|-------	|-------	|-------	|-------	|-------	|-------	|-------	|-----	|-------	|
| number of ratings             	| 2    	| 8     	| 15    	| 54    	| 70    	| 30    	| 3     	| 10    	| 0   	| 1     	|
| true probability distribution 	| 0.01 	| 0.041 	| 0.078 	| 0.280 	| 0.363 	| 0.155 	| 0.016 	| 0.052 	| 0.0 	| 0.005 	|
| predicted distribution        	| 0.00 	| 0.001 	| 0.007 	| 0.017 	| 0.534 	| 0.420 	| 0.021 	| 0.000 	| 0.0 	| 0.000 	|
{: .tablelines}


### 2. Regressing the average rating

At one point I felt like I wanted to do something more straightforward, so I dropped the fancy loss function. Instead, I averaged the ratings and then scaled them to [0,1] range as labels and then used sigmoid activation. I used one label for each image - the average rating and mean squared error as loss and metric.

Following is the distribution of average ratings. We can see that the distribution is normal and centered around a bit over 5.

![Average rating distribution](/assets/histogram_of_ratings_train_150000.png){:width="60%"}


At first the results were obviously bad. The model kept predicting very low probabilites for each image. As can be seen from the figure below, model gave very left-skewed predictions. In the original distribution of average ratings, they were in a normal distribution with 5 (or 0.5 in [0,1] scale) being the average. Even though I only care about the ordering of the images by their average rating, the biggest issue with this prediction distribution was that information about lower ratings disappeared.

![Predictions using sigmoid activation](/assets/sigmoid.png)

Thus I tried another approach of normalizing the labels into [-1,1] and using tanh activation function instead of sigmoid and then rescaling the predictions from [-1,1] to [min_average, max_average]. The rescaling part was not really necessary. The model was trained on 5000 images and these are the best photos among 2000 test images, batch size 128.


#### The highest rated photos according to my model

![top2](/assets/top2.png){:width="30%"}
true        5.180851
predicted  10.000000

![top3](/assets/top3.png){:width="30%"}
true       5.492823
predicted  9.762916

![top5](/assets/top5.png){:width="30%"}
true       6.714286
predicted  9.706500

![top4](/assets/top4.png){:width="30%"}
true       5.157895
predicted  9.704150

![top1](/assets/top1.png){:width="30%"}
true       5.185714
predicted  9.648434

#### The lowest rated photos according to my model

![bad2](/assets/bad2.png){:width="30%"}
true       4.073171
predicted  0.040609

![bad1](/assets/bad1.png){:width="30%"}
true       4.854305
predicted  0.034514

![bad3](/assets/bad3.png){:width="30%"}
true       3.855556
predicted  0.051697

![bad4](/assets/bad4.png){:width="30%"}
true       6.456647
predicted  0.050105

![bad5](/assets/bad5.png){:width="30%"}
true       4.627530
predicted  0.057129


### 3. Binary classification

Since I was not really satisfied with the results, I decided to make the model as robust as possible and turned the problem into a binary classification task. Every image that had average rating above 5 was considered as beautiful and others as not beautiful. I used sigmoid activation. For replicating purposes, I mention that I added L2 regularization with *l = 0.1*. The training data size was 2000 images and batch size 64.

We can see from the graph below that the model that I trained is overfitted on the training data and performs as well as majority voting on the validation dataset.

![Loss history of binary classification](/assets/model-1559743073-binary_crossentropy-binary_accuracy-2000                -0.4-layers3-0.00030000000000000003-64-binary.png)

Here are some examples of images, their true labels and predicted labels. Label 1 means beautiful, 0 means not beautiful.

![Boy](/assets/boy.png){:width="40%"}
True label: 0, predicted: 1

![Sunset](/assets/sunset.png){:width="40%"}
True label: 1, predicted: 1

![Cat](/assets/cat.png){:width="40%"}
True label: 0, predicted: 0

![Tree](/assets/tree.png){:width="40%"}
True label: 0, predicted: 0

We can see that the conventionally beautiful sunset has been predicted as beautiful and other pictures are quite OK as well. However, it is always good to analyse the confusion matrix of a binary classifier. We can see from below, that the model is smarter than just majority voting, however it does get a larger proportion of positive examples correct.

#### Confusion matrix

| true\predicted | 0  	| 1   	|
|----------------|----	|-----	|
| 0       | 94 	| 154 	|
| 1       | 97 	| 455 	|
{: .tablelines}



## Additional comments about what I also tried and noticed

I tried normalizing over columns, such that count of each rating is considered independent and with regards to this ratings scale. I used sigmoid activation. For example, the if the count of ratings for rating 5 varies between 5 and 200, then 200 would be considered as 1 and 5 as 0 and the model would try to predict the rating amount in regard to the overall distribution of such rating. Afterwards, I rescaled the predictions with the maximum and minimum values from the training dataset, however the results were underwhelming, as the model predicted very low values for each rating.

I noticed that the ratings have uneven distributions. For example, 1-s and 10-s are being given less frequently than 5-s, which is the most common rating. I also noticed that some beautiful photos have been given bad ratings just because they have been submitted to the wrong category. The AVA dataset contains ratings from different photo challenges, thus the ratings are also related to the photo's fit with the category and not only about its general aesthetics.

The dataset included some photos that were corrupt. There was a file with suggested training image id-s for small scale training and it contained id-s of images that were not present in the dataset, additionally there were some id-s in the ratings data file that were not included among images. I ignored such files. I chose the training images randomly, since the suggested image id-s had low integrity. The dataset was retrieved from academictorrents.com and this might be the reason why some of the images were missing or corrupt.

### Conclusions

Even though I was hoping to achieve better results, I believe that these methods can be further improved with additional hyperparameter tuning. In the future it would be very useful to use additional metadata about the photos, especially EXIF data.


## References

[^fn1]: [AVA: A Large-Scale Database for Aesthetic Visual Analysis](http://www.cat.uab.cat/Public/Publications/2012/MMP2012/Murray2012.pdf)
[^fn2]: [NIMA: Neural Image Assessment](https://arxiv.org/pdf/1709.05424.pdf)
