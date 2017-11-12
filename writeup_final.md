#**Vehicle Detection Project**

---

**Vehicle Detection Project**

The goals / steps of this project are the following:

* Perform a Histogram of Oriented Gradients (HOG) feature extraction on a labeled training set of images and train a classifier Linear SVM classifier
* Optionally, you can also apply a color transform and append binned color features, as well as histograms of color, to your HOG feature vector. 
* Note: for those first two steps don't forget to normalize your features and randomize a selection for training and testing.
* Implement a sliding-window technique and use your trained classifier to search for vehicles in images.
* Run your pipeline on a video stream (start with the test_video.mp4 and later implement on full project_video.mp4) and create a heat map of recurring detections frame by frame to reject outliers and follow detected vehicles.
* Estimate a bounding box for vehicles detected.

[//]: # (Image References)
[image1]: ./examples/car_examples.png
[image2]: ./examples/notcar_examples.png
[image3]: ./examples/car_not_car_img.png
[image4]: ./examples/car_hog.png
[image5]: ./examples/notcar_hog.png
[image6]: ./examples/sliding_window_scales.png
[image7]: ./examples/output_boxes.png
[image8]: ./examples/test_outputs.png
[video1]: ./test_videos_output/project_video.mp4
[video1]: ./test_videos_output/challenge_video.mp4

## [Rubric](https://review.udacity.com/#!/rubrics/513/view) Points
### Here I will consider the rubric points individually and describe how I addressed each point in my implementation.  

---

### Data Exploration
The data set I used for this project contained a sample of 8792  cars and 8968  non-cars.
All the images were of size:  (64, 64, 3)

Below is a list of 10 random car images:

![alt text][image1]

Below is a list of 10 random images that aren't cars:

![alt text][image2]
 
### Histogram of Oriented Gradients (HOG)

#### 1. Extracting HOG features from the training images.

The code for extracting the Hog features is in the cell 4 of the IPython notebook.  

I started by reading in all the `vehicle` and `non-vehicle` images.  Here is an example of one of each of the `vehicle` and `non-vehicle` classes:

![alt text][image3]

I then explored different color spaces and different hog parameters (`orientations`, `pixels_per_cell`, and `cells_per_block`).  I grabbed random images from each of the two classes and displayed them to get a feel for what the `skimage.hog()` output looks like.

Here is an example using the `YCrCb` color space and HOG parameters of `orientations=8`, `pixels_per_cell=(8, 8)` and `cells_per_block=(2, 2)`:


![alt text][image4]

![alt text][image5]

#### 2. Settling on the final choice of HOG parameters.

I primarily experimented with number of orientations and pixels per cell. 
I trained the SVC and ran it on a a small clip from the video file near the bridge as I was not able to conclude the performance by looking at just the accuracy of the test set.
Increasing the number of orientations leads to lot of false positives while low orientations missed out on detecting cars on few frames.

I tried with values from 9 to 14 and settled on 11 for the number of orientations.
I also experimented with 16 pixels per cell to increase throughput when I used RBF for SVM.

Final Hog parameters:
* Orientations: 11
* Pixels per cell: (8,8)
* Cells per block: (2,2)


#### 3. Classifier Training using HOG features.

I used the Grid search to arrive at the choice of the classifier. 
I trained a linear SVM with C value of 1 using the Hog features. 
The features were normalized to zero mean and unit variance before training the classifier.
I also explored the color features but it ended up detecting lot of false positives.

### Sliding Window Search

#### 1. Sliding window search.

The Helper Functions for Sliding Window Search were mostly used from the class exercises. 
The find_cars method extracts the hog features only once and then sub sampled to get all overlapping windows.
I search only the bottom half of the frame as the cars of interest appear only there. It also eliminates tree tops and other noisy regions.
I use 4 different scale values(1, 1.5, 2, 3) to search the frame. The output region of these frames are shown below.
As a follow up, I will try to optimize the areas further. 
Below images displays the areas the car is searched at different scales.

![alt text][image6]

#### 2. Car Detection Pipeline

I used the YCrCb 3-channel after experimenting with various color spaces. Although some color spaces like HSV seem to perform better on the training and test set, it performs poorly on the video.
I didnt use the spatially binned color and histograms of color in the feature vector after lots of experimentation, which provided lots of false positives. 


Here are some example images of detection at different scales:

![alt text][image7]

---

### Video Implementation

#### 1. Final video output.  

After running the pipeline on the project video, it detected a lot of false positives. The next point describes on how the false positives were eliminated.
I tried the pipeline on both the project and the challenge video.

Here's a [link to my video result](./test_videos_output/project_video.mp4)

Here's a [link to the challenge video result](./test_videos_output/challenge_video.mp4)


#### 2. Describe how (and identify where in your code) you implemented some kind of filter for false positives and some method for combining overlapping bounding boxes.

I recorded the positions of positive detections in each frame of the video.  From the positive detections I created a heatmap and then thresholded that map to identify vehicle positions.  I then used `scipy.ndimage.measurements.label()` to identify individual blobs in the heatmap.  I then assumed each blob corresponded to a vehicle.  I constructed bounding boxes to cover the area of each blob detected.  


Here are the output of the six test images. The blue boxes are the final detection. The right portion of each output image has three sub images. The top one is the original image. The one below that is the heat map from the boxes detected by various scales. The bottommost one is the heat map after applying a threshold.
The threshold used after experimentation is 6.


![alt text][image8]


---

### Discussion

#### 1. Briefly discuss any problems / issues you faced in your implementation of this project.  Where will your pipeline likely fail?  What could you do to make it more robust?

I first started by exploring various color spaces and the corressponding hog, spatial binning and histograms of colors.
Based on the output of the test images, I narrowed the color spaces to HSV, YCrCb, and YUV. 

I then used linear SVM initially to train the classifier and finalize the parameters for hog, spatial binning and histogram of colors.
I then ran the initial classifier on the project video and noticed the classifier which was trained with YCrCb performed better.
Then I experimented with the spatial binning and histogram of colors and concluded it was not helping as lots of false positives were detected when there were tree shadows on the road.

I used the GridSearchCV to narrow on the SVM parameters. I experimented with different kernels(Linear) and C values(1,5,10,30).
The GridSearchCV narrowed on Linear kernel with a C value of 1. The RBF kernel takes orders of magnitude more time to run than a linear kernel.
 
I finalised on the YCrCb colorspace as it was missing very few car detections. 
The false positives were eliminated using the thresholded heatmap approach.

The pipeline shows poor detection for white cars. Also the detection of small cars at the end mid point of frame is poor.

I would try the following to make my pipeline more rigid:
* The classifier is likely to fail when there are shadows. Explore more color spaces. Try out single channels.
* Augment the data to increase the training set. Include some examples of darker non car images.
* Keep history of car detection and use those locations to do a localized search in the next frame.
* To further eliminate false positives, flag an area as a car only if it has been successfully detected on 5 consecutive frames.
* Optimize the sliding window to reduce the areas searched in the frame. This will improve the throughput. 

