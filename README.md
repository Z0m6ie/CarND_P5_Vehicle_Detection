## Vehicle Detection Project
### Project 5
###### David Peabody

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
[image0]: ./test_images/10.png
[image0.5]: ./test_images/image6.png
[image1]: ./output_images/car1.png
[image2]: ./output_images/hog1.png
[image3]: ./output_images/boxes11.png
[image4]: ./output_images/boxes1.png
[image4.1]: ./output_images/boxes3.png
[image4.2]: ./output_images/boxes7.png
[image5]: ./output_images/heat1.png
[image5.1]: ./output_images/heat2.png
[image5.2]: ./output_images/heat3.png
[image7]: ./output_images/final1.png
[video1]: ./output_images/project_video_out.mp4

## [Rubric](https://review.udacity.com/#!/rubrics/513/view) Points
### Here I will consider the rubric points individually and describe how I addressed each point in my implementation.  

---
### Writeup / README

#### 1. Provide a Writeup / README that includes all the rubric points and how you addressed each one.  You can submit your writeup as markdown or pdf.  [Here](https://github.com/udacity/CarND-Vehicle-Detection/blob/master/writeup_template.md) is a template writeup for this project you can use as a guide and a starting point.  

You're reading it!

### Histogram of Oriented Gradients (HOG)

#### 1. Explain how (and identify where in your code) you extracted HOG features from the training images.

The code for this step is contained in the second and third cell of the notebook. `P5-Vehicle-detection.ipynb`

I started by reading in all the `vehicle` and `non-vehicle` images.  Here is an example of one of each of the `vehicle` and `non-vehicle` classes:


| Car       | Not-Car      |
| ------------- |:-------------:|
| ![alt text][image0]    | ![alt text][image0.5]|


I extracted the hog features using SKlearn's `skimage.hog()` function. HOG, Histogram of Oriented Gradients generates a feature from a image by first normalizing the image and then extracting the dominant gradients in each cell within the box. A visualization of the feature looks like.

for ease of being able to see how a hog can represent an image I altered the parameters of the HOG to `4 pixels_per_cell` and `1 cells_per_block`

| Original       | HOG     |
| ------------- |:-------------:|
| ![alt text][image0]    | ![alt text][image2]|


#### 2. Explain how you settled on your final choice of HOG parameters.

I tried various combinations of parameters for the HOG feature as well as the color features. This was mainly done through trial and error on single images as well as trying to maximize the score on the `LinearSVC()` classifier.

The final parameter I settled on were:
```python
color_space = 'YUV' # Can be RGB, HSV, LUV, HLS, YUV, YCrCb
orient = 11  # HOG orientations
pix_per_cell = 16 # HOG pixels per cell
cell_per_block = 2 # HOG cells per block
hog_channel = "ALL" # Can be 0, 1, 2, or "ALL"
spatial_size = (32, 32) # Spatial binning dimensions
hist_bins = 32    # Number of histogram bins
spatial_feat = True # Spatial features on or off
hist_feat = True # Histogram features on or off
hog_feat = True # HOG features on or off
```

#### 3. Describe how (and identify where in your code) you trained a classifier using your selected HOG features (and color features if you used them).

To be able to identify if a car, or part of a car, in a search window, I trained a `LinearSVC` classifier on the car and non-car data. To optimize the training of the algorithm I used `GridsearchCV()` to cross-validate and search for the optimum `C` parameter.

```python
X = np.vstack((car_features, notcar_features)).astype(np.float64)                        
# Fit a per-column scaler
X_scaler = StandardScaler().fit(X)
# Apply the scaler to X
scaled_X = X_scaler.transform(X)
# Define the labels vector
y = np.hstack((np.ones(len(car_features)), np.zeros(len(notcar_features))))
print('Using:',orient,'orientations',pix_per_cell,
    'pixels per cell and', cell_per_block,'cells per block')
# Use a linear SVC
svc = LinearSVC()
parameters = {'C':[0.1, 0.5, 1]}
clf = GridSearchCV(svc, parameters)
clf.fit(scaled_X, y)
```

### Sliding Window Search

#### 1. Describe how (and identify where in your code) you implemented a sliding window search.  How did you decide what scales to search and how much to overlap windows?

Instead of the traditional sliding window search I used the HOG Sub-sampling window search. This only has to extract hog features once and then can be sub-sampled to get all of its overlaying windows. Each window is defined by a scaling factor where a scale of 1 would result in a window that's 8 x 8 cells then the overlap of each window is in terms of the cell distance. This means that a cells_per_step = 2 would result in a search window overlap of 75%. Its possible to run this same function multiple times for different scale values to generate multiple-scaled search windows. I ultimately used 3 scaled with a cells_per_step of 1, 2, 3 respectively.

The Grid looks like:

![alt text][image3]

#### 2. Show some examples of test images to demonstrate how your pipeline is working.  What did you do to optimize the performance of your classifier?

Ultimately I searched on three scales using YUV 3-channel HOG features plus spatially binned color and histograms of color in the feature vector, which provided a nice result.  Here are some example images:

![alt text][image4]
I included an image where an incorrect box was placed because the reality is the search and predict pipeline will make the occasional error. The philosophy is that by using thresholds below we can ignore one off errors.

![alt text][image4.1]
![alt text][image4.2]
---

### Video Implementation

#### 1. Provide a link to your final video output.  Your pipeline should perform reasonably well on the entire project video (somewhat wobbly or unstable bounding boxes are ok as long as you are identifying the vehicles most of the time with minimal false positives.)

Here's a [link to my video result](./output_images/project_video_out.mp4)


#### 2. Describe how (and identify where in your code) you implemented some kind of filter for false positives and some method for combining overlapping bounding boxes.

I passed all the squares from the search grid through the predict function of our classifier. From the positive detections I created a heatmap and then thresholded that map to `2` to identify vehicle positions. I also played with the confidence levels of the predictor to reduce false predictions and ended with ensuring the confidence levels are above `0.3`.

Here's an example result showing the heatmap from a series of frames of video, the result of `scipy.ndimage.measurements.label()` and the bounding boxes then overlaid on the last frame of video:

### Here are 3  heatmaps:

![alt text][image5]

![alt text][image5.1]

![alt text][image5.2]

 I then used `scipy.ndimage.measurements.label()` to identify individual blobs in the heatmap.  I then assumed each blob corresponded to a vehicle.  I constructed bounding boxes to cover the area of each blob detected.

### Here the resulting bounding boxes are drawn onto the last frame in the series including threshold:
![alt text][image7]

---

### Discussion

#### 1. Briefly discuss any problems / issues you faced in your implementation of this project.  Where will your pipeline likely fail?  What could you do to make it more robust?

This pipeline could easily fail due to non traditional vehicle types, here we only trained for cars, what if we encountered a motorbike or a tractor. In addition if the lighting was so that the cars saturation matched that of the environment it may cause trouble.

The pipeline could be made more robust through:

A larger variety of overlapping scales.

Increasing the confidence level requirement on the predictor.

Using a decision tree or something like ADABoost to select the parameters which most effected the prediction to reduce the feature space.
