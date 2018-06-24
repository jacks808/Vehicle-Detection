## Writeup Template
### You can use this file as a template for your writeup if you want to submit it as a markdown file, but feel free to use some other method and submit a pdf if you prefer.

---

**Vehicle Detection Project**

The goals / steps of this project are the following:

* Perform a Histogram of Oriented Gradients (HOG) feature extraction on a labeled training set of images and train a classifier Linear SVM classifier
* Optionally, you can also apply a color transform and append binned color features, as well as histograms of color, to your HOG feature vector. 
* Note: for those first two steps don't forget to normalize your features and randomize a selection for training and testing.
* Implement a sliding-window technique and use your trained classifier to search for vehicles in images.
* Run your pipeline on a video stream (start with the test_video.mp4 and later implement on full project_video.mp4) and create a heat map of recurring detections frame by frame to reject outliers and follow detected vehicles.
* Estimate a bounding box for vehicles detected.

[//]: # "Image References"
[image1]: ./examples/car_not_car.png
[image2]: ./examples/HOG_example.jpg
[image3]: ./examples/sliding_windows.jpg
[image4]: ./examples/sliding_window.jpg
[image5]: ./examples/bboxes_and_heat.png
[image6]: ./examples/labels_map.png
[image7]: ./examples/output_bboxes.png
[video1]: ./project_video.mp4

## [Rubric](https://review.udacity.com/#!/rubrics/513/view) Points
### Here I will consider the rubric points individually and describe how I addressed each point in my implementation.  

---
### Writeup / README

#### 1. Provide a Writeup / README that includes all the rubric points and how you addressed each one.  You can submit your writeup as markdown or pdf.  [Here](https://github.com/udacity/CarND-Vehicle-Detection/blob/master/writeup_template.md) is a template writeup for this project you can use as a guide and a starting point.  

You're reading it!

### Histogram of Oriented Gradients (HOG)

#### 1. Explain how (and identify where in your code) you extracted HOG features from the training images.

The code for this step is contained in the[`main_hog.py`](https://github.com/jacks808/Vehicle-Detection/blob/master/main_hog.py) .  

I started by reading in all the `vehicle` and `non-vehicle` images.  Here is an example of one of each of the `vehicle` and `non-vehicle` classes:

![](https://ws4.sinaimg.cn/large/006tKfTcly1fsl84dau35j310g0pyago.jpg)

I then explored different color spaces and different `skimage.hog()` parameters (`orientations`, `pixels_per_cell`, and `cells_per_block`).  I grabbed random images from each of the two classes and displayed them to get a feel for what the `skimage.hog()` output looks like.



Here is an example using the `YCrCb` color space and HOG parameters of `orientations=9`, `pixels_per_cell=(8, 8)` and `cells_per_block=(2, 2)`:

![](https://ws3.sinaimg.cn/large/006tKfTcly1fsl8mwd110j30zy0ugaix.jpg)

#### 2. Explain how you settled on your final choice of HOG parameters.

I tried various combinations of parameters and final choose this parameter: 

```python
'color_space': 'YCrCb',  # Can be RGB, HSV, LUV, HLS, YUV, YCrCb
'orient': 9,  # HOG orientations
'pix_per_cell': 8,  # HOG pixels per cell
'cell_per_block': 2,  # HOG cells per block
'hog_channel': 'ALL',  # Can be 0, 1, 2, or "ALL"
'spatial_size': (32, 32),  # Spatial binning dimensions
'hist_bins': 16,  # Number of histogram bins
'spatial_feat': True,  # Spatial features on or off
'hist_feat': True,  # Histogram features on or off
'hog_feat': True,  # HOG features on or off
```

which is shown at [here](https://github.com/jacks808/Vehicle-Detection/blob/master/main_hog.py#L24)

because use this parmeter, the `svm` will get the best predict accuracy: `99.21%`

![](https://ws2.sinaimg.cn/large/006tKfTcly1fsl8w1faomj30ee030t9i.jpg)

#### 3. Describe how (and identify where in your code) you trained a classifier using your selected HOG features (and color features if you used them).

I trained a linear SVM using a lot of feature as shown bellow:

1. color_space: 'YCrCb'
2. spatial feature with spatil size: `(32, 32)`
3. hist feature with hist_bins: `16`
4. hog feature with all `RGB` channel, and `orient = 9` , `pix_per_cell = 8`, `cell_per_block = 2`

ps: all of this code can be found at: `extract_features` function in `lesson_functions.py`

### Sliding Window Search

#### 1. Describe how (and identify where in your code) you implemented a sliding window search.  How did you decide what scales to search and how much to overlap windows?

I use the slide window code that provided by Udacity. And I decided to search window positions at height `400` to `600`, and  `1.5 scale` , use this subimage to search cars and came up with this :

![image-20180623174233367](https://ws2.sinaimg.cn/large/006tKfTcly1fsl94nfgpxj31kw0yfhdu.jpg)

May be there is some bug of my `svc`. After fix that. I got this :

![image-20180623174400422](https://ws4.sinaimg.cn/large/006tKfTcly1fsl9643o1dj31kw0yfe82.jpg)

The `svc` can classify the car, but there is to many window there. So I decide to add `heat map ` to fix this, here is how the heat map work:

1. Find all boxes in a image that contains a car
2. Calcluate all box heat value
3. only output the heat greater than threshold boxes

Here is the code:

![image-20180623174721564](https://ws3.sinaimg.cn/large/006tKfTcly1fsl99lbj6qj317e0kmwj9.jpg)

#### 2. Show some examples of test images to demonstrate how your pipeline is working.  What did you do to optimize the performance of your classifier?

![image-20180623174956348](https://ws2.sinaimg.cn/large/006tKfTcly1fsl9cafftkj31kw0w07wi.jpg)

I choose differnt `scale` from `0.5` to `3.0` , but I found that if the window is too small. the classifier will get a lot of wrong window such as:

![image-20180623180454121](https://ws3.sinaimg.cn/large/006tKfTcly1fsl9rurl6qj31kw0w0b2a.jpg)

But the small image of a car could never appear there. So I decide to use scale `1.0` and `1.5`to find cars. There is a benefit of choose that value:

> Less image: because only use 1.0 and 1.5  scale to search cars, The amount of calculation is less than scale : `0.5`, `1.0`, `1.5`, `2.0`, `2.5`

### Video Implementation

#### 1. Provide a link to your final video output.  Your pipeline should perform reasonably well on the entire project video (somewhat wobbly or unstable bounding boxes are ok as long as you are identifying the vehicles most of the time with minimal false positives.)
Here's a [link to my video result](./project_video.mp4)



#### 2. Describe how (and identify where in your code) you implemented some kind of filter for false positives and some method for combining overlapping bounding boxes.

I use `MoviePy` to handle video data:

```python
# output processed image
clip1 = VideoFileClip(filename="./project_video.mp4")
white_clip = clip1.fl_image(process_image)
white_clip.write_videofile("./out_{}.mp4".format(time.time()), audio=False)
```

Here is the `process_image` function: 

![image-20180624112421097](https://ws4.sinaimg.cn/large/006tNc79ly1fsm3tfmjtfj317g0u0wku.jpg)

First of all, I use `svc` to find all possible car bonding box in different scale value. Line: 328~331

![image-20180624112705131](https://ws3.sinaimg.cn/large/006tNc79ly1fsm3w8glfgj31kw0yfe82.jpg)

Use heat map to find out where is the most possible position of a car. Line: 334~335

![image-20180624112729702](https://ws4.sinaimg.cn/large/006tNc79ly1fsm3wmwspjj310m0s075n.jpg)

Apply threshold to filter only appear once car image

![image-20180624112748199](https://ws1.sinaimg.cn/large/006tNc79ly1fsm3wyomlqj310a0rgabe.jpg)

Finally use  `scipy.ndimage.measurements.label()` to identify individual blobs in the heatmap.

Line: 345

![image-20180624112837974](https://ws3.sinaimg.cn/large/006tNc79ly1fsm3xtpyq7j31080rah16.jpg)

---

### Discussion

#### 1. Briefly discuss any problems / issues you faced in your implementation of this project.  Where will your pipeline likely fail?  What could you do to make it more robust?

Here I'll talk about the approach I took, what techniques I used, what worked and why, where the pipeline might fail and how I might improve it if I were going to pursue this project further.  

* I use Macbook pro to handle this video, but the speed of `process_iamge` is very slow. It take 1 second to process one frame image. process whole video need about 20 minutes. I think there is some problem on the slide window code. I learn some about `YOLO` . Maybe change slide window to `yolo` can make this speed up. 
* Although `Hog` + `SVM` work very well. But I still want to try some `CNNs` such as `AlexNet`, `ResNet`, `GoogLeNet`.
* Add more data. From now on. all of the data is provide by Udacity, But there is many weather, Rain, Cloud, Snow. Different weather means different light condition. Not only weather but also the time of a day make a huge different of the light condition. So maybe I need more data. 

I beleve after doing that. My code will be more robust. 