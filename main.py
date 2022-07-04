""" Introduction to open cv

OpenCV stands for Open Source Computer Vision. To put it simply, it is a library used for image processing.
 In fact, it is a huge open-source library used for computer
vision applications, in areas powered by Artificial Intelligence or Machine Learning algorithms,
 and for completing tasks that need image processing.

Officially launched in 1999. It's written in c++."""

# import necessary libraries
import cv2
import matplotlib.pyplot as plt
import numpy as np

# read image
"""By using cv2.imread() we can load or read the image . 
This method dose not throw an error even if the file name is wrong 
it returns the None type.It accepts two parameters the first parameter is file name/path , 
the second parameter accept types of image  grayscale or color image. we use 0 for gray image and 1 for color image."""

img = cv2.imread("Photos/cats.jpg", 1)
print(img.shape)


# To display our image we use cv2.imshow() method
# The first parameter will be title shown on image window
# The second parameter is the image parameter.

cv2.imshow("original", img)

#waitkey()
""" function of Python OpenCV allows users to display a window for given milliseconds or until any key is pressed. 
It takes time in milliseconds as a parameter and waits for the given time to destroy the window, 
if we do not  pass any value except zero(as optional we can write the number 0 as the argument) it waits till any key 
is pressed. """

cv2.waitKey()

# Converting the image in to grayscale
"""Grayscale is the process by which an image is converted from a full color to shades of gray(black and white)
But this is one method of converting color image to gray image. We can do this easily, 
when we load/read the image by using imread method we can set the second parameter to 0 .
This will automatically convert the image to gray scale image. 
In openCV a lot of openCV functions actually require you to convert an image to grayscale before it actually operates
on the images. And that's because grayscale images are actually much easier to process mainly because they have less 
information but still important segments of the image"""

gray_image = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
cv2.imshow("gray_image", gray_image)
cv2.waitKey()

# Image Histogram
"""Image histograms are plots that display the frequency distribution of different color intensities. 
Image histograms help visualize the color intensity distribution.
For some images, there might be more pixels of the same or close color intensity. Or in other words,
for some images, the image histogram might be skewed, which sometimes reduces the quality of the image.
The image is either more bright or less bright than necessary.
To correct this, we equalize the histogram, or in simple terms, we try to flatten the histogram.
By doing so, the image's contrast is adjusted, and thus we get a better image."""

# calculating histogram
""" OpenCV provides the function cv2.calcHist to calculate the histogram of an image.
cv2.calcHist(images, channels, mask, bins, ranges)
where:
1. images - is the image we want to calculate the histogram of wrapped as a list, so if our image is in variable image 
we will pass [image]
2. channels - is the the index of the channels to consider wrapped as a list ([0] for gray-scale images as there's only 
one channel and [0], [1] or [2] for color images if we want to consider the channel green, blue or red respectively),
3. mask - is a mask to be applied on the image if we want to consider only a specific region,
4. bins - is a list containing the number of bins to use for each channel,
5. ranges - is the range of the possible pixel values which is [0, 256] in case of RGB color space """

# view histogram for color image  color
color = ("b", "g", "r")

for i, col in enumerate(color):
    histogram = cv2.calcHist([img], [i], None, [256], [0, 256])
    plt.plot(histogram, color = col)
    plt.xlim([0,256])
plt.show()

# implementing histogram equalization
"""It's effortless to equalize a histogram of an image. The function cv2.equalizeHist() does the job for us. 
We have to pass the skewed image as input.The function returns a new image with adjusted contrast.
But to use the equalizeHist() the image must be converted in to grayscale"""
equal_hist_image = cv2.equalizeHist(gray_image)

# let's see the two images
cv2.imshow("histogram equalization image", equal_hist_image)
cv2.waitKey()

# let's compare
"""matplotlib.pyplot.hist(x, bins=None, range=None)"""
# Original image
plt.hist(gray_image.ravel(), 256, [0, 256])
plt.show()

# image with equal contrast/ intensity
plt.hist(equal_hist_image.ravel(), 256, [0, 256])
plt.show()

# destroyAllWindows() will close all open windows.
cv2.destroyAllWindows()

# saving the image
# The first parameter is file name/path where to save the image
cv2.imwrite("Photos/output.jpg", img)
