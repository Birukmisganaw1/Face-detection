import cv2
import numpy as np

# Image manipulations
""" Image transformations 
Transformation are geometric distortions enacted upon an image. We use transformations to correct distortions
or perspective issues from arising from the point of view an image was captured.
Types: 
1, Affine : Affine transformation includes scaling, rotations, translation, key point to notice is that 
in Affine transformations, lines that were parallel in the original images are also parallel in the transformed images.
So parallelism between lines are maintained.
2, Non-Affine/projective transformation : also called Homography : This transformation does not preserve parallel , 
angles and length"""
""" Translations 
Transitions are actually very simple and it's basically moving an image in one directions it can be left, right up down
or even diagonally(If we implement an x, y translations at the same time) 
"""
img = cv2.imread("Photos/lady.jpg")

height, width = img.shape[:2]
qu_height, qu_width = height/4, width/4

T = np.float32([[1, 0, qu_width], [0, 1, qu_height]])
img_translation = cv2.warpAffine(img, T, (width, height))
cv2.imshow("Translation", img_translation)


# Rotate the image
# Getting the center of the picture by calculating
rot = (width // 2, height // 2)
rotation_matrix = cv2.getRotationMatrix2D(rot, 90, .5)
rotated_image = cv2.warpAffine(img, rotation_matrix, (width, height))
cv2.imshow("rotated_image", rotated_image)

"""Notice that all the black space surrounding the image, To overcome that we can use easy function called 
cv2.transpose()"""
img_rot = cv2.transpose(img)
cv2.imshow("transpose", img_rot)

# resize image
resize_width = int(width * 0.7)
resize_height = int(height * 0.7)
resize_image = cv2.resize(img, (resize_width, resize_height))
cv2.imshow("resize", resize_image)

# Image pyramids
# Useful when scaling the images in object detection.
smaller = cv2.pyrDown(img)
larger = cv2.pyrUp(img)
cv2.imshow("original", img)
cv2.imshow("smaller", smaller)
cv2.imshow("larger", larger)

# Cropping images
"""Cropping images refers to extracting a segment of that image. openCV actually doesn't have a function to crop an 
image. cropping  can be easily done using numpy"""

print(img.shape)
# # cropping the original image to height(y) 0 - 400 and width(x) 0 - 600
img_cropped = img[0:400, 0:600]
cv2.imshow('original', img)
cv2.imshow('image cropped', img_cropped)

# Arithmetic operation
"""These are simple operations that allow us to directly add or subtract to the color intensity.
The over all effect is increasing and decreasing brightness. when we perform addition the pixel value must not be
greater than 255 and also when we perform the subtract operation the pixel value must not be less than 0"""

# matrix must have the same dimension with the original image.
M = np.ones(img.shape, dtype="uint8") * 75
# This will increase the brightness of the image
added = cv2.add(img, M)
# This will decrease the brightness of the image
sub = cv2.subtract(img, M)
cv2.imshow("added", added)
cv2.imshow("sub", sub)

# Bitwise operations
"""Bitwise operations function in a binary manner and are represented as grayscale images. A given pixel is turned “off”
if it has a value of zero, and it is turned “on” if the pixel has a value greater than zero."""

# Making a square
square = np.zeros((300, 300), np.uint8)
cv2.rectangle(square, (50, 50), (250, 250), 255, -2)
cv2.imshow("rectangle", square)

# Making ellipse
ellipse = np.zeros((300, 300), np.uint8)
cv2.ellipse(ellipse, (150, 150), (150, 150), 30, 0, 180, 255, -1)
cv2.imshow("ellipse", ellipse)

And = cv2.bitwise_and(ellipse, square)
cv2.imshow("And", And)
Or = cv2.bitwise_or(ellipse, square)
cv2.imshow("or", Or)
Not = cv2.bitwise_not(square)
cv2.imshow("Not", Not)
xor = cv2.bitwise_xor(ellipse, square)
cv2.imshow("xor", xor)

# Blurring
"""Blurring is an operation where we average the pixel within a region(kernel) """
kernel_3x3 = np.ones((3, 3), np.float32) / 9
# we use the cv2.filter2D to convolve with the kernel with an image
blurred = cv2.filter2D(img, -1, kernel_3x3)
cv2.imshow("original", img)
cv2.imshow("blurred", blurred)

# Other method blurring
""" Averaging blurring is  done by convolving the image with a normalized box filter, This methods takes the pixels under
 the box and replaces the central element, Kernel size need to be odd and positive."""
blu = cv2.blur(img, (3, 3))
cv2.imshow("averaging", blu)

# Other method blurring
""" The Gaussian blur feature is obtained by blurring (smoothing) an image using a Gaussian function to reduce the noise
level. It can be considered as a nonuniform low-pass filter that preserves low spatial frequency and reduces image noise
and negligible details in an image."""

gaussian = cv2.GaussianBlur(img, (5, 5), 0)
cv2.imshow("gaussian", gaussian)

# Other method blurring
"""Median blurring takes median all pixels under kernel area and central """

median = cv2.medianBlur(img, 5)
cv2.imshow("img_median", median)

# Other method blurring
""" Bilateral blurring is very effective in noise removal while keeping edges sharp, But it's slower"""
bilateral = cv2.bilateralFilter(img, 9, 75, 75)
cv2.imshow("bilateral", bilateral)

# Sharpening
""" Sharpening is the opposite of blurring, it strengths edges in an image."""
sharp = np.array([[-1, -1, -1],
                  [-1, 9, -1],
                  [-1, -1, -1]])
sharpened = cv2.filter2D(img, -1, sharp)
cv2.imshow("original", img)
cv2.imshow("sharpened", sharpened)

# Thresholding
""" Thresholding is the act of converting an image to binary form. Thresholding is one of the most common (and basic) 
segmentation techniques in computer vision and it allows us  to separate the foreground
(i.e., the objects that we are interested in) from the background of the image.
We have simple thresholding where we manually supply parameters to segment the image — this works extremely well 
in controlled lighting conditions where we can ensure high contrast between  the foreground and background of the image.
Thresholding is the binarization of an image. In general, we seek to convert a grayscale image to a binary image, 
where the pixels are either 0 or 255.
Threshold types:
cv2.THRESH_BINARY: Most common
cv2.THRESH_BINARY_INV : Most Common
cv2.THRESH_TRUNC
cv2.THRESH_TOZERO
cv2.THRESH_TOZERO_INV
Image need to be converted  to grayscale before thresholding.
cv2.threshold(image, threshold value, max value, threshold type) """

gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
# if the pixel in the image is less than 125 set it to 0 if it is above set it 255
ret, thresh = cv2.threshold(gray, 125, 255, cv2.THRESH_BINARY)
cv2.imshow('canny images', thresh)
# # THRESH_BINARY_INV
ret, thresh_two = cv2.threshold(gray, 125, 255, cv2.THRESH_BINARY_INV)
cv2.imshow('INV', thresh_two)

# # THRESH_TRUNC
ret, thresh_three = cv2.threshold(gray, 125, 255, cv2.THRESH_TRUNC)
cv2.imshow('Trunc', thresh_three)

# # THRESH_TOZERO
ret, thresh_four = cv2.threshold(gray, 125, 255, cv2.THRESH_TOZERO)
cv2.imshow('To zero', thresh_four)

# # THRESH_TOZERO_INV
ret, thresh_five = cv2.threshold(gray, 125, 255, cv2.THRESH_TOZERO_INV)
cv2.imshow('To zero INV', thresh_five)

# Adaptive Thresholding

"""One downside of simple thresholding is that we have to manually specify a specific threshold value
.In some cases this might work, in more advanced case this wil not work. So one thing we could do is 
we would essentially let the computer find the optimal threshold value by itself.
There are two methods:
1 cv2.ADAPTIVE_THRESH_MEAN_C
2 cv2.ADAPTIVE_THRESH_GAUSSIAN_C

Both are greate in some case Mean works better and in other case GAUSSIAN works better """

adaptive_thresh_mean = cv2.adaptiveThreshold(gray, 255, cv2.ADAPTIVE_THRESH_MEAN_C, cv2.THRESH_BINARY, 11, 3)
adaptive_thresh_gauss = cv2.adaptiveThreshold(gray, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C, cv2.THRESH_BINARY, 11, 3)
cv2.imshow('gray', gray)
cv2.imshow('mean', adaptive_thresh_mean)
cv2.imshow('gauss', adaptive_thresh_gauss)
cv2.imshow('img', img)

# Dilation and Erosion
""" Dilation : Add pixels to the boundaries of object in an image.
Erosion : Removes pixels at the boundaries of object in an image.
"""

kernel = np.ones((5, 5), np.uint8)
ero = cv2.erode(gray, kernel, iterations=1)
img_dilate = cv2.dilate(gray, kernel, iterations=1)
cv2.imshow("gray", gray)
cv2.imshow("erosion", ero)
cv2.imshow('Dilation', img_dilate)
cv2.waitKey()
cv2.destroyAllWindows()
