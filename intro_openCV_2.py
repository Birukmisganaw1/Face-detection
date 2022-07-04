import cv2
import numpy as np
import matplotlib.pyplot as plt

# black image
img = np.zeros((512, 512, 3), np.uint8)

"""Draw a line
cv2.line() method is used to draw a line on any image.

Parameters:
image: It is the image on which line is to be drawn.
start_point: It is the starting coordinates of line. The coordinates are represented as tuples of two values i.e. (X coordinate value, Y coordinate value).
end_point: It is the ending coordinates of line. The coordinates are represented as tuples of two values i.e. (X coordinate value, Y coordinate value).
color: It is the color of line to be drawn. For BGR, we pass a tuple. eg: (255, 0, 0) for blue color.
thickness: It is the thickness of the line in px."""

cv2.line(img, (0,0), (512, 512), (56, 56, 256), 7)
# cv2.imshow("line_image", img)
# cv2.waitKey()

"""Draw rectangle
cv2.rectangle() method is used to draw a rectangle on any image.
Syntax: cv2.rectangle(image, start_point, end_point, color, thickness)
Parameters:
image: It is the image on which rectangle is to be drawn.
start_point: It is the starting coordinates of rectangle. The coordinates are represented as tuples of two values 
i.e. (X coordinate value, Y coordinate value).
end_point: It is the ending coordinates of rectangle. The coordinates are represented as tuples of two values
i.e. (X coordinate value, Y coordinate value).
color: It is the color of border line of rectangle to be drawn. For BGR, we pass a tuple. eg: (255, 0, 0) for blue color.
thickness: It is the thickness of the rectangle border line in px.
Thickness of -1 px will fill the rectangle shape by the specified color."""
cv2.rectangle(img, (100, 100 ), (300, 200), (215, 215, 0), 6)
# cv2.imshow("img", img)
# cv2.waitKey()

""" draw a circle

Syntax: cv2.circle(image, center_coordinates, radius, color, thickness)
Parameters:
image: It is the image on which circle is to be drawn. 
center_coordinates: It is the center coordinates of circle. The coordinates are represented as tuples of two values 
i.e. (X coordinate value, Y coordinate value). 
radius: It is the radius of circle. 
color: It is the color of border line of circle to be drawn. For BGR, we pass a tuple. eg: (255, 0, 0) for blue color. 
thickness: It is the thickness of the circle border line in px. Thickness of -1 px will fill the circle shape by 
the specified color. """

# cv2.circle(img, (100, 333), 56, (255, 0, 0), -1)
# cv2.imshow("circle", img)
# cv2.waitKey()

""" Add a text on the image

Parameters:
image: It is the image on which text is to be drawn.
text: Text string to be drawn.
org: It is the coordinates of the bottom-left corner of the text string in the image. The coordinates are represented as
tuples of two values i.e. (X coordinate value, Y coordinate value).
font: It denotes the font type. Some of font types are FONT_HERSHEY_SIMPLEX, FONT_HERSHEY_PLAIN, , etc.
fontScale: Font scale factor that is multiplied by the font-specific base size.
color: It is the color of text string to be drawn. For BGR, we pass a tuple. eg: (255, 0, 0) for blue color.
thickness: It is the thickness of the line in px.
lineType: This is an optional parameter.It gives the type of the line to be used.
bottomLeftOrigin:* This is an optional parameter. When it is true, the image data origin is at the bottom-left corner.
Otherwise, it is at the top-left corner."""
img = cv2.putText(img, 'open cv', (10, 400), cv2.FONT_HERSHEY_SIMPLEX, 4, (0, 118, 34), 5, cv2.LINE_AA)
cv2.imshow('gi', img)
cv2.waitKey(0)
cv2.destroyAllWindows()
