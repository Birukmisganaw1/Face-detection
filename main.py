import cv2
import os

# Picture's path
path = r'/home/biruk/Desktop/face/Photos'

# loop picture's names that present in Photos folder
for img_r in os.listdir(path):
    # img_path
    img_path = os.path.join(path, img_r)

    # Read the image
    img = cv2.imread(img_path)

    # get the shape of the image
    shape = img.shape
    # get the width of the image
    width = shape[1]
    # get the height of the image
    height = shape[0]

    # calculate the size of the width to resize the image
    re_width = int(width * 0.1)
    # calculate the size of the height to resize the image
    re_height = int(height * 0.1)

    # resize the image
    img_resize = cv2.resize(img, (re_width, re_height))

    # convert the  color image to gray scale image
    gray = cv2.cvtColor(img_resize, cv2.COLOR_BGR2GRAY)

    # load the classifier/cascade
    face_classifier = cv2.CascadeClassifier('haarcascade_frontalface_default.xml')

    # If faces are detected in an image, it returns the positions of detected face as rect(x,y,w,h).
    face_rectangle = face_classifier.detectMultiScale(gray, scaleFactor=1.05, minNeighbors=6)

    # print number of face detected
    print(f'Number of face detect =  {len(face_rectangle)}')

    # get the x,y,width, height value

    for (x, y, w, h) in face_rectangle:
        # draw rectangle
        cv2.rectangle(img_resize, (x, y), (x + w, x + h), (0, 255, 0), thickness=2)

    cv2.imshow('img', img_resize)

    cv2.waitKey(0)

