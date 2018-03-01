#Group of 3- Shraddha Sharma, Kushal Dhar and Shubham Gupta

#submitted by Shubham Gupta


import cv2
import numpy as np

flag = True
print 'Enter path of image'
path = raw_input()
img = cv2.imread(path)
while flag == True:
    print 'Choose your task: (Enter 1 to perform task 1)'
    print '1. Display image'
    print '2. Add scalar to image'
    print '3. Subtract scalar from image'
    print '4. Multiply scalar to image'
    print '5. Divide image by scalar'
    print '6. Resize image by 1/2'
    print '7. Exit'
    user_input = raw_input()
    if user_input == '1':
        cv2.imshow('image', img)
        cv2.waitKey(0)
        cv2.destroyAllWindows()
    elif user_input == '7':
        flag = False
    else:
        print 'Enter scalar value'
        scalar = int(raw_input())
        if user_input == '2':
            img_new = cv2.add(img, scalar)
        elif user_input == '3':
            img_new = cv2.subtract(img, scalar)
        elif user_input == '4':
            img_new = cv2.divide(img, scalar)
        elif user_input == '5':
            img_new = cv2.multiply(img, scalar)
        elif user_input == '6':
            img_new = cv2.resize(img, None, fx=0.5, fy=0.5)

        cv2.imshow('image', img_new)
        cv2.waitKey(0)
        cv2.destroyAllWindows()
