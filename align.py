
'''
This module is responsible for alignment of face image with
a given painting image. User has to align his nose with the
nose template provided in the video till it flashes green in
color. A series of affine transforms then warps the image to 
align its landmarks with those of the painting image. 

The pre-trained dlib library is used to detect facial landmarks
in both the face and the painting. 

'''


import os 
import cv2
import sys
import numpy as np
import io
import dlib
from matplotlib import pyplot as plt

def findBiggestFace(Img):

    '''
Find largest face in an image.

    '''


    assert Img is not None

    faces = findFaces(Img)
    if (len(faces) > 0) or len(faces) == 1:
        return max(faces, key=lambda rect: rect.width() * rect.height())
    else:
        return None
    
def findFaces(Img):

    '''
Find all face bounding rectangles
in image
    '''

    assert Img is not None

    try:
        return detector(Img, 1)
    except Exception as e:
        print("Warning: {}".format(e))
        # In rare cases, exceptions are thrown.
        return []
    
def findLandmarks(Img, box):
    '''
Find landmark points (pixel values) of various landmarks
The landmark points can be found at this link:
http://www.pyimagesearch.com/wp-content/uploads/2017/04/facial_landmarks_68markup.jpg
    '''

    assert Img is not None
    assert box is not None

    points = predictor(Img, box)
    return points, list(map(lambda p: (p.x, p.y), points.parts()))


def align(img_path):


    '''
Aligns user image with painting image using manual alignment
and facial affine transforms.
params: img_path - path to style image
    '''


    filename1 = img_path
    predictor_path = "shape_predictor_68_face_landmarks.dat"
    global detector, predictor
    detector = dlib.get_frontal_face_detector()
    predictor = dlib.shape_predictor(predictor_path)

    img1 = cv2.imread(filename1)
    box = findBiggestFace(img1)
    points,shape = findLandmarks(img1, box)



    img_cropped = cv2.resize(img1[box.top()-50:box.bottom()+50, box.left()-50:box.right()+50], (500,500))
    #img_cropped = cv2.resize(img1, (500,500))

    box1 = findBiggestFace(img_cropped)
    points1, shape1 = findLandmarks(img_cropped, box1)
    TEMPLATE = np.float32(shape1)



    TEMPLATE = TEMPLATE

    facialPoints = [41, 8, 47]

    imgDim = 500

    cap = cv2.VideoCapture(0)
    i = 0 

    while(True):
        text_box = 255*np.ones((80,400,3))


        ret, frame = cap.read()

        try:
            Img = frame

            if i == 0:
                box_v = findBiggestFace(Img)
                i = 1
            points_v,shape_v = findLandmarks(Img, box_v)

            lol = cv2.resize(Img[box_v.top()-100:box_v.bottom()+100, box_v.left()-100:box_v.right()+100], (500,500))
            img_cropped_v = cv2.resize(Img[box_v.top()-50:box_v.bottom()+50, box_v.left()-50:box_v.right()+50], (500,500))
            img_cropped_v = cv2.resize(Img,(500,500))
            box_v1 = findBiggestFace(img_cropped_v)
            points1_v, shape1_v = findLandmarks(img_cropped_v, box_v1)


            ###PAINTING NOSE###
            for k in range(27,36):
                cv2.circle(img_cropped_v, (int(shape1[k][0]), int(shape1[k][1])), 2, (255,0,0), -1)



            ##################################################
            ############CONDITIONS FOR MOVEMENT ##############
            ##################################################
            font = cv2.FONT_HERSHEY_SIMPLEX

            ############################
            ####HORIZONTAL MOVEMENT#####
            ############################


            if shape1_v[27][0] - shape1[27][0] >2 and shape1_v[28][0] - shape1[28][0] > 2 and shape1_v[29][0] - shape1[29][0]>2             and shape1_v[30][0] - shape1[30][0] > 2:

                font = cv2.FONT_HERSHEY_SIMPLEX
                #### SUBJECTS NOSE###
                
                for k in range(27,31):
                    cv2.circle(img_cropped_v, (int(shape1_v[k][0]), int(shape1_v[k][1])), 2, (0,0,255), -1)

                flag1 = 1


            elif shape1_v[27][0] - shape1[27][0] < -2 and shape1_v[28][0] - shape1[28][0] < -2 and shape1_v[29][0] - shape1[29][0] < -2             and shape1_v[30][0] - shape1[30][0] < -2:  

                #### SUBJECTS NOSE###
                for k in range(27,31):
                	cv2.circle(img_cropped_v, (int(shape1_v[k][0]), int(shape1_v[k][1])), 2, (0,0,255), -1)

                flag1 = 2


            else:

                #### SUBJECTS NOSE###
                
                for k in range(27,31):
                    cv2.circle(img_cropped_v, (int(shape1_v[k][0]), int(shape1_v[k][1])), 2, (0,255,0), -1)

                flag1 = 0


            ###########################################
            ##########VERTICAL MOVEMENT ###############
            ###########################################

            if shape1_v[31][1] - shape1[31][1] >2 and shape1_v[32][1] - shape1[32][1] > 2 and shape1_v[33][1] - shape1[33][1] > 2             and shape1_v[34][1] - shape1[34][1] > 2 and shape1_v[35][1] - shape1[35][1] > 2:
                font = cv2.FONT_HERSHEY_SIMPLEX

                        #### SUBJECTS NOSE###
                
                for k in range(31,36):
                    cv2.circle(img_cropped_v, (int(shape1_v[k][0]), int(shape1_v[k][1])), 2, (0,0,255), -1)
              

                flag2 = 1



            elif shape1_v[31][1] - shape1[31][1] <-2 and shape1_v[32][1] - shape1[32][1] <-2 and shape1_v[33][1] - shape1[33][1] <-2             and shape1_v[34][1] - shape1[34][1] <-2 and shape1_v[35][1] - shape1[35][1] <-2:


                        #### SUBJECTS NOSE###
                for k in range(31, 36):
                    cv2.circle(img_cropped_v, (int(shape1_v[k][0]), int(shape1_v[k][1])), 2, (0,0,255), -1)


                flag2 = 2

            else:

                #### SUBJECTS NOSE###
                for k in range(31,36):
                    cv2.circle(img_cropped_v, (int(shape1_v[k][0]), int(shape1_v[k][1])), 2, (0,255,0), -1)

                flag2 = 0



            if flag1 ==1 and flag2 == 1:
                cv2.putText(text_box, "Look right and up", (5,40), font, 1,(0,0,255),1,cv2.LINE_AA)
            elif flag1 == 1 and flag2 == 2:
                cv2.putText(text_box, "Look right and down", (5,40), font,1,(0,0,255),1,cv2.LINE_AA)
            elif flag1 == 1 and flag2 == 0:
                cv2.putText(text_box, "Look right", (5,40), font, 1,(0,0,255),1,cv2.LINE_AA)
            elif flag1 == 2 and flag2 == 1:
                cv2.putText(text_box, "Look left and up", (5,40), font, 1,(0,0,255),1,cv2.LINE_AA)
            elif flag1 == 2 and flag2 == 2:
                cv2.putText(text_box, "Look left and down", (5,40), font, 1,(0,0,255),1,cv2.LINE_AA)
            elif flag1 ==2 and flag2 == 0:
                cv2.putText(text_box, "Look left", (5,40), font, 1,(0,0,255),1,cv2.LINE_AA)
            elif flag1 == 0 and flag2 == 1:
                cv2.putText(text_box, "Look up", (5,40), font, 1,(0,0,255),1,cv2.LINE_AA)
            elif flag1 == 0 and flag2 == 2:
                cv2.putText(text_box, "Look down", (5,40), font, 1,(0,0,255),1,cv2.LINE_AA)
            elif flag1 == 0 and flag2 == 0:
                cv2.putText(text_box, "Hold pose", (5,40), font, 1,(0,255,0),1,cv2.LINE_AA)


            cv2.imshow('1', img_cropped_v)
            cv2.imshow('text', text_box)
            cv2.imshow('2', img_cropped)


        except Exception as e:
            pass




        if cv2.waitKey(1) & 0xFF == ord('q'):
            cv2.imwrite('img.jpg', lol)
            #cv2.imwrite('unaligned.jpg', img_cropped_v)
            break

    cap.release()
    cv2.destroyAllWindows()




    ############################################
    ############# WARPING IMAGE ################
    ############################################

    facialPoints = [0, 8, 16]
    facialPoints2 = [36, 45, 57]

    img1 = cv2.resize(cv2.imread(filename1), (500,500))
    #img1 = cv2.resize(cv2.imread('img.jpg'), (500,500))


    box = findBiggestFace(img1)
    _, TEMPLATE = findLandmarks(img1, box)
    TEMPLATE = np.float32(TEMPLATE)

    TEMPLATE = TEMPLATE/500

    filename2 = 'img.jpg'

    imgDim = 500

    Img = cv2.resize(cv2.imread(filename2), (500,500))

    box = findBiggestFace(Img)

    _,landmarks = findLandmarks(Img, box)

    npLandmarks = np.float32(landmarks)
    npfacialPoints = np.array(facialPoints)

    H = cv2.getAffineTransform(npLandmarks[npfacialPoints],
                               500*TEMPLATE[npfacialPoints])

    thumbnail = cv2.warpAffine(Img, H, (imgDim, imgDim))



    #######################
    ### Second warping ####
    #######################

    Img = thumbnail

    box = findBiggestFace(Img)

    _,landmarks = findLandmarks(Img, box)

    npLandmarks = np.float32(landmarks)
    npfacialPoints = np.array(facialPoints2)

    H1 = cv2.getAffineTransform(npLandmarks[npfacialPoints],
                               500*TEMPLATE[npfacialPoints])

    thumbnail2 = cv2.warpAffine(Img, H1, (imgDim, imgDim))


    return img1, lol, thumbnail2
	

