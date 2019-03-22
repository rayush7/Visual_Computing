import cv2
import os
import numpy as np 
import random
from skimage.feature import hog
from skimage.io import imread
from sklearn.externals import joblib
import glob
from config import *
from sklearn.svm import LinearSVC
from skimage.transform import pyramid_gaussian
from imutils.object_detection import non_max_suppression
import imutils
from skimage import color
import matplotlib.pyplot as plt 

#-----------------------------------------------------------------------------------------------------------------------------------------------------

#Important Hyperparameters

#Training Dataset was created from INRIA Person Dataset
pos_im_path = './train_data/images/pos_person'
neg_im_path = '../train_data/images/neg_person'
min_wdw_sz = [16, 32]
step_size = [5, 5]
orientations = 9
pixels_per_cell = [6, 6]
cells_per_block = [2, 2]
pos_feat_path = './train_data/features/pos'
neg_feat_path = './train_data/features/neg'
model_path = './train_data/models/svm_classifier.pickle'
threshold = .3

#---------------------------------------------------------------------------------------------------------------------------------------

def extract_features():
    des_type = 'HOG'

    # If feature directories don't exist, then create them
    if not os.path.isdir(pos_feat_path):
        os.makedirs(pos_feat_path)

    # If feature directories don't exist, then create them
    if not os.path.isdir(neg_feat_path):
        os.makedirs(neg_feat_path)

    print("Calculating the descriptors for the positive samples and saving them")
    for im_path in glob.glob(os.path.join(pos_im_path, "*")):
        #print im_path
        
        im = imread(im_path, as_grey=True)
        if des_type == "HOG":
            fd = hog(im, orientations, pixels_per_cell, cells_per_block, visualize=visualize, transform_sqrt=normalize)
        fd_name = os.path.split(im_path)[1].split(".")[0] + ".feat"
        fd_path = os.path.join(pos_feat_path, fd_name)
        joblib.dump(fd, fd_path)
    print("Positive features saved in {}".format(pos_feat_path))

    print("Calculating the descriptors for the negative samples and saving them")
    for im_path in glob.glob(os.path.join(neg_im_path, "*")):
        im = imread(im_path, as_grey=True)
        if des_type == "HOG":
            fd = hog(im,  orientations, pixels_per_cell, cells_per_block, visualize=visualize, transform_sqrt=normalize)
        fd_name = os.path.split(im_path)[1].split(".")[0] + ".feat"
        fd_path = os.path.join(neg_feat_path, fd_name)
    
        joblib.dump(fd, fd_path)
    print("Negative features saved in {}".format(neg_feat_path))

    print("Completed calculating features from training images")


#---------------------------------------------------------------------------------------------------------------------------------------


def train_svm():

    # SVM Classifier
    clf_type = 'LIN_SVM'
    #clf_type = 'Random_Forest'

    fds = []
    labels = []
    # Load the positive features
    for feat_path in glob.glob(os.path.join(pos_feat_path,"*.feat")):
        fd = joblib.load(feat_path)
        fds.append(fd)
        labels.append(1)

    # Load the negative features
    for feat_path in glob.glob(os.path.join(neg_feat_path,"*.feat")):
        fd = joblib.load(feat_path)
        fds.append(fd)
        labels.append(0)
    print(np.array(fds).shape,len(labels))
    if clf_type is "LIN_SVM":
        clf = LinearSVC()
        print("Training a Linear SVM Classifier")
        clf.fit(fds, labels)

        # If feature directories don't exist, create them
        if not os.path.isdir(os.path.split(model_path)[0]):
            os.makedirs(os.path.split(model_path)[0])
        joblib.dump(clf, model_path)
        print("Classifier saved to {}".format(model_path))

#---------------------------------------------------------------------------------------------------------------------------------------

def sliding_window(image, window_size, step_size):
    '''
    This function returns a patch of the input 'image' of size 
    equal to 'window_size'. The first image returned top-left 
    co-ordinate (0, 0) and are increment in both x and y directions
    by the 'step_size' supplied.

    So, the input parameters are-
    image - Input image
    window_size - Size of Sliding Window 
    step_size - incremented Size of Window

    The function returns a tuple -
    (x, y, im_window)
    '''
    for y in range(0, image.shape[0], step_size[1]):
        for x in range(0, image.shape[1], step_size[0]):
            yield (x, y, image[y: y + window_size[1], x: x + window_size[0]])

def detector(filename,frame_no,show_result):
    im = cv2.imread(filename)
    im = imutils.resize(im, width = min(400, im.shape[1]))
    min_wdw_sz = (64, 128)
    step_size = (5, 5)
    downscale = 1.25

    clf = joblib.load(model_path)

    #List to store the detections
    detections = []
    #The current scale of the image 
    scale = 0

    for im_scaled in pyramid_gaussian(im, downscale = downscale):
        #The list contains detections at the current scale
        if im_scaled.shape[0] < min_wdw_sz[1] or im_scaled.shape[1] < min_wdw_sz[0]:
            break
        for (x, y, im_window) in sliding_window(im_scaled, min_wdw_sz, step_size):
            if im_window.shape[0] != min_wdw_sz[1] or im_window.shape[1] != min_wdw_sz[0]:
                continue
            im_window = color.rgb2gray(im_window)
            fd = hog(im_window, orientations, pixels_per_cell, cells_per_block, visualize=visualize, transform_sqrt=normalize)

            fd = fd.reshape(1, -1)
            pred = clf.predict(fd)

            if pred == 1:
                
                if clf.decision_function(fd) > 0.5:
                    detections.append((int(x * (downscale**scale)), int(y * (downscale**scale)), clf.decision_function(fd), 
                    int(min_wdw_sz[0] * (downscale**scale)),
                    int(min_wdw_sz[1] * (downscale**scale))))
                 

            
        scale += 1

    clone = im.copy()

    for (x_tl, y_tl, _, w, h) in detections:
        cv2.rectangle(im, (x_tl, y_tl), (x_tl + w, y_tl + h), (0, 255, 0), thickness = 2)

    rects = np.array([[x, y, x + w, y + h] for (x, y, _, w, h) in detections])
    sc = [score[0] for (x, y, score, w, h) in detections]

    #print('Type SC: ',type(sc))

    if(len(sc)!=0):
       #print("sc: ", sc)
       #print('Type of SC: ',type(sc))
       sc = np.array(sc)

       pick = non_max_suppression(rects, probs = sc, overlapThresh = 0.3)
       #print("shape, ", pick.shape)

       bb_id = 1
       frame_sol = []

       for(xA, yA, xB, yB) in pick:
       	   cv2.rectangle(clone, (xA, yA), (xB, yB), (0, 255, 0), 2)
       	   temp_sol = [frame_no, bb_id, xA, yA, xB-xA, yB-yA]
       	   bb_id = bb_id + 1
       	   frame_sol.append(temp_sol)


       if show_result:
          #plt.axis("off")
          #plt.imshow(cv2.cvtColor(im, cv2.COLOR_BGR2RGB))
          #plt.title("Raw Detection before NMS")
          #plt.show()

          plt.axis("off")
          plt.imshow(cv2.cvtColor(clone, cv2.COLOR_BGR2RGB))
          plt.title("Final Detections after applying NMS")
          plt.show()

       return temp_sol

    else:
       return -1

#def test_folder(data_root, _W, _H, _N):

#    filenames = glob.iglob(os.path.join(data_root, '*'))
#    for filename in filenames:
#        detector(filename)

def test(data_root, _W, _H, _N,show_result):

	test_image = os.listdir(data_root)
	test_image = [os.path.join(data_root,x) for x in test_image]

	test_image.sort()
	sol = []
	frame_no = 1

	for image_name in test_image:
		print('Processing frame', frame_no, ' out of', _N, ' frames')
		frame_sol = detector(image_name,frame_no,show_result)
		if frame_sol != -1:
		   sol.append(frame_sol)

		frame_no = frame_no + 1

	return sol

#---------------------------------------------------------------------------------------------------------------------------------------

def pedestrian_ml(data_root, _W, _H, _N,show_result):
	""" Input Argument : data_root : test data set
						_W : width of frame
						_H : height of frame
						_N : number of frames
	"""
	

	#Extract HOG Features
	extract_features()

	#Train SVM
	train_svm()

	#Test on the given dataset
	sol = test(data_root, _W, _H, _N,show_result)

	return sol
