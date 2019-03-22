import cv2
import os
import numpy as np 
import random

#----------------------------------------------------------------------------------------------------------------------------------------------------------

def pedestrians_bg_sub(data_root, _W, _H, _N):

	''' Return a list of bounding boxes in the format frame, bb_id, x,y,dx,dy '''

	algo = 'KNN'
	kernel = np.ones((1,1),np.uint8)


	#create Background Subtractor objects
	if algo == 'KNN':
		backSub = cv2.createBackgroundSubtractorMOG2()
	elif algo == 'MOG2':
		backSub = cv2.createBackgroundSubtractorKNN()

	imglist = os.listdir(data_root)
	imglist = [os.path.join(data_root,x) for x in imglist]

	imglist.sort()

	N = len(imglist)

	#print(N)
	#print(_N)

	threshold = 50
	aspect_ratio_max = 3.0 #min aspect ratio
	aspect_ratio_min = 1.1 #max aspect ratio
	min_contour_area = 100     #min area 
	max_contour_area = 100000    #max area 

	#Important Info
	sol = []
	frame_id = 0


	for img_name in imglist:

		frame = cv2.imread(img_name,0)
		#cv2.imshow('Frame', frame)

		#Increment the Frame ID	
		frame_id = frame_id + 1
		bb_id = 0

		#update the background model
		fgMask = cv2.Canny(frame, threshold, threshold * 2)
		fgMask = backSub.apply(frame)
		fgMask = cv2.morphologyEx(fgMask, cv2.MORPH_OPEN, kernel)

		

		contours, hierarchy = cv2.findContours(fgMask,cv2.RETR_TREE,cv2.CHAIN_APPROX_SIMPLE)
		contours_poly = [None]*len(contours)
		boundRect = [None]*len(contours)

		for i,c in enumerate(contours):

			contours_poly[i] = cv2.approxPolyDP(c,3, True)
			boundRect[i] = cv2.boundingRect(contours_poly[i])

			color = (0,255,0)
			#print(cv2.contourArea(c))

			box_aspect_ratio = float(boundRect[i][3]/boundRect[i][2])

			if cv2.contourArea(c)>min_contour_area and cv2.contourArea(c)<max_contour_area and box_aspect_ratio > aspect_ratio_min:
				bb_id = bb_id + 1
				cv2.rectangle(frame, (int(boundRect[i][0]), int(boundRect[i][1])), (int(boundRect[i][0]+boundRect[i][2]), int(boundRect[i][1]+boundRect[i][3])), color, 2)
				#temp_sol = [frame_id,bb_id, int(boundRect[i][0]), int(boundRect[i][1]) , int(boundRect[i][0]+boundRect[i][2]), int(boundRect[i][1]+boundRect[i][3])]
				temp_sol = [frame_id,bb_id, int(boundRect[i][0]), int(boundRect[i][1]) , int(boundRect[i][2]), int(boundRect[i][3])]

				sol.append(temp_sol)

		#show the current frame and the fg masks
		cv2.imshow('Frame', frame)
		cv2.imshow('FG Mask', fgMask)
		cv2.waitKey(10)

	cv2.destroyAllWindows()
	#print(sol)
	return sol
