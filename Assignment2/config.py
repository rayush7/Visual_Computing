pos_im_path = '../data/images/pos_person'
neg_im_path = '../data/images/neg_person'
min_wdw_sz = [68, 124]
step_size = [10, 10]
orientations = 9
pixels_per_cell = [6, 6]
cells_per_block = [2, 2]
visualize = False
normalize = True
pos_feat_ph = '../data/features/pos'
neg_feat_ph = '../data/features/neg'
model_path = '../data/models/svm_classifier.pickle'
threshold = .3
