import glob
import pickle
import time

import matplotlib.pyplot as plt
from moviepy.editor import VideoFileClip
from scipy.ndimage.measurements import label
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.svm import LinearSVC

from lesson_functions import *

logging.basicConfig(level=logging.INFO,
                    format='[%(levelname)5s]%(asctime)s - %(message)s \t\t@[%(filename)s:%(lineno)d] ',
                    filemode='w')

# define params:
mode = 'pipeline'
mode = 'train'

MODEL_PICKLE = './model.pickle'

feature_extra_param = {
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
    'y_start_stop': [400, 600],  # Min and max in y to search in slide_window()
    'scale': 1.5
}


def load_training_images():
    """
    load car and non-car data
    :return: tuple of (car-data, non-car-data)
    """
    cars = []
    notcars = []

    images = glob.glob('./dataset/*/*/*.png', )
    for image in images:
        if 'non-vehicles' in image:
            notcars.append(image)
        else:
            cars.append(image)

    return cars, notcars


def data_look(car_list, notcar_list):
    """
    look at data info
    :param car_list: cars image list
    :param notcar_list: not cars image list
    :return: a data dict to describe data set
    """
    data_dict = {}
    # Define a key in data_dict "n_cars" and store the number of car images
    data_dict["n_cars"] = len(car_list)

    # Define a key "n_notcars" and store the number of notcar images
    data_dict["n_notcars"] = len(notcar_list)

    # Read in a test image, either car or notcar
    example_img = mpimg.imread(car_list[0])

    # Define a key "image_shape" and store the test image shape 3-tuple
    data_dict["image_shape"] = example_img.shape

    # Define a key "data_type" and store the data type of the test image.
    data_dict["data_type"] = example_img.dtype

    # Return data_dict
    return data_dict


def describe_training_set(cars, notcars):
    """
    describe data set
    :param cars:
    :param notcars:
    :return:
    """
    data_info = data_look(cars, notcars)

    logging.info(
        'Your function returned a count of {} cars and {} non-cars'.format(data_info["n_cars"], data_info["n_notcars"]))
    logging.info('of size: {} and data type: {} '.format(data_info["image_shape"], data_info["data_type"]))

    # Choose random car / not-car indices and plot example images
    car_ind = np.random.randint(0, len(cars))
    not_car_index = np.random.randint(0, len(notcars))

    # Read in car / not-car images
    car_image = mpimg.imread(cars[car_ind])
    not_car_image = mpimg.imread(notcars[not_car_index])

    # Plot the examples
    plt.subplot(121)
    plt.imshow(car_image)
    plt.title('Example Car Image')
    plt.subplot(122)
    plt.imshow(not_car_image)
    plt.title('Example Not-car Image')
    plt.show()


def train():
    """
    train a svm model
    :return:
    """
    cars, notcars = load_training_images()
    describe_training_set(cars, notcars)

    logging.info("begin to extra features for cars")
    car_features = extract_features(cars, feature_extra_param)

    logging.info("begin to extra features for not cars")
    notcar_features = extract_features(notcars, feature_extra_param)

    # Create an array stack of feature vectors
    X = np.vstack((car_features, notcar_features)).astype(np.float64)

    # Define the labels vector
    y = np.hstack((np.ones(len(car_features)), np.zeros(len(notcar_features))))

    # Split up data into randomized training and test sets
    rand_state = np.random.randint(0, 100)
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=rand_state)

    # Fit a per-column scaler
    logging.info("shape of x train : {}".format(X_train.shape))
    X_scaler = StandardScaler().fit(X_train)
    # save X_scaler to pickle file
    with open("./X_scaler.pickle", 'wb') as file:
        pickle.dump(X_scaler, file)

    # Apply the scaler to X
    X_train = X_scaler.transform(X_train)
    X_test = X_scaler.transform(X_test)
    logging.info('Using:{}'.format(feature_extra_param))
    logging.info('Feature vector length: {} '.format(len(X_train[0])))

    # Use a linear SVC
    svc = LinearSVC()

    # Check the training time for the SVC
    t = time.time()
    svc.fit(X_train, y_train)
    t2 = time.time()
    print(round(t2 - t, 2), 'Seconds to train SVC...')

    # Check the score of the SVC
    print('Test Accuracy of SVC = ', round(svc.score(X_test, y_test), 4))

    # Check the prediction time for a single sample
    t = time.time()
    n_predict = 10
    print('My SVC predicts: ', svc.predict(X_test[0:n_predict]))
    print('For these', n_predict, 'labels: ', y_test[0:n_predict])
    t2 = time.time()
    print(round(t2 - t, 5), 'Seconds to predict', n_predict, 'labels with SVC')

    # save svc model to pickle file
    with open(MODEL_PICKLE, 'wb') as clf_pickle_file:
        pickle.dump(svc, clf_pickle_file)


def show_image(image):
    plt.imshow(image)
    plt.show()


def convert2int(value):
    if value is None:
        return 0
    else:
        return int(value)


def find_cars(img, svc, X_scaler, feature_extra_param, scale):
    """
    search cars in `img` with svc
    :param img: 被搜索的图像
    :param feature_extra_param:
    :return: draw image
    """
    ystart = convert2int(feature_extra_param['y_start_stop'][0])
    ystop = convert2int(feature_extra_param['y_start_stop'][1])
    # scale = feature_extra_param['scale']
    orient = feature_extra_param['orient']
    pix_per_cell = feature_extra_param['pix_per_cell']
    cell_per_block = feature_extra_param['cell_per_block']
    spatial_size = feature_extra_param['spatial_size']
    hist_bins = feature_extra_param['hist_bins']

    box_list = []

    # copy a image
    draw_img = np.copy(img)
    img = img.astype(np.float32) / 255

    # 剪裁图像
    img_to_search = img[ystart:ystop, :, :]
    color_transfer_img_to_search = convert_color(img_to_search, conv='RGB2YCrCb')
    if scale != 1:
        img_shape = color_transfer_img_to_search.shape
        dist_size = (np.int(img_shape[1] / scale), np.int(img_shape[0] / scale))
        logging.debug("scale: {}, dist size: {}".format(scale, dist_size))
        color_transfer_img_to_search = cv2.resize(src=color_transfer_img_to_search, dsize=dist_size)

    # ger RGB channel
    ch1 = color_transfer_img_to_search[:, :, 0]
    ch2 = color_transfer_img_to_search[:, :, 1]
    ch3 = color_transfer_img_to_search[:, :, 2]

    # Define blocks and steps as above
    nxblocks = (ch1.shape[1] // pix_per_cell) - cell_per_block + 1
    nyblocks = (ch1.shape[0] // pix_per_cell) - cell_per_block + 1
    nfeat_per_block = orient * cell_per_block ** 2

    # 64 was the orginal sampling rate, with 8 cells and 8 pix per cell
    window = 64
    nblocks_per_window = (window // pix_per_cell) - cell_per_block + 1
    cells_per_step = 2  # Instead of overlap, define how many cells to step
    nxsteps = (nxblocks - nblocks_per_window) // cells_per_step + 1
    nysteps = (nyblocks - nblocks_per_window) // cells_per_step + 1

    # Compute individual channel HOG features for the entire image
    hog1 = get_hog_features(ch1, orient, pix_per_cell, cell_per_block, feature_vec=False)
    hog2 = get_hog_features(ch2, orient, pix_per_cell, cell_per_block, feature_vec=False)
    hog3 = get_hog_features(ch3, orient, pix_per_cell, cell_per_block, feature_vec=False)

    for xb in range(nxsteps):  # slide window with x
        for yb in range(nysteps):  # slide window with y
            ypos = yb * cells_per_step
            xpos = xb * cells_per_step
            # Extract HOG for this patch
            hog_feat1 = hog1[ypos:ypos + nblocks_per_window, xpos:xpos + nblocks_per_window].ravel()
            hog_feat2 = hog2[ypos:ypos + nblocks_per_window, xpos:xpos + nblocks_per_window].ravel()
            hog_feat3 = hog3[ypos:ypos + nblocks_per_window, xpos:xpos + nblocks_per_window].ravel()
            hog_features = np.hstack((hog_feat1, hog_feat2, hog_feat3))  # 5292

            xleft = xpos * pix_per_cell
            ytop = ypos * pix_per_cell

            # Extract the image patch
            subimg = cv2.resize(color_transfer_img_to_search[ytop:ytop + window, xleft:xleft + window],
                                (64, 64))  # 64,64,3

            # Get color features
            spatial_features = bin_spatial(subimg, size=spatial_size)
            hist_features = color_hist(subimg, nbins=hist_bins)

            # Scale features and make a prediction
            feature = np.hstack((spatial_features, hist_features, hog_features)).reshape(1, -1)
            test_features = X_scaler.transform(feature)  # feature.shape: 1, 6108

            # test_features = X_scaler.transform(np.hstack((shape_feat, hist_feat)).reshape(1, -1))
            test_prediction = svc.predict(test_features)

            if test_prediction == 1:
                xbox_left = np.int(xleft * scale)
                ytop_draw = np.int(ytop * scale)
                win_draw = np.int(window * scale)

                top_left = (xbox_left, ytop_draw + ystart)
                bottom_right = (xbox_left + win_draw, ytop_draw + win_draw + ystart)
                box_list.append((top_left, bottom_right))
                cv2.rectangle(draw_img, top_left, bottom_right, (0, 0, 255), 6)

    return box_list


def add_heat(heatmap, bbox_list):
    # Iterate through list of bboxes
    for box in bbox_list:
        # Add += 1 for all pixels inside each bbox
        # Assuming each "box" takes the form ((x1, y1), (x2, y2))
        heatmap[box[0][1]:box[1][1], box[0][0]:box[1][0]] += 1

    # Return updated heatmap
    return heatmap  # Iterate through list of bboxes


def apply_threshold(heatmap, threshold):
    # Zero out pixels below the threshold
    heatmap[heatmap <= threshold] = 0
    # Return thresholded map
    return heatmap


def draw_labeled_bboxes(img, labels):
    # Iterate through all detected cars
    for car_number in range(1, labels[1] + 1):
        # Find pixels with each car_number label value
        nonzero = (labels[0] == car_number).nonzero()
        # Identify x and y values of those pixels
        nonzeroy = np.array(nonzero[0])
        nonzerox = np.array(nonzero[1])
        # Define a bounding box based on min/max x and y
        bbox = ((np.min(nonzerox), np.min(nonzeroy)), (np.max(nonzerox), np.max(nonzeroy)))
        # Draw the box on the image
        cv2.rectangle(img, bbox[0], bbox[1], (0, 0, 255), 6)
    # Return the image
    return img


def process_image(image):
    """
    use slide window to handle image.
    :param image:
    :return:
    """
    # show_image(image)
    final_box_list = []
    for zoom in np.arange(1, 2, 0.5):  # zoom in 1.0 1.5
        box_list = find_cars(image, svc, X_scaler, feature_extra_param, scale=zoom)
        for box in box_list:
            final_box_list.append(box)

    # Add heat to each box in box list
    heat = np.zeros_like(image[:, :, 0]).astype(np.float)
    heat = add_heat(heat, final_box_list)

    # Apply threshold to help remove false positives
    threshold = 4
    heat = apply_threshold(heat, threshold)

    # Visualize the heatmap when displaying
    heatmap = np.clip(heat, 0, 255)

    # Find final boxes from heatmap using label function
    labels = label(heatmap)
    draw_img = draw_labeled_bboxes(np.copy(image), labels)

    return draw_img


if __name__ == '__main__':
    logging.info("mode: " + mode)

    if mode == 'train':
        train()
    elif mode == 'pipeline':
        # load model from pickle file
        svc = None
        with open(MODEL_PICKLE, 'rb') as pickle_file:
            svc = pickle.load(pickle_file)
            logging.info("classifier is: {}".format(svc))

        with open("./X_scaler.pickle", 'rb') as file:
            X_scaler = pickle.load(file)

        # output processed image
        # clip1 = VideoFileClip(filename="./project_video.mp4").subclip(9, 10)
        clip1 = VideoFileClip(filename="./project_video.mp4").subclip(40, 42)
        white_clip = clip1.fl_image(process_image)
        white_clip.write_videofile("./out_{}.mp4".format(time.time()), audio=False)
