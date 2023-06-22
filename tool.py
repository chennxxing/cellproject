import numpy as np
import cv2
import matplotlib.pyplot as plt
from PIL import Image
import os
import scipy.signal
from skimage.segmentation import active_contour
import csv
import shutil
from scipy.ndimage import gaussian_filter, median_filter

from skimage import (
    data, restoration, util
)



def view_result(arg_para):
    file_path_1 = arg_para["file_path_1"]
    file_path_2 = arg_para["file_path_2"]
    begin_index = 1
    end_index = 100
    image_size = 512

    #### find vessel from B channel
    image = input_image_channelB(file_path_2, image_size, begin_index, end_index)
    image1 = cv2.equalizeHist(image)
    # plt.figure(figsize=(8, 8))
    # plt.imshow(image1,cmap='gray')
    circles = cv2.HoughCircles(image1, cv2.HOUGH_GRADIENT, 1, 100,
                               param1=80,
                               param2=15,
                               maxRadius=50,
                               minRadius=20)
    #print(circles[0])

    position = circles[0][0]

    center_x_1 = int(position[0] - position[2] * 2)
    center_x_2 = int(position[0] + position[2] * 2)
    center_y_1 = int(position[1] - position[2] * 2)
    center_y_2 = int(position[1] + position[2] * 2)
    center_axis = [center_x_1, center_x_2, center_y_1, center_y_2]

    #### noise center
    center_x_3 = int(position[0] - position[2] * 0.5)
    center_x_4 = int(position[0] + position[2] * 0.5)
    center_y_3 = int(position[1] - position[2] * 0.5)
    center_y_4 = int(position[1] + position[2] * 0.5)
    center_axis_noise = [center_x_3, center_x_4, center_y_3, center_y_4]

    ### set parameters for vessel part
    image = input_image(file_path_1, image_size, begin_index, end_index, center_axis_noise)
    image_center = image[center_y_1:center_y_2, center_x_1:center_x_2]


    # plt.figure(figsize=(8, 8))
    # plt.imshow(image_center,cmap='gray')

    #image_center = cv2.medianBlur(image_center, 9)
    binary_image = binary_threshold(image_center, 41, arg_para["vessel1"], arg_para["vessel2"])
    image_center = binary_image.astype('uint8')
    # plt.figure(figsize=(8, 8))
    # plt.imshow(binary_image,cmap='gray')

    contours, hierarchy = cv2.findContours(image_center, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    center_set = set()
    final_contours = find_contours(contours, image_size, 800, True, center_set)
    for i in range(len(final_contours)):
        for j in range(len(final_contours[i])):
            final_contours[i][j][0][0] += center_x_1
            final_contours[i][j][0][1] += center_y_1

    for i in final_contours:
        contour_one_1 = [i]
        locations = contours_in(contour_one_1, image_size)
        for j in locations:
            center_set.add(j)


    ### set parameters
    binary_image = binary_threshold(image, 71, arg_para["signal1"], arg_para["signal2"])

    image2 = image.copy()
    image2 = cv2.cvtColor(image2, cv2.COLOR_GRAY2RGB)

    binary_image = binary_image.astype('uint8')
    contours, hierarchy = cv2.findContours(binary_image, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    final_contours2 = find_contours(contours, image_size, 200, False, center_set)
    final_contour = final_contours + final_contours2

    cv2.drawContours(image2, final_contour, -1, (255, 0, 0), 1)
    #plt.figure(figsize=(8, 8))
    #plt.imshow(image2, cmap='gray')
    return image2, final_contour, len(final_contours), center_axis_noise

def analysis_process(arg_dict, point_location, countour_sum, number, denoise_area):
    result_list = []
    for i in range(arg_dict["begin_index"], arg_dict["end_index"] + 1):
        sum_value = 0
        image_num = str(i).zfill(3)
        image_file = arg_dict["file_path_1"] + image_num + '.tif'

        image_single = np.asarray(Image.open(image_file))

        average_noise = np.average(image_single[denoise_area[2]:denoise_area[3], denoise_area[1]:denoise_area[2]])
        image_single = image_single - average_noise
        image_single[image_single < 0] = 0

        # background = restoration.rolling_ball(image_single,radius=100)
        # image_single = image_single - background

        for j in point_location:
            sum_value += image_single[j]
        sum_value = sum_value / len(point_location)
        result_list.append(sum_value)

    mean_value = np.mean(result_list[:20])
    std_value = np.std(result_list[:20])

    result_list = np.array(result_list)
    # result_list_smooth = scipy.signal.medfilt(result_list, 5)

    result_list_smooth = scipy.signal.savgol_filter(result_list, window_length=11, polyorder=3, mode="nearest")

    result_list = list(result_list)
    result_list_smooth = list(result_list_smooth)

    max_value = np.max(result_list_smooth)
    max_index = result_list_smooth.index(max_value)

    result_smooth_index = [i for i in range(len(result_list_smooth)) if
                           result_list_smooth[i] > mean_value + 5 * std_value]

    '''
    if 40 >= max_index >= 20:
        if len(result_smooth_index) > 4 and max_value>1750:
            save_name = arg_dict["output_path"] + 'possible_signal/'
        else:
            save_name = arg_dict["output_path"] + 'others/'
    else:
        save_name = arg_dict["output_path"] + 'others/'
    '''

    figure, axes = plt.subplots(1, 2, constrained_layout=True, figsize=(10, 5))
    l1, = axes[0].plot(range(arg_dict["begin_index"], arg_dict["end_index"] + 1), result_list, color='black')
    l2, = axes[0].plot(range(arg_dict["begin_index"], arg_dict["end_index"] + 1), result_list_smooth, linestyle='--',
                       color='red')
    axes[0].legend(handles=[l1, l2], labels=['original', 'smooth'], loc='best')

    before_simulation = np.average(result_list_smooth[:20])
    peak_value = np.max(result_list_smooth)
    increase_rate = (peak_value - before_simulation) / before_simulation

    # plt.figure(figsize=(8, 8))
    # plt.plot(range(1,201),result_list)
    # plt.savefig(save_name+str(k)+'_hist.png')

    max_value = np.max(result_list)
    max_index = result_list.index(max_value)
    image_num = str(max_index + 1).zfill(3)
    image_file = arg_dict["file_path_1"] + image_num + '.tif'
    image_single = np.asarray(Image.open(image_file)).astype('float')
    image_single = (image_single / np.max(image_single) * 255).astype('uint8')
    cv2.drawContours(image_single, countour_sum, -1, (255, 0, 0), 1)

    axes[1].imshow(image_single, cmap='gray')
    if len(result_smooth_index) <= 4:
        plt.suptitle("Region size is " + str(len(point_location)))
    else:
        plt.suptitle("Region size is " + str(len(point_location)) + ". Peak value is " + str(int(max_value)) +
                     ". Time of response " + str(result_smooth_index[0]) + ". Duration " + str(
            len(result_smooth_index)) + ". Increase rate " + str(increase_rate))

    if len(point_location) > 2500:
        save_name = arg_dict["output_path"] + 'endfoot/'
    elif len(point_location) < 300 or peak_value < 30 or increase_rate < 3:
        save_name = arg_dict["output_path"] + 'others/'
    else:
        save_name = arg_dict["output_path"] + 'cellbody/'

    plt.savefig(save_name + str(number) + '.png')




def save_analysis_result(image, final_contour, arg_dict, length, center_axis_noise):
    if os.path.exists(arg_dict["output_path"]):
        shutil.rmtree(arg_dict["output_path"])

    os.makedirs(arg_dict["output_path"])

    if not os.path.exists(arg_dict["output_path"] + 'cellbody/'):
        os.makedirs(arg_dict["output_path"] + 'cellbody/')

    if not os.path.exists(arg_dict["output_path"] + 'endfoot/'):
        os.makedirs(arg_dict["output_path"] + 'endfoot/')

    if not os.path.exists(arg_dict["output_path"] + 'others/'):
        os.makedirs(arg_dict["output_path"] + 'others/')


    plt.figure(figsize=(8, 8))
    plt.imshow(image, cmap='gray')
    plt.savefig(arg_dict["output_path"] + 'summary.png')


    os.makedirs(arg_dict["output_path"]+'locations/')


    number = 0
    location_sum = []
    countour_sum = []

    for k in range(length):
        test_sample = [final_contour[k]]
        location = contours_in(test_sample, arg_dict["image_size"])
        location_sum = location_sum + location
        countour_sum = countour_sum + test_sample

    np.save(arg_dict["output_path"] + 'locations/' + str(number) + '.npy', location_sum)

    analysis_process(arg_dict, location_sum, countour_sum, number, center_axis_noise)


    for k in range(length,len(final_contour)):
        number = k - length + 1
        test_sample = [final_contour[k]]
        location = contours_in(test_sample, arg_dict["image_size"])
        np.save(arg_dict["output_path"]+'locations/'+str(number)+'.npy',location)

        analysis_process(arg_dict, location, test_sample, number, center_axis_noise)


    image = input_image_channelB(arg_dict["file_path_2"], arg_dict["image_size"], arg_dict["begin_index"], arg_dict["end_index"])
    image1 = cv2.equalizeHist(image)
    # plt.figure(figsize=(8, 8))
    # plt.imshow(image1,cmap='gray')
    circles = cv2.HoughCircles(image1, cv2.HOUGH_GRADIENT, 1, 100,
                               param1=80,
                               param2=15,
                               maxRadius=50,
                               minRadius=20)
    circle = circles[0][0]

    #s = np.linspace(0, 2 * np.pi, 400)
    #r = circle[0].astype('int') + (circle[2] * 2).astype('int') * np.sin(s)
    #c = circle[1].astype('int') + (circle[2] * 2).astype('int') * np.cos(s)
    #init = np.array([c, r]).T

    image_size = 512

    x_begin = max(0, int(circle[1] - 2 * circle[2]))
    x_end = min(image_size - 1, int(circle[1] + 2 * circle[2]))
    y_begin = max(0, int(circle[0] - 2 * circle[2]))
    y_end = min(image_size - 1, int(circle[0] + 2 * circle[2]))
    area_list = []

    for i in range(arg_dict["begin_index"], arg_dict["end_index"]+1):
        # print(i)
        image_num = str(i).zfill(3)
        image_file = arg_dict["file_path_2"] + image_num + '.tif'
        image_single = np.asarray(Image.open(image_file)).astype('float')
        image_single = image_single / 65535 * 255
        image_single = image_single.astype('uint8')

        image_area = image_single[x_begin: x_end, y_begin:y_end]
        image_area = gaussian_filter(image_area, sigma=3)
        image_area = cv2.equalizeHist(image_area)
        image_area[image_area >= 50] = 255
        image_area[image_area < 50] = 0
        image_area = cv2.medianBlur(image_area, 3)

        contours, hierarchy = cv2.findContours(image_area, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        area_single = 0
        area_single_max = 0
        for j in range(len(contours)):
            if len(contours[j]) > 20:
                area_single = len(contours_in([contours[j]], image_area.shape[0]))
                area_single_max = max(area_single_max, area_single)

        '''
        image_single = cv2.medianBlur(image_single, 5)

        clahe = cv2.createCLAHE(clipLimit=20.0, tileGridSize=(8, 8))
        image_single = clahe.apply(image_single)
        snake = active_contour(image_single,
                               init, alpha=0.4, beta=20, gamma=0.01)
        c = np.expand_dims(snake.astype('float32'), 1)
        c = cv2.UMat(c)
        area = cv2.contourArea(c)
        
        '''

        area_single_max = area_single_max*arg_dict["edge_size(um)"]*arg_dict["edge_size(um)"]/arg_dict["image_size"]/arg_dict["image_size"]
        area_list.append(area_single_max)


    with open(arg_dict["output_path"]+"arteriole_area.csv", 'w', newline='', encoding='utf-8') as csvfile:
        writer = csv.writer(csvfile)
        writer.writerow(["predict_area(um^2)"])
        for i in range(len(area_list)):
            writer.writerow([area_list[i]])






def input_image(path_name, image_shape, index_begin, index_end, denoise_area):
    image = np.zeros((image_shape, image_shape))
    for i in range(index_begin, index_end + 1):
        image_num = str(i).zfill(3)
        image_file = path_name + image_num + '.tif'
        image_single = np.asarray(Image.open(image_file))
        average_noise = np.average(image_single[denoise_area[2]:denoise_area[3], denoise_area[1]:denoise_area[2]])
        image_single = image_single - average_noise
        image_single[image_single < 0] = 0


        midstep = np.stack((image,image_single),axis=2)
        image = np.max(midstep, axis=2)

    image = image/np.max(image)*255.
    image = image.astype('uint8')
    image = image_area = cv2.medianBlur(image, 5)
    ### image enhencement

    #image = cv2.equalizeHist(image)
    #background = restoration.rolling_ball(image)
    #image = image - background



    #for i in range(512):
    #    for j in range(512):
    #        image[i,j] = np.max(image[i,j],0)

    #image = cv2.medianBlur(image, 3)
    return image

def input_image_channelB(path_name, image_shape, index_begin, index_end ):
    image = np.zeros((image_shape, image_shape))
    for i in range(index_begin, index_end + 1):
        image_num = str(i).zfill(3)
        image_file = path_name + image_num + '.tif'
        image_single = np.asarray(Image.open(image_file)).astype('float')
        image += image_single
    image = image/65535*255.

    image = image.astype('uint8')
    image = cv2.medianBlur(image,5)
    return image

##### transfer image into binary data based on regional thrshold
def binary_threshold(image, regional_size, threshold_num, threshold_ratio):
    height = image.shape[0]
    number_flip = int((regional_size - 1)/2)
    image_mid = np.zeros((height+2*number_flip, height+2*number_flip))
    image_final = np.zeros((height,height))
    image = image.astype('float')
    image_mid[number_flip:(number_flip+height), number_flip:(number_flip+height)] = image
    image_mid[:number_flip,:] = image_mid[number_flip:number_flip*2,:]

    image_mid[(-number_flip):,:] = image_mid[(-2*number_flip):(-number_flip),:]
    image_mid[:,:number_flip] = image_mid[:,number_flip:number_flip*2]
    image_mid[:,(-number_flip):] = image_mid[:,(-2*number_flip):(-number_flip)]
    for i in range(height):
        for j in range(height):
            selected = image_mid[i:i+regional_size,j:j+regional_size]
            average = np.sum(selected) / (regional_size**2)
            #threshold = (256 - image[i,j])/(256 - average)
            threshold = average/image[i,j]
            if threshold < threshold_ratio and image[i,j] > threshold_num:
                image_final[i,j] = 255
            else:
                image_final[i,j] = 0
    return image_final

### count inside area of each counter
def contours_in(contours, image_size):
	p = np.zeros(shape=(image_size, image_size))
	cv2.drawContours(p, contours, -1, 255, -1)
	a = np.where(p==255)[0].reshape(-1,1)
	b = np.where(p==255)[1].reshape(-1,1)
	coordinate = np.concatenate([a,b], axis=1).tolist()
	inside = [tuple(x) for x in coordinate]
	return inside

def find_contours(contours, image_size, contour_min_size, center, center_set):
    final_contours = []
    for i in contours:
        sub_flag = True
        contour_one = [i]
        locations = contours_in(contour_one, image_size)
        if not center:
            for j in locations:
                if j in center_set:
                    sub_flag = False

                #if j[0] == 511 or j[1] == 511 or j[0] == 0 or j[1] == 0:
                #    sub_flag= False


        if len(locations) > contour_min_size and sub_flag:
            final_contours.append(i)

    return final_contours