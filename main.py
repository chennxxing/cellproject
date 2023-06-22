import numpy as np
import cv2
import matplotlib.pyplot as plt
import os
import matplotlib
import scipy.signal
import tkinter as tk
from skimage import restoration
from tool import view_result, save_analysis_result
from PIL import Image, ImageTk
import tkinter.ttk as ttk


arg_para = dict()
arg_para2 = dict()

#### 20 0.97 10 0.9
arg_para["file_path_1"] = './data_051123/PA60um3.2fps5sws_001/ChanB_001_001_001_'
arg_para["file_path_2"] = './data_051123/PA60um3.2fps5sws_001/ChanC_001_001_001_'
arg_para["begin_index"] = 1
arg_para["end_index"] = 200
arg_para["image_size"] = 512
arg_para["edge_size(um)"] = 60
arg_para["output_path"] = './data_051123/result_PA1/'

def open_vessel():
    global vessel_img, slider_label, my_slider, my_label
    top1 = tk.Toplevel()
    top1.title("Vessel Image")
    number = 1
    filename = arg_para["file_path_2"]+str(number).zfill(3)+'.tif'
    vessel_img = np.asarray(Image.open(filename)).astype('float')
    vessel_img = vessel_img/np.max(vessel_img)*255
    vessel_img = vessel_img.astype('uint8')
    vessel_img = ImageTk.PhotoImage(image=Image.fromarray(vessel_img))
    my_label = tk.Label(top1, image=vessel_img)
    my_label.pack()
    my_slider = ttk.Scale(top1, from_=1, to=arg_para["end_index"], orient=tk.HORIZONTAL, value=1, command=slide_action, length=360)
    my_slider.pack(pady=20)
    slider_label = tk.Label(top1, text="1 of "+str(arg_para["end_index"]))
    slider_label.pack(pady=5)


def slide_action(x):
    number = int(int(my_slider.get()))
    filename = arg_para["file_path_2"] + str(number).zfill(3) + '.tif'
    vessel_img2 = np.asarray(Image.open(filename)).astype('float')
    vessel_img2 = vessel_img2 / np.max(vessel_img2) * 255
    vessel_img2 = vessel_img2.astype('uint8')

    vessel_img2 = ImageTk.PhotoImage(image=Image.fromarray(vessel_img2))
    my_label.config(image = vessel_img2)
    my_label.image = vessel_img2
    slider_label.config(text=str(number)+ " of "+str(arg_para["end_index"]))



def open_signal():
    global signal_img, slider_label2, my_slider2, my_label2
    top2 = tk.Toplevel()
    top2.title("Signal Image")
    number = 1
    filename = arg_para["file_path_1"]+str(number).zfill(3)+'.tif'
    signal_img = np.asarray(Image.open(filename)).astype('float')
    signal_img = cv2.normalize(signal_img, None, 0, 255, cv2.NORM_MINMAX)
    #signal_img = signal_img/np.max(signal_img)*255
    signal_img = signal_img.astype('uint8')

    #signal_img = cv2.equalizeHist(signal_img)
    #background = restoration.rolling_ball(signal_img)
    #signal_img = signal_img - background

    signal_img = ImageTk.PhotoImage(image=Image.fromarray(signal_img))
    my_label2 = tk.Label(top2, image=signal_img)
    my_label2.pack()
    my_slider2 = ttk.Scale(top2, from_=1, to=arg_para["end_index"], orient=tk.HORIZONTAL, value=1, command=slide_action2, length=360)
    my_slider2.pack(pady=20)
    slider_label2 = tk.Label(top2, text="1 of "+str(arg_para["end_index"]))
    slider_label2.pack(pady=5)


def slide_action2(x):
    number = int(int(my_slider2.get()))
    filename = arg_para["file_path_1"] + str(number).zfill(3) + '.tif'
    signal_img2 = np.asarray(Image.open(filename)).astype('float')
    signal_img2 = cv2.normalize(signal_img2, None, 0, 255, cv2.NORM_MINMAX)
    #signal_img2 = signal_img2 / np.max(signal_img2) * 255
    signal_img2 = signal_img2.astype('uint8')

    #signal_img2 = cv2.equalizeHist(signal_img2)
    #background = restoration.rolling_ball(signal_img2)
    #signal_img2 = signal_img2 - background

    signal_img2 = ImageTk.PhotoImage(image=Image.fromarray(signal_img2))
    my_label2.config(image = signal_img2)
    my_label2.image = signal_img2
    slider_label2.config(text=str(number)+ " of "+str(arg_para["end_index"]))



def graph():
    try:
        global img, img2, get_contour, image_init, arg_para, vessel_len, denoise_center
        arg_para["vessel1"] = np.double(vessel_para1_box.get())
        arg_para["vessel2"] = np.double(vessel_para2_box.get())
        arg_para["signal1"] = np.double(cell_para1_box.get())
        arg_para["signal2"] = np.double(cell_para2_box.get())
        image_init, get_contour, vessel_len, denoise_center = view_result(arg_para)

        #plt.figure(figsize=(8, 8))
        #plt.imshow(image, cmap='gray')
        #plt.show()
        img = ImageTk.PhotoImage(image=Image.fromarray(image_init))

        #canvas2.create_image(0,0, anchor=tk.CENTER,image=img)

        my_label = tk.Label(canvas2, image=img)
        my_label.place(x=0, y=0, relwidth=1, relheight=1)

    except ValueError:
        answer.config(text="Need to input float numbers")

def copy_image():
    try:
        global img2, image_result, contour_result, arg_para2, vessel_len2, denoise_center2
        img2 = img
        image_result = image_init
        contour_result = get_contour
        my_label = tk.Label(canvas1, image=img)
        my_label.place(x=0, y=0, relwidth=1, relheight=1)
        arg_para2 = arg_para
        vessel_len2 = vessel_len
        denoise_center2 = denoise_center

        #canvas1.create_image(0, 0, anchor=tk.CENTER, image=img2)
    except ValueError:
        answer.config(text="Need to preview a new image")

def save_function():
    try:
        save_analysis_result(image_result, contour_result, arg_para2, vessel_len2, denoise_center2)

    except ValueError:
        answer.config(text="Need to have image in the left")


root = tk.Tk()
root.title("Figure Analysis")
root.geometry("1300x900")

for i in range(2):
    root.grid_columnconfigure(i,weight=1)
for i in range(6):
    root.grid_rowconfigure(i,weight=1)

frame1 = tk.Frame(root)
canvas1 = tk.Canvas(root)
canvas2 = tk.Canvas(root)
frame1.grid(row = 0, column = 0, columnspan = 2, sticky = tk.NSEW)
canvas1.grid(row = 1, column=0, rowspan = 5, sticky=tk.NSEW)
canvas2.grid(row = 1, column=1, rowspan = 5, sticky=tk.NSEW)

#### frame 1
vessel_label = tk.Label(frame1, text="vessel channel")
vessel_label.place(relx=0, rely=0)
vessel_button = tk.Button(frame1, text="preview", command=open_vessel)
vessel_button.place(relx=0.08, rely=0)

signal_label = tk.Label(frame1, text="singal channel")
signal_label.place(relx= 0, rely=0.3)
signal_button = tk.Button(frame1, text="preview", command=open_signal)
signal_button.place(relx = 0.08, rely=0.3)

vessel_label1 = tk.Label(frame1, text = "vessel parameter 1")
vessel_label1.place(relx=0.2, rely = 0, relwidth=0.1, relheight=0.2)
vessel_para1_box = tk.Entry(frame1, textvariable=tk.StringVar(value='30'))
vessel_para1_box.place(relx=0.3, rely = 0, relwidth=0.1, relheight=0.2)

vessel_label2 = tk.Label(frame1, text = "vessel parameter 2")
vessel_label2.place(relx=0.2, rely = 0.3, relwidth=0.1, relheight=0.2)
vessel_para2_box = tk.Entry(frame1, textvariable=tk.StringVar(value='1'))
vessel_para2_box.place(relx=0.3, rely = 0.3, relwidth=0.1, relheight=0.2)

cell_label1 = tk.Label(frame1, text = "signal parameter 1")
cell_label1.place(relx=0.4, rely = 0, relwidth=0.1, relheight=0.2)
cell_para1_box = tk.Entry(frame1, textvariable=tk.StringVar(value='50'))
cell_para1_box.place(relx=0.5, rely = 0, relwidth=0.1, relheight=0.2)

cell_label2 = tk.Label(frame1, text = "signal parameter 2")
cell_label2.place(relx=0.4, rely = 0.3, relwidth=0.1, relheight=0.2)
cell_para2_box = tk.Entry(frame1, textvariable=tk.StringVar(value='0.95'))
cell_para2_box.place(relx=0.5, rely = 0.3, relwidth=0.1, relheight=0.2)


explain1 = tk.Label(frame1, text = "parater 1 should choose between 1 to 255, smaller number cause more signal appear")
explain1.place(relx = 0, rely= 0.55)
explain2 = tk.Label(frame1, text= "parater 2 should choose between 0 to 1.5, larger number cause more signal appear")
explain2.place(relx = 0, rely= 0.75)

answer = tk.Label(frame1, text="")
answer.place(relx= 0.6, rely=0.7)

preview_button = tk.Button(frame1, text="preview", command=graph)
preview_button.place(relx=0.7, rely = 0)
prefer_button = tk.Button(frame1, text="save to left",command= copy_image)
prefer_button.place(relx=0.7, rely = 0.3)
save_button = tk.Button(frame1, text="save images", command=save_function)
save_button.place(relx=0.8, rely = 0.2)


root.mainloop()




