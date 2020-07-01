#Simple Gear Tooth Counter
#This is a simple tool for gear tooth (not only) counting
#Please share your questions to Simon Ameye - AVL AST FRANCE

import numpy as np
import os
import cv2
import matplotlib.widgets as wgt
import matplotlib.patches as patches
import matplotlib.pyplot as plt
from imageio import imread
from scipy import ndimage
from scipy.signal import find_peaks
from tkinter.filedialog import askopenfilename
from tkinter import Tk
from pyautogui import screenshot
from skimage import img_as_ubyte


def Gear_function (im,Sdist,im_for_color):
    color_values = np.array([np.mean(im_for_color[:,:,0]),np.mean(im_for_color[:,:,1]),np.mean(im_for_color[:,:,2])])
    color_values[color_values==0]=1e-5
    L = im.shape[1]
    H = im.shape[0]
    im = im[:,:,:3]
    #filter by color
    bnim = np.zeros((H,L))
    color_diff_vect = (im[:,:,:]/color_values)
    bnim[(np.amax(color_diff_vect,axis = 2)-np.amin(color_diff_vect,axis=2))<=Sdist] = 1
    #define work zone
    center_point = [np.int(H/2),np.int(L/2)]
    if np.sum(bnim)>0:
        center_point[0] = (ndimage.measurements.center_of_mass(bnim))[0].astype("int")
        center_point[1] = (ndimage.measurements.center_of_mass(bnim))[1].astype("int") #PROBLEM
    r = min(center_point[1], L - center_point[1], H - center_point[0], center_point[0])-1
    bnim_centered = bnim[(center_point[0]-r):(center_point[0]+r),(center_point[1]-r):(center_point[1]+r)]
    #go to polar image
    polar_image = cv2.linearPolar(bnim_centered,(bnim_centered.shape[0]/2, bnim_centered.shape[1]/2), r, cv2.WARP_FILL_OUTLIERS)
    #fft
    bary = np.sum(polar_image,1)
    x_bary = np.linspace(0,np.pi*2,len(bary))
    values = np.arange(len(bary))
    X = np.fft.fft(bary)
    x_small = values[2:np.int(max(values)/2)]
    X_small = X[2:np.int(max(values)/2)]
    N = max(values)
    #find peaks
    peaks, _ = find_peaks(np.abs(X_small))
    if len(peaks) == 0:peaks = np.array([0])
    sorted_peaks_args = np.flip(peaks[np.argsort(np.abs(X_small[peaks]))])
    sorted_peaks_args = np.append(sorted_peaks_args,np.zeros(1000)).astype("int")
    return center_point,r,bnim_centered,x_small,X_small,x_bary,bary,sorted_peaks_args,N

def update(val):
    Sdist = Dist_slider.val
    Sdist = np.power((Sdist)/(Sdist-1),2)*3
    Sval = Val_slider.val
    center_point,r,bnim_centered,x_small,X_small,x_bary,bary,sorted_peaks_args,N = Gear_function(im,Sdist,im_for_color)
    nb_of_teeth = x_small[sorted_peaks_args[np.int(Sval)]]
    ax[0,0].clear()
    ax[0,0].imshow(im)
    ax[0,0].plot(center_point[1],center_point[0], 'x')
#    rect = patches.Rectangle((center_point[1]-r,center_point[0]-r),2*r,2*r,linewidth=1,edgecolor='C1',facecolor='none')
    circle = patches.Circle((center_point[1],center_point[0]),r,fill=False,color = "C1")
#    ax[0,0].add_patch(rect)
    ax[0,0].add_patch(circle)
    ax[0,0].axis('off')
    ax[0,1].clear()
    ax[0,1].imshow(bnim_centered, cmap='binary')
    ax[0,1].axis('off')
    ax[1,0].clear()
    ax[1,0].plot(x_bary,bary)
    if (min(bary)!=max(bary)) : ax[1,0].set_ylim(min(bary),max(bary))
    ax[1,0].set_yticklabels([])
    ax[1,0].set_frame_on(False)
    ax[1,0].axes.get_yaxis().set_visible(False)
    ax[1,1].clear()
    ax[1,1].plot(x_small,np.abs(X_small))
    ax[1,1].plot([nb_of_teeth,nb_of_teeth],[0,np.max(np.abs(X_small))],color = "C1")
    ax[1,1].set_yticklabels([])
    ax[1,1].set_frame_on(False)
    ax[1,1].axes.get_yaxis().set_visible(False)
    if (max(np.abs(X_small))>0) : ax[1,1].set_ylim(0, max(np.abs(X_small)))
    init_angle = -(np.angle(X_small[sorted_peaks_args[np.int(Sval)]])/nb_of_teeth)
    angles = np.array(init_angle+(np.linspace(0,2*np.pi,nb_of_teeth+1)))
    lines_vec_x = np.array(r*np.cos(angles))
    lines_vec_y = np.array(r*np.sin(angles))
    lines_vec_zeros = lines_vec_x*0
    lines_vec_x_zeros = np.reshape(np.array([lines_vec_x,lines_vec_zeros]).T,(2*len(lines_vec_x),1))
    lines_vec_y_zeros = np.reshape(np.array([lines_vec_y,lines_vec_zeros]).T,(2*len(lines_vec_y),1))
    real_graph.plot(lines_vec_x_zeros+center_point[1],lines_vec_y_zeros+center_point[0],color = "C1")
    for txt in fig.texts:
        txt.remove()
    plt.figtext(0.5, 0.5, (str(nb_of_teeth)),color='C1', fontsize=20)
    fig.canvas.draw_idle()

def ROI(iminit):
    # Select ROI for image crop
    print("2) Crop the region of interest and press Enter")
    showCrosshair = False
    fromCenter = False
    r = cv2.selectROI("Region of interest",iminit,fromCenter,showCrosshair)
    cv2.destroyAllWindows()
    im = iminit[int(r[1]):int(r[1]+r[3]), int(r[0]):int(r[0]+r[2])]
    # Display cropped image
    # Select ROI for color picker
    print("3) Select a region for the gear color and press Enter")
    r = cv2.selectROI("Region for the gear color",im,fromCenter,showCrosshair)
    # Crop image
    im_for_color = im[int(r[1])+1:int(r[1]+r[3]-1), int(r[0])+1:int(r[0]+r[2])-1]
    im_for_color = im_for_color[:,:,:3]
    # Display cropped image
    cv2.destroyAllWindows()
    return im,im_for_color

def takeScreenshot(self): 
    global im,im_for_color
    print("1) A Screenshot has been performed")
    myScreenshot = screenshot()
    iminit = img_as_ubyte(myScreenshot)
    (im,im_for_color) = ROI(iminit)
    update("val")
    
def browseim(self):
    global im,im_for_color
    root = Tk()
    root.withdraw()
    print("1) Browse your image")
    curr_directory = os.getcwd()
    filename = askopenfilename(initialdir = curr_directory + "/examples", title = "Select picture")
    iminit = imread(filename)
    (im,im_for_color) = ROI(iminit)
    update("val")

plt.close('all')
#initialize
plt.rcParams['toolbar'] = 'None'
fig = plt.figure("A Simple Gear Tooth Counter") 
ax = fig.subplots(2,2)
Sdist = 0.3
real_graph = ax[0,0]
bn_graph = ax[0,1]
tempo_graph = ax[1,0]
freq_graph = ax[1,1]
axes = plt.axes([0.2, 0.95, 0.65, 0.03])
Dist_slider = wgt.Slider(axes, 'Image Filter', 0, 1-0.01, valinit=0.3, valstep=0.01)
axes = plt.axes([0.2, 0.9, 0.65, 0.03])
Val_slider = wgt.Slider(axes, 'Harmonic', 0, 10, valinit=0, valstep=1)
try:
    im = imread("StartLogo.png")
except:
    im = np.ones([100,100,4])
im_for_color = im[75:77,25:27,0:3]
update("val")
Dist_slider.on_changed(update)
Val_slider.on_changed(update)
axnext = plt.axes([0.6, 0.02, 0.2, 0.03])
bscreenshot = wgt.Button(axnext, 'Take Screenshot')
bscreenshot.on_clicked(takeScreenshot)
axnext = plt.axes([0.3, 0.02, 0.2, 0.03])
bbrowse = wgt.Button(axnext, 'Browse')
bbrowse.on_clicked(browseim)
plt.show()