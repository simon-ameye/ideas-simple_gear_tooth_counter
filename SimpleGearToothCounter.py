#Simple Gear Tooth Counter
#This is a simple tool for gear tooth (not only) counting
#Please share your questions to Simon Ameye - AVL AST FRANCE

import numpy as np
import os
import cv2
import matplotlib.widgets as wgt
import matplotlib.patches as patches
import matplotlib.pyplot as plt
import matplotlib.colors as colors
from imageio import imread
from scipy import ndimage
from scipy.signal import find_peaks
from tkinter.filedialog import askopenfilename
from tkinter import Tk
from pyautogui import screenshot
from skimage import img_as_ubyte
import matplotlib.image as mpimg

def Gear_function (im,Sdist,Soff):
    color_values = im[clicks[1, 1], clicks[1, 0]][0:3]
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
    n, m = bnim_centered.shape
    x, y = np.indices((n, m))
    distances = np.sqrt((x - 0.5*n)**2 + (y - 0.5*m)**2)/r
    bnim_centered[(bnim_centered == 1) * (distances < (Soff))] = -1
    #go to polar image
    polar_image = cv2.linearPolar(1*(bnim_centered==1),(bnim_centered.shape[0]/2, bnim_centered.shape[1]/2), r, cv2.WARP_FILL_OUTLIERS)
    #fft
    bary = np.sum(polar_image,1)
    x_bary = np.linspace(0,np.pi*2,len(bary))
    values = np.arange(len(bary))
    X = np.fft.fft(bary)/len(bary)
    x_small = values[2:np.int(max(values)/2)]
    X_small = X[2:np.int(max(values)/2)]
    N = max(values)
    #find peaks
    peaks, _ = find_peaks(np.abs(X_small))
    if len(peaks) == 0:peaks = np.array([0])
    sorted_peaks_args = np.flip(peaks[np.argsort(np.abs(X_small[peaks]))])
    sorted_peaks_args = np.append(sorted_peaks_args,np.zeros(1000)).astype("int")
    return center_point,r,bnim_centered,x_small,X_small,x_bary,bary,sorted_peaks_args,N,color_values

def update(val):
    Sdist = Dist_slider.val
    Sdist = np.power((Sdist)/(Sdist-1),2)*3
    Sval = Val_slider.val
    Soff = Off_slider.val
    center_point,r,bnim_centered,x_small,X_small,x_bary,bary,sorted_peaks_args,N,color_values = Gear_function(im,Sdist,Soff)
    nb_of_teeth = x_small[sorted_peaks_args[np.int(Sval)]]
    real_graph.clear()
    real_graph.imshow(im)
    circle = patches.Circle((center_point[1],center_point[0]),r,fill=False,color = "tab:blue")
    real_graph.add_patch(circle)
    real_graph.axis('off')
    real_graph.set_title('Original image', fontsize = 10, color = "black")
    bn_graph.clear()
    pic = bn_graph.imshow(bnim_centered, cmap=colors.ListedColormap([color_values/255, 'white', 'orange']))
    pic.set_clim(-1,1)
    bn_graph.axis('off')
    bn_graph.set_title('Filtered image', fontsize = 10, color = "black")
    tempo_graph.clear()
    tempo_graph.plot(x_bary,bary, color = 'tab:blue')
    if (min(bary)!=max(bary)) : tempo_graph.set_ylim(min(bary),max(bary))
    tempo_graph.set_frame_on(False)
    tempo_graph.axes.get_yaxis().set_visible(False)
    tempo_graph.axes.get_xaxis().set_visible(False)
    tempo_graph.set_title('Gear profile VS angle', fontsize = 10, color = "black")
    freq_graph.clear()
    freq_graph.bar(x_small, np.abs(X_small), width=2, color = 'tab:blue')
    freq_graph.bar(x_small[sorted_peaks_args[np.int(Sval)]], np.abs(X_small[sorted_peaks_args[np.int(Sval)]]), width=2, color = "orange")
    freq_graph.set_yticklabels([])
    freq_graph.set_frame_on(False)
    freq_graph.axes.get_yaxis().set_visible(False)
    freq_graph.axes.get_xaxis().set_visible(False)
    freq_graph.set_title('Confidence level VS nb of teeth', fontsize = 10, color = "black")
    if (max(np.abs(X_small))>0) : freq_graph.set_ylim(0, max(np.abs(X_small)))
    init_angle = -(np.angle(X_small[sorted_peaks_args[np.int(Sval)]])/nb_of_teeth)
    angles = np.array(init_angle+(np.linspace(0,2*np.pi,nb_of_teeth+1)))
    lines_vec_x = np.array(r*np.cos(angles))
    lines_vec_y = np.array(r*np.sin(angles))
    lines_vec_zeros = lines_vec_x*0
    lines_vec_x_zeros = np.reshape(np.array([lines_vec_x,lines_vec_zeros]).T,(2*len(lines_vec_x),1))
    lines_vec_y_zeros = np.reshape(np.array([lines_vec_y,lines_vec_zeros]).T,(2*len(lines_vec_y),1))
    real_graph.plot(lines_vec_x_zeros+center_point[1],lines_vec_y_zeros+center_point[0],color = "orange",linewidth=0.5)
    angles = np.linspace(0, 2*np.pi, nb_of_teeth * 10)
    init_angle = np.angle(X_small[sorted_peaks_args[np.int(Sval)]])
    tempo_graph.plot(angles, np.cos(init_angle + angles * nb_of_teeth) * np.abs(X_small[sorted_peaks_args[np.int(Sval)]]) + np.mean(bary),  color = "orange")
    for txt in fig.texts:
        txt.remove()
    real_graph.plot(clicks[1,0], clicks[1,1], marker = "1", color = "red", markersize = 10)
    freq_graph.annotate(str(nb_of_teeth), (nb_of_teeth,0), textcoords="offset points", xytext=(0,-5), ha='center', va='top', color='orange', fontsize=16)
    fig.canvas.draw_idle()

def takeScreenshot(self): 
    global im,im_for_color
    print("1) A Screenshot has been performed")
    myScreenshot = screenshot()
    im = img_as_ubyte(myScreenshot)
    update("val")
    
def browseim(self):
    global im,im_for_color
    root = Tk()
    root.withdraw()
    print("1) Browse your image")
    curr_directory = os.getcwd()
    filename = askopenfilename(initialdir = curr_directory + "/examples", title = "Select picture")
    im = imread(filename)[:,:,:4]
    update("val")
    
def onclick(event): 
    global clicks
    try : 
        print("button=%d, x=%d, y=%d, xdata=%f, ydata=%f" % ( 
         event.button, event.x, event.y, event.xdata, event.ydata))
        clicks[event.button - 1, :] = [event.xdata, event.ydata]
        print(clicks)
        update("val")
    except : 
        print("Click on pic")

plt.close('all')
#initialize
plt.rcParams['toolbar'] = 'None'
fig = plt.figure("A Simple Gear Tooth Counter") 
Sdist = 0.37
clicks = np.array([[0, 0], #left click = crop top left
          [186, 237], #midle click = color clic
          [0, 0]]) #right click = crop bottom right
gs = fig.add_gridspec(15, 2)
real_graph = fig.add_subplot(gs[2:10,0])
bn_graph = fig.add_subplot(gs[2:10,1])
tempo_graph = fig.add_subplot(gs[11:17,0])
freq_graph = fig.add_subplot(gs[11:14,1])
axes = plt.axes([0.2, 0.95, 0.65, 0.03])
Dist_slider = wgt.Slider(axes, 'Image Filter', 0, 1-0.01, valinit=0.4, valstep=0.01)
axes = plt.axes([0.2, 0.9, 0.65, 0.03])
Off_slider = wgt.Slider(axes, 'Offset', 0, 1, valinit=0.85, valstep=0.01)
axes = plt.axes([0.2, 0.85, 0.65, 0.03])
Val_slider = wgt.Slider(axes, 'Harmonic', 0, 10, valinit=0, valstep=1)
try:
    im = imread("StartLogo.png")[:,:,:4]
except:
    im = np.ones([300,300,3])
im_for_color = im[186:201,237:257,0:4]
update("val")
Dist_slider.on_changed(update)
Val_slider.on_changed(update)
Off_slider.on_changed(update)
axnext = plt.axes([0.6, 0.025, 0.2, 0.035])
bscreenshot = wgt.Button(axnext, 'Take Screenshot')
bscreenshot.on_clicked(takeScreenshot)
axnext = plt.axes([0.2, 0.025, 0.2, 0.035])
bbrowse = wgt.Button(axnext, 'Browse')
bbrowse.on_clicked(browseim)

clic = real_graph.figure.canvas.mpl_connect('button_press_event', onclick)




plt.show()
