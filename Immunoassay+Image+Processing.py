# get_ipython().magic('pylab')
import sys
if len(sys.argv) == 1 and sys.argv[0][-31:] == "Immunoassay+Image+Processing.py" and len(sys.argv[0]) != 31:
    sys.exit("Please type in python in front of script name ")
import sklearn
import warnings
from sklearn import datasets, linear_model
from sklearn.cross_validation import train_test_split
warnings.simplefilter(action = "ignore", category = FutureWarning)
import pandas as pd
from numpy import arange,array,ones
from scipy import stats
from sklearn.linear_model import LinearRegression
import matplotlib
from matplotlib import style
import matplotlib.pyplot as plt
import numpy as np
import skimage as sk
import skimage.transform as skt
import skimage.feature as skf
import skimage.color as skc
import skimage.io as skio
import skimage.filters as skif
import time
import multiprocessing as mp
import scipy.sparse as scipys
import scipy as scipy
import scipy.spatial as scipysp
import skimage.draw as skid
import os
from os import listdir
from skimage import segmentation
from os.path import isfile, join
from skimage import morphology
from skimage.morphology import watershed
from skimage.feature import peak_local_max
from skimage import measure
from skimage import restoration
from skimage import img_as_float, morphology
from skimage import exposure, filters
from skimage import data, color, img_as_float
from skimage.draw import circle_perimeter
from skimage.util import img_as_ubyte
from scipy import ndimage as ndi
import matplotlib.patches as mpatches
from skimage.filters import threshold_otsu
from skimage.segmentation import clear_border
from skimage.measure import label, regionprops
from skimage.morphology import closing, square
from skimage.color import label2rgb
# get_ipython().magic('matplotlib inline')

""" For Testing purposes (for other processing techniques) """
# colors = ['#1f77b4', '#ff7f0e', '#2ca02c', '#d62728',
#           '#9467bd', '#8c564b', '#e377c2', '#7f7f7f',
#           '#bcbd22', '#17becf']
# sns.set(style='whitegrid', palette=colors, rc={'axes.labelsize': 16})

class StandardCurve:
    """ A standard Curve object after all the images in a directory are evaluated """

    def __init__(self, name, xdata, ydata):
        self.name = name
        self.xdata = xdata
        self.ydata = ydata
        self.slope = None
        self.y_intercept = None
        self.line = None
        self.r_squared = None

    def compute_standard_curve(self, xdata, ydata):
        #Testing for scatter plot without regression line
        #plt.scatter(xdata, ydata)
        #plt.xlabel("Concentration (ng/ml)")
        #plt.ylabel("Intensity")
        #plt.show()

        """ Get slope_intercept form of a line based on two lists of data """
        def slope_intercept(xv, yv):
            y = np.array(yv)
            x = np.array(xv)
            m1 = (((np.mean(y) * np.mean(x)) - np.mean(y*x)) /
                     ((np.mean(x) * np.mean(x)) - np.mean(x*x)))
            m1 = round(m1, 2)
            b1 = (np.mean(y) - m1*np.mean(x))
            b1 = round(b1, 2)
            return m1, b1

        m, b = slope_intercept(xdata, ydata)
        regression_line = [m*x + b for x in xdata]
        plt.scatter(xdata, ydata, color = "red")
        plt.plot(xdata, regression_line)
        plt.xlabel("Concentration (ng/ml)")
        plt.ylabel("Intensity")
        plt.title("Standard Curve")
        plt.show()
        def squared_error(ys_data ,ys_line):
            squared_err = list()
            for i in range(0, len(ys_data)):
                squared_err.append((ys_data[i] - ys_line[i]) * (ys_data[i] - ys_line[i]))
            result = sum(squared_err)
            return result

        def coefficient_of_determination(ys_data,ys_line):
            y_mean_line = [np.mean(ys_data) for y in ys_data]
            squared_error_regr = squared_error(ys_data, ys_line)
            squared_error_y_mean = squared_error(ys_data, y_mean_line)
            return 1 - (squared_error_regr/squared_error_y_mean)
        self.r_squared = coefficient_of_determination(ydata, regression_line)
        self.line = regression_line
        self.slope = m
        self.y_intercept = b
        self.line = regression_line
        return

    """ Use slope and y-intercept from the regression line equation to predict y_value of input_x """
    def computePredictedIntensity(self, input_x):
            intensity = self.slope*input_x + self.y_intercept
            return intensity
    def get_xdata(self):
        return self.xdata

    def get_ydata(self):
        return self.ydata

    def get_slope(self):
        return self.slope

    def get_y_intercept(self):
        return self.y_intercept

    def get_xdata(self):
        return self.xdata

    def get_ydata(self):
        return self.ydata

    def get_r_squared(self):
        return self.r_squared

    def get_line(self):
        return self.line

    """ Cacculates the standard deviation of residual errors for y (How wrong is the prediction) """
    def rootMeanSquareErrorY(self, y):
        y_actual = np.array(y)
        y_predicted = np.array(self.ydata)
        error = (y_actual - y_predicted)**2
        mean_error = round(np.mean(error))
        error_sqrt = mean_error**(1/2)
        return error_sqrt

    """ Compute Predicted Concentration """
    def computePredictedConcentration(self, input_y):
        pred_conc = (input_y - self.y_intercept) / self.slope
        return pred_conc

    """ rmes for x (concentration) """
    def rootMeanSquareErrorX(self, x):
        x_actual = np.array(x)
        x_predicted = np.array(self.xdata)
        error = (x_actual - x_predicted)**2
        mean_error = round(np.mean(error))
        error_sqrt = mean_error**(1/2)
        return error_sqrt

""" Reads in a Directory and returns a list of image files """
def readDir(directory):
    files = list()
    for f in listdir(directory):
        files.append(join(directory, f))
    return files


""" Displays the image """
def showImage(image):
    plt.imshow(image)
    plt.show()
def showImageFile(file):
    pic = skio.imread(file, as_grey = True)
    showImage(pic)


""" Tries to separate background by selecting seeds and using watershed algorithm. Learned from skimage tutorial Online """
def seperateBackground(image):
    sobel = filters.sobel(image)
    plt.rcParams['image.interpolation'] = 'nearest'
    plt.rcParams['image.cmap'] = 'gray'
    plt.rcParams['figure.dpi'] = 200
    # plt.imshow(sobel) #Uncomment to show and use plt.show to show image. But it doesn't work if there are other imshows that occur after, so make sure this is the only one.
    #                     #Same with everything else with imshow
    # plt.show()
    blurred = filters.gaussian(sobel, sigma=2.0)
    # plt.imshow(blurred)
    # plt.show()  #Uncomment this only to show blurred image
    light_spots = np.array((image > 300).nonzero()).T
    light_spots.shape
    bool_mask = np.zeros(image.shape, dtype=np.bool)
    bool_mask[tuple(light_spots.T)] = True
    seed_mask, num_seeds = ndi.label(bool_mask)
    ws = morphology.watershed(blurred, seed_mask)
    background = max(set(ws.ravel()), key=lambda g: np.sum(ws == g))
    background_mask = (ws == background)
    cleaned = image * ~background_mask
    return cleaned

""" Takes in a list of image files and returns a dictionary of image file name mapped to its average intensity value """
def aveImageIntensity(image_files):
    imdict = dict()
    for file in image_files:
        pic = skio.imread(file, as_grey = True)
        im = seperateBackground(pic)
        d = im > np.amin(im) + 400
        c = im < np.amin(im) + 100
        im[c] = 0
        im[d] = 0
        total = np.sum(im)
        size = len(np.nonzero(im)[0])
        average = total / size
        name = file[9:]
        imdict[name] = average
#     for i in range(0, pic1.shape[0]):
#         for j in range(0, pic1.shape[1]):
#             if pic1[i][j] == 0:
#                 pic1[i][j] = randint(0, 1) * 500
    return imdict

def predictIntensity(imageFileDirectory):
    files = readDir(imageFileDirectory)
    imdict = aveImageIntensity(files)
    return imdict
""" For digital Immunoassay. Also learned from skimage tutorial on labels and otsu threshold """
def digitalImmunoassayIntensity(image_files):
    files = readDir(image_files)
    imdict = dict()
    for file in files:
        image = skio.imread(file, as_grey = True)
        name = file[10:]

        # apply threshold
        thresh = threshold_otsu(image)
        bw = closing(image > thresh, square(3))

        # remove artifacts connected to image border
        cleared = clear_border(bw)

        # label image regions
        label_image = label(cleared)
        image_label_overlay = label2rgb(label_image, image=image)

        # fig, ax = plt.subplots(figsize=(10, 6)) #For plotting
        # ax.imshow(image_label_overlay)
        num_regions = 0
        for region in regionprops(label_image):
            # take regions with large enough areas
            if region.area >= 100 and region.area <= 400:
                # Below is to get the ave intensity of each region
                # im = region.coords
                # total = np.sum(im)
                # size = len(np.nonzero(im)[0])
                # average = total / size
                # print(average)

                # Below is to draw rectangles around segmented regions
                # minr, minc, maxr, maxc = region.bbox
                # rect = mpatches.Rectangle((minc, minr), maxc - minc, maxr - minr,
                #                           fill=False, edgecolor='red', linewidth=2)
                # ax.add_patch(rect)
                num_regions = num_regions + 1
        imdict[name] = num_regions
        # ax.set_axis_off() #For Plotting
        # plt.tight_layout()
        # plt.show()
        # plt.imshow(image)
        # plt.show()
    return imdict

""" Computes in nanogram/ml concentration> Valid units are ng, pg, mg, and ug """
def curveFitting(imdict):
    xdata = list()
    ydata = list()
    for key in imdict:
        unit = key[-6:-4]
        if unit == "ng":
            xdata.append(int(key[:-6]))
        elif unit == "pg":
            xdata.append(int(key[:-6]) * 0.001)
        elif unit == "ug":
            xdata.append(int(key[:-6]) * 1000)
        elif unit == "mg":
            xdata.append(int(key[:-6]) * 1000000)
        else:
            sys.exit("Invalid unit")
        ydata.append(imdict[key])
    xdata.sort()
    ydata.sort()
    return StandardCurve("Test", xdata, ydata)
""" Get xdata for new standard curve """
def get_xdata(imdict):
    xdata = list()
    for key in imdict:
        unit = key[-6:-4]
        if unit == "ng":
            xdata.append(int(key[:-6]))
        elif unit == "pg":
            xdata.append(int(key[:-6]) * 0.001)
        elif unit == "ug":
            xdata.append(int(key[:-6]) * 1000)
        elif unit == "mg":
            xdata.append(int(key[:-6]) * 1000000)
        else:
            sys.exit("Invalid unit")
        xdata.sort()
        return xdata
""" Get ydata for new standard curve """
def get_ydata(imdict):
    ydata = list()
    for key in imdict:
        ydata.append(imdict[key])
    ydata.sort()
    return ydata


""" Taking input from command line and running the interactive script """
if len(sys.argv) == 1:
    sys.exit("Please specify image directory")
elif len(sys.argv) > 2:
    sys.exit("Too many arguments")
else:
    filename = "./" + sys.argv[1]
    if sys.argv[1] == "images":
        imagedict = predictIntensity(filename)
    elif sys.argv[1] == "digital":
        imagedict = digitalImmunoassayIntensity(filename)
    else:
        sys.exit("Invalid file name")
    curve = curveFitting(imagedict)
    curve.compute_standard_curve(curve.xdata, curve.ydata)
    exit = False
    while not exit:
        print()
        print("Write one of the following valid commands:")
        print(" type in 'Concentration YourConcentration' in ng/ml to get predicted intensity. Ex: if you want to get 20ng/ml, type in 'Concentration 20' ")
        print(" type in 'Intensity YourIntensity' to get predicted concentration.")
        print(" type 'r_squared' to get r_squared value of the regression line")
        print(" type 'slope' to get slope")
        print(" type 'y-intercept' to get y-intercept")
        print(" type 'xdata' to get list of concentrations in ng/ml")
        print(" type 'ydata' to get list of intensities")
        print( " type 'line' to get regression line values")
        print(" type 'rmseY ActualIntensityValue' to get root mean sqaure error of that specified intensity value")
        print(" type 'rmseX ActualConcentrationValue' to get root mean square error of that specified concentration value")
        print(" type 'exit()' to exit this script")
        print("Please follow EXACTLY one of the following commands, else the program will print an error. Case matters")
        userInput = input()
        try:
            words = userInput.split()
            first = words[0]
        except ValueError:
            print("Invalid argument")
        if first == "Concentration":
            try:
                concentration = float(words[1])
                pred_intensity = curve.computePredictedIntensity(concentration)
                print(pred_intensity)
            except:
                print("Invalid second argument")
        elif first == "Intensity":
            try:
                intensity = float(words[1])
                pred_concentration = curve.computePredictedConcentration(intensity)
                print(pred_concentration)
            except:
                print("Invalid second argument")
        elif first == "r_squared":
            print(curve.get_r_squared())
        elif first == "slope":
            print(curve.get_slope())
        elif first == "y-intercept":
            print(curve.get_y_intercept())
        elif first == "line":
            print(curve.get_line())
        elif first == "xdata":
            print(curve.get_xdata())
        elif first == "ydata":
            print(curve.get_ydata())
        elif first == "rmseY":
            try:
                y_value = float(words[1])
                actual = curve.rootMeanSquareErrorY(y_value)
                print(actual)
            except:
                print("Invalid second argument")
        elif first == "rmseX":
            try:
                x_value = float(words[1])
                actual_x = curve.rootMeanSquareErrorX(x_value)
                print(actual_x)
            except:
                print("Invalid second argument")
        elif first == "exit()":
            exit = True
        else:
            print("Invalid first argument")
