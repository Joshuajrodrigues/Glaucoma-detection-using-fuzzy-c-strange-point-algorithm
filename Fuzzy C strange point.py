

# ------------------ FUZZY C STRANGE POINTS CLUSTERING --------------------------


import array as arr
import numpy as np
from PIL import Image
from matplotlib import pyplot as plt
from time import time
from cal_disk import cal_area
from cdr import cal_cdr
import cv2

# nega 227
start_time = time()

# ROI extractor ---------------------------------------------
img_raw = cv2.imread("160.jpg")

cv2.namedWindow("roiwin", cv2.WINDOW_NORMAL)
r = cv2.selectROI('roiwin', img_raw)  # roi

imCrop = img_raw[int(r[1]):int(r[1]+r[3]), int(r[0]):int(r[0]+r[2])]
imCrop = cv2.cvtColor(imCrop, cv2.COLOR_BGR2RGB)
#cv2.imshow("Image", imCrop)
plt.imsave('roiImage2.jpg', imCrop, format='jpg')

img = Image.open("roiImage2.jpg")

width, height = img.size
bimg = np.zeros((height, width, 3), dtype="uint8")  # 0-255

plt.imsave('blankImg1.jpg', bimg, format='jpg')

img1 = Image.open("blankImg1.jpg")
img2 = Image.open("blankImg1.jpg")
img3 = Image.open("blankImg1.jpg")

eye = img.load()
for i in range(width):
    for j in range(height):
        r, g, b = eye[i, j]
        eye[i, j] = (r-r, g-0, b-b)
img.save('greeneye.jpg')


# number of clusters = 2 = disk and cup

data = np.array(img)
cmin = data[..., 1].min()
cmax = data[..., 1].max()

# print(cmin,cmax) 56 251

constDist = np.linalg.norm(cmax - cmin)

tempArr = arr.array('d', [])
tempArrY = arr.array('d', [])


for i in range(width):
    for j in range(height):
        x, y, z = eye[i, j]  # (0,78,0)
        tempArrY.append(y)

# furthest poinr from max and min in a 1 dimention is the median value
cs = np.median(tempArrY)

if np.linalg.norm(cmin-cs) == np.linalg.norm(cmax-cs):
    cstr = cs
    # print(cstr)
elif np.linalg.norm(cmin-cs) < np.linalg.norm(cmax-cs):
    cstr = (cs+(abs(cmax-cs)/2))
    # print(cstr)
elif np.linalg.norm(cmin-cs) > np.linalg.norm(cmax-cs):
    cstr = (cmin+(abs(cs-cmin)/2))

print(cs, cstr, cmin, cmax)

for i in range(width):
    for j in range(height):
        x = eye[i, j]  # (0,78,0)
        coords = i, j

        cminx = (x[1]-cmin)**2
        cmaxx = (x[1]-cmax)**2
        cstrx = (x[1]-cstr)**2

        uc1 = 1/((cminx/cminx)+(cminx/cmaxx)+(cminx/cstrx))
        uc2 = 1/((cmaxx/cmaxx)+(cmaxx/cminx)+(cmaxx/cstrx))
        uc3 = 1/((cstrx/cstrx)+(cstrx/cminx)+(cstrx/cmaxx))

        if np.isnan(uc1) or np.isnan(uc2) or np.isnan(uc3):
            continue
        elif (uc1 > uc3) and (uc1 > uc2):
            continue
        elif (uc2 > uc1) and (uc2 > uc3):
            img2.putpixel(coords, x)  # This is the cup cluster
        else:
            img3.putpixel(coords, x)  # This is the disc cluster

img2.save("Fuzzy_C_strange_point_cup.jpg")
img3.save("Fuzzy_C_strange_point_disk.jpg")

# Thresholding to convert the image to Black and white image
img = Image.open('Fuzzy_C_strange_point_cup.jpg')
width, height = img.size

pixels = img.load()

intensityA = cstr
intensityB = cmax
# anything inside this range goes to 255 everything outside is 0
for i in range(width):
    for j in range(height):
        r, g, b = pixels[i, j]
        if((intensityA <= r and r <= intensityB) or (intensityA <= g and g <= intensityB) or (intensityA <= b and b <= intensityB)):
            r = g = b = 255
        else:
            r = g = b = 0
        pixels[i, j] = (r, g, b)
# Save BW cup image
img.save('Fuzzy_C_strange_point_cup_IS.jpg')


img = Image.open('Fuzzy_C_strange_point_disk.jpg')
pixels = img.load()
intensityA = cmin
intensityB = cmax

for i in range(width):
    for j in range(height):
        r, g, b = pixels[i, j]
        if((intensityA <= r and r <= intensityB) or (intensityA <= g and g <= intensityB) or (intensityA <= b and b <= intensityB)):
            r = g = b = 255
        else:
            r = g = b = 0
        pixels[i, j] = (r, g, b)
# Save BW disc image
img.save('Fuzzy_C_strange_point_disk_IS.jpg')

# Function calls to calculate the cup to disc Ratio (CDR)
img_path = "./Fuzzy_C_strange_point_cup_IS.jpg"
areaCup = cal_area(img_path)
print(areaCup)
img_path = "./Fuzzy_C_strange_point_disk_IS.jpg"
areaDisc = cal_area(img_path)
print(areaDisc)

cup_disc_ratio = cal_cdr(areaCup, areaDisc)
print("CDR = ", cup_disc_ratio)

# Time for the entire process
proc_time = time()-start_time
print("time = ", proc_time)
