import numpy as np
from skimage.transform import resize
from skimage import measure
from skimage.measure import regionprops
import matplotlib.patches as patches
import matplotlib.pyplot as plt
from skimage.io import imread

license_plate = imread("/Users/krishrana/Desktop/delhi-car-plate-realistic-looking-600w-770701510.jpg", as_gray=True)
height = license_plate.shape[0]
width = license_plate.shape[1]
print(height)
print(width)

labelled_plate = measure.label(license_plate) 

fig, ax1 = plt.subplots(1)
ax1.imshow(license_plate, cmap="gray")
# the next two lines is based on the assumptions that the width of
# a letter should be between 5% and 15% of the license plate,
# and height should be between 35% and 60%
# this will eliminate some
character_dimensions = (0.35*height, 0.60*height, 0.03*width, 0.15*width)
min_height, max_height, min_width, max_width = character_dimensions
print(character_dimensions)

characters = []
counter=0
column_list = []
for regions in regionprops(labelled_plate):
    y0, x0, y1, x1 = regions.bbox
    #print(y0,y1,x0,x1)
    region_height = y1 - y0
    region_width = x1 - x0
    print(region_height)
    print(region_width)
    #print('=========================================')

    if region_height > min_height and region_height < max_height and region_width > min_width and region_width < max_width:
        roi = license_plate[y0:y1, x0:x1]

        # draw a red bordered rectangle over the character.
        rect_border = patches.Rectangle((x0, y0), x1 - x0, y1 - y0, edgecolor="red",
                                       linewidth=2, fill=False)
        ax1.add_patch(rect_border)

        # resize the characters to 20X20 and then append each character into the characters list
        resized_char = resize(roi, (20, 20))
        characters.append(resized_char)

        # this is just to keep track of the arrangement of the characters
        column_list.append(x0)

print(characters)
plt.show()