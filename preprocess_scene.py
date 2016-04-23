from pycocotools.coco import COCO
import numpy as np
import math
import skimage.io as io
from skimage.transform import resize
from skimage.exposure import is_low_contrast
import matplotlib.pyplot as plt
import pylab
import sys
import random

dataDir='/rmullapu-local'
dataType='train2014'
annFile='%s/annotations/instances_%s.json'%(dataDir,dataType)

# initialize COCO api for instance annotations
coco=COCO(annFile)

# display COCO categories and supercategories
cats = coco.loadCats(coco.getCatIds())
nms=[cat['name'] for cat in cats]
print 'COCO categories: \n\n', ' '.join(nms)

nms = set([cat['supercategory'] for cat in cats])
print 'COCO supercategories: \n', ' '.join(nms)

cat_heirarchy = {}
for super_cat in nms:
    sub_cats = []
    for cat in cats:
        if cat['supercategory'] == super_cat:
             sub_cats.append(cat['name'])
    cat_heirarchy[super_cat] = sub_cats

super_cat_ids = {}
for key, value in cat_heirarchy.iteritems():
    catIds = coco.getCatIds(catNms=value);
    imgIds = []
    for cat in catIds:
        imgIds = imgIds + coco.getImgIds(catIds = [cat]);
    super_cat_ids[key] = imgIds;

# Statistics on super category instances
#total = 0
#for key, value in super_cat_ids.iteritems():
#    print key, len(value)
#    total = total + len(value)
#print total

label_map = {}
class_id = 0
for key, value in super_cat_ids.iteritems():
    num_samples = 0
    if (not (key == 'outdoor' or key == 'indoor')):
        continue
    print "Processing ", key
    print
    for val in value:
        if num_samples > 10000:
            break
        img = coco.loadImgs(val)[0]
        # Load up an image
        I = io.imread('%s/%s/%s'%(dataDir,dataType,img['file_name']))
        #io.imsave('coco_super_category_crops/super_%s.jpg'%(key), I)
        #print I.shape
        try:
            if (len(I.shape) == 3):
                I_crop = resize(I, (256, 256), mode='edge')
            else:
                I_crop = resize(I, (256, 256), mode='edge')
            scene_name = 'scene_%s_%d.jpg'%(key, num_samples)
            if is_low_contrast(I_crop):
                continue
            label_map[scene_name] = class_id
            io.imsave('coco_indoor_outdoor/%s'%(scene_name), I_crop)
            num_samples = num_samples + 1
        except:
            continue
    class_id = class_id + 1


# Write the image names and labels in a shuffled order
img_names = list(label_map.keys())
random.shuffle(img_names)

label_file = open('scene_labels.txt', 'w')
for name in img_names:
    label_file.write('%s %d\n'%(name, label_map[name]))
label_file.close()
