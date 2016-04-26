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
dataType='val2014'
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

num_super = 0
super_id = {}
for key, value in super_cat_ids.iteritems():
    super_id[key] = num_super
    num_super = num_super + 1

# Count the number of crops in each super category
num_super_crops = {}
for key, value in super_cat_ids.iteritems():
    num_crops = 0
    for val in value:
        img = coco.loadImgs(val)[0]
        annIds = []
        for cat in cat_heirarchy[key]:
            cat_id = coco.getCatIds(cat)
            annIds = annIds + coco.getAnnIds(imgIds=img['id'], catIds=cat_id, iscrowd = None)
        anns = coco.loadAnns(annIds)
        #print len(anns)
        for ann in anns:
            num_crops = num_crops + 1
    num_super_crops[key] = num_crops

for key, value in num_super_crops.iteritems():
    print key, value

# Make crops for each super category (grab a 5th of the total data)
crop_id = 0
label_map = {}
for key, value in super_cat_ids.iteritems():
    num_crops = 0
    print "Processing ", key
    print
    for val in value:
        if num_crops > (num_super_crops[key]/50):
            break
        img = coco.loadImgs(val)[0]
        # Load up an image
        I = io.imread('%s/%s/%s'%(dataDir,dataType,img['file_name']))
        #io.imsave('coco_super_category_crops/super_%s.jpg'%(key), I)
        #print I.shape
        annIds = []
        for cat in cat_heirarchy[key]:
            cat_id = coco.getCatIds(cat)
            annIds = annIds + coco.getAnnIds(imgIds=img['id'], catIds=cat_id, iscrowd = None)
        anns = coco.loadAnns(annIds)
        #print len(anns)
        for ann in anns:
            #print ann['bbox']
            x_start = int(ann['bbox'][0])
            x_end = int(ann['bbox'][0]) + int(ann['bbox'][2])
            y_start = int(ann['bbox'][1])
            y_end = int(ann['bbox'][1]) + int(ann['bbox'][3])
            crop_id = crop_id + 1
            # Adding some context
            if (y_end - y_start < 256):
                y_start = max(y_start - 16, 0)
                y_end = min(y_end + 16, I.shape[0])
            if (x_end - x_start < 256):
                x_start = max(x_start - 16, 0)
                x_end = min(x_end + 16, I.shape[1])
            try:
                if (len(I.shape) == 3):
                    I_crop = resize(I[y_start:y_end, x_start:x_end, :], (256, 256), mode='edge')
                else:
                    I_crop = resize(I[y_start:y_end, x_start:x_end], (256, 256), mode='edge')
                crop_name = 'crop_%s_%d.jpg'%(key,crop_id)
                if is_low_contrast(I_crop):
                    continue
                label_map[crop_name] = super_id[key]
                io.imsave('coco_super_category_crops_val/%s'%(crop_name), I_crop)
                num_crops = num_crops + 1
            except:
                continue

# Write the image names and labels in a shuffled order
img_names = list(label_map.keys())
random.shuffle(img_names)

label_file = open('superclass_labels_val.txt', 'w')
for name in img_names:
    label_file.write('%s %d\n'%(name, label_map[name]))
label_file.close()

# load up vgg run it through a few categories to see what happens
