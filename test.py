from pycocotools.coco import COCO
import numpy as np
import math
import skimage.io as io
from skimage.transform import resize
import matplotlib.pyplot as plt
import pylab
import sys

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

# Sample one image from each super category
crop_id = 0
for key, value in super_cat_ids.iteritems():
    img = coco.loadImgs(value[0])[0]
    # Load up an image
    I = io.imread('%s/%s/%s'%(dataDir,dataType,img['file_name']))
    io.imsave('coco_super_category_crops/super_%s.jpg'%(key), I)
    print I.shape
    annIds = []
    for cat in cat_heirarchy[key]:
        cat_id = coco.getCatIds(cat)
        annIds = annIds + coco.getAnnIds(imgIds=img['id'], catIds=cat_id, iscrowd = None)
    anns = coco.loadAnns(annIds)
    print len(anns)
    for ann in anns:
        print ann['bbox']
        x_start = int(ann['bbox'][0])
        x_end = int(ann['bbox'][0]) + int(ann['bbox'][2])
        y_start = int(ann['bbox'][1])
        y_end = int(ann['bbox'][1]) + int(ann['bbox'][3])
        crop_id = crop_id + 1
        # Adding some context
        # TODO: see if this is necessary
        #if (y_end - y_start < 256):
        #    y_mid = (y_end + y_start)/2
        #    y_start = max(y_mid - 32, 0)
        #    y_end = min(y_mid + 32, I.shape[0])
        #if (x_end - x_start < 256):
        #    x_mid = (x_end + x_start)/2
        #    x_start = max(x_mid - 32, 0)
        #    x_end = min(x_mid + 32, I.shape[1])
        I_crop = resize(I[y_start:y_end, x_start:x_end, :], (256, 256), mode='edge')
        crop_name = 'coco_super_category_crops/crop_%s_%d.jpg'%(key,crop_id)
        io.imsave(crop_name, I_crop)
sys.exit(0)
# get all images containing given categories, select one at random
catIds = coco.getCatIds(catNms=['person']);
imgIds = coco.getImgIds(catIds=catIds );
print len(imgIds)
imgs = coco.loadImgs(imgIds[0:100])
print imgs

# load up vgg run it through a few categories to see what happens
