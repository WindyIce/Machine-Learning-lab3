from PIL import Image
import glob,os
import numpy as np

size=24,24
counter=0
for faceDatasets in glob.glob('datasets/original/face/*.jpg'):
    file,ext=os.path.splitext(faceDatasets)
    im=Image.open(faceDatasets)

    #im = im.resize(size)
    im = np.array(im)
    R = im[:, :, 0]
    G = im[:, :, 1]
    B = im[:, :, 2]
    grey = (R + G + B) / 3
    Image.fromarray(grey).save('datasets/grey/face/'+counter.__str__()+'.tif')

    counter+=1

