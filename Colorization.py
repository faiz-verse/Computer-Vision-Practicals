import cv2
import numpy as np
import os

base_dir = os.path.dirname(os.path.abspath(__file__))
model_dir = os.path.join(base_dir, 'Model')

prototxt = os.path.join(model_dir, 'colorization_deploy_v2.prototxt')
model = os.path.join(model_dir, 'colorization_release_v2.caffemodel')
pts_npy = os.path.join(model_dir, 'pts_in_hull.npy')

net = cv2.dnn.readNetFromCaffe(prototxt, model)
pts = np.load(pts_npy)

class8_ab = net.getLayerId('class8_ab')
conv8_313_rh = net.getLayerId('conv8_313_rh')
net.getLayer(class8_ab).blobs = [pts.transpose().reshape(2, 313, 1, 1)]
net.getLayer(conv8_313_rh).blobs = [np.full([1, 313], 2.606, dtype='float32')]

image_path = 'Monster1.jpg'
img = cv2.imread(image_path)

if img is None:
    print("Failed to load image.")
    exit()

cv2.imshow("Original Image", img)


img_lab = cv2.cvtColor(img, cv2.COLOR_BGR2Lab)
img_l = img_lab[:, :, 0]

img_rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
img_resized = cv2.resize(img_rgb, (224, 224))
img_lab_rs = cv2.cvtColor(img_resized, cv2.COLOR_RGB2Lab)
l_rs = img_lab_rs[:, :, 0]
l_rs -= 50

blob = cv2.dnn.blobFromImage(l_rs, scalefactor=1.0, size=(224, 224), mean=(50,))
net.setInput(blob)
ab_dec = net.forward()[0, :, :, :].transpose((1, 2, 0))

ab_dec_us = cv2.resize(ab_dec, (img.shape[1], img.shape[0]))

lab_out = np.concatenate((img_l[:, :, np.newaxis], ab_dec_us), axis=2)
bgr_out = cv2.cvtColor(lab_out.astype(np.uint8), cv2.COLOR_Lab2BGR)

cv2.imshow("Colorized Image", bgr_out)
cv2.waitKey(0)
cv2.destroyAllWindows()
