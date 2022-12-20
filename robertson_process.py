#!/usr/bin/env python3

import numpy as np
import cv2
import matplotlib.pyplot as plt
import glob
from fractions import Fraction

jpegFiles = glob.glob('capture_*.jpg')

robertson_images = []
robertson_times = []
for fn in jpegFiles:
    (bn,ext) = fn.split('.')
    (garbage, num, den) = bn.split('_')
    exptime = Fraction(int(num),int(den))
    robertson_images.append(cv2.imread(fn))
    robertson_times.append(float(exptime))

robertson_times = np.array(robertson_times, dtype=np.float32)

calibrate_robertson = cv2.createCalibrateRobertson()

response_robertson = calibrate_robertson.process(robertson_images, robertson_times)


np.save('response_r.npy',response_robertson)
plt.plot(response_robertson[0:,:,0], 'r')
plt.plot(response_robertson[0:,:,1], 'g')
plt.plot(response_robertson[0:,:,2], 'b')
plt.show()
