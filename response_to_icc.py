#!/usr/bin/env python3

import imagecodecs
import numpy as np
import argparse
import os
import sys
import matplotlib.pyplot as plt

ap = argparse.ArgumentParser()
ap.add_argument('-i', '--input', required=True,
    help='path to input file')

args = vars(ap.parse_args())

filebase = os.path.splitext(args['input'])[0]
iccname = filebase + '.icc'

response = np.load(args['input'])

response_g = np.reshape(response[:,:,1],response.shape[0])

#make it monotonic, starting from the middle since the ends are usually the least stable
response_mono = np.concatenate([np.minimum.accumulate(response_g[126::-1])[::-1], np.maximum.accumulate(response_g[127:])])

#The clamp value always seems to get a really screwy response, replace it with a value that is derived from the ratio of the previous two,
#e.g. constant slope in log space
response_mono[-1] = response_mono[-2]*(response_mono[-2]/response_mono[-3])

response_mono[0] = response_mono[1]*(response_mono[2]/response_mono[3])
#Peak at 1.0
response_mono /= np.amax(response_mono)

#FIXME:  Don't hardcode to sRGB primaries and whitepoint
icc_profile = imagecodecs.cms_profile('rgb', whitepoint=[0.3127,0.3290,1.0], primaries=[0.64, 0.33, 0.2126, 0.3, 0.6, 0.7152, 0.15, 0.06, 0.0722], transferfunction=response_mono)

with open(iccname, 'wb') as icc_file:
    icc_file.write(icc_profile)

plt.semilogy(response_mono)
plt.show()