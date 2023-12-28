#!/usr/bin/env python3

import jax
import jax.numpy as jnp
import numpy as np
#from sklearn.isotonic import IsotonicRegression
from scipy.interpolate import splrep, splev
import scipy
import time
from functools import partial
import cv2
import matplotlib.pyplot as plt
import glob
from fractions import Fraction

def from_srgb(x):
    return np.where(x < 0.04045, x/12.92, np.power((x+0.055)/1.055, 2.4))

def to_srgb(x):
    return np.where(x <= 0.0031308, x*12.92, 1.055*np.power(x, 1/2.4) - 0.055)

def robertson_weights(minv, maxv, nsteps):
    weights = jnp.linspace(-2.0,2.0,(maxv-minv)+1)
    #e4 = jnp.exp(4.0)
    #scale = e4/(e4-1)
    #shift = 1.0/(1.0-e4)
    weights = jnp.exp(-weights*weights)
    weights -= jnp.amin(weights)
    weights /= jnp.amax(weights)
    fweights = np.zeros(nsteps)
    fweights[minv:maxv+1] = weights
    return fweights

@jax.jit
def merge_robertson(images, times, response, minv, maxv, weights):
    resimg = jnp.zeros(images.shape[1:])
    wsum = jnp.zeros(images.shape[1:])
    sorted_idx = jnp.argsort(times)

    minti = jnp.ones(images.shape[1:])*1.0e6
    maxti = jnp.ones(images.shape[1:])*-1.0e6
    r_max = response[maxv]
    r_min = response[minv]

    for j in range(len(sorted_idx)):
        idx = sorted_idx[j]
        img = images[idx]
        exptime = times[idx]
        w = weights[img]
        im = response[img]
        if(j == 0):
            w = jnp.clip(w, 1e-6, None) # Always give nonzero weight to darkest frame - from LHDR
        else:
            # Get the previous (darker) image
            idx_d = sorted_idx[j-1]
            img_d = images[idx_d]
            exptime_d = times[idx_d]
            # Calculate the expected brightness of this pixel from the darker image.  If it's close to saturation, drop the weight - from LHDR
            wmod = 1.0 - jnp.power((exptime/exptime_d)*(response[img_d]/r_max), 4.0)
            wmod = jnp.clip(wmod, 0.0, None)
            w *= wmod

        # Anti-saturation from LHDR
        # Find minimum exposure time at which saturation is present,
        # and maximum exptime where black is present
        minti = jnp.where(img >= maxv, jnp.clip(minti, None, exptime), minti)
        maxti = jnp.where(img <= minv, jnp.clip(maxti, exptime, None), maxti)

        resimg += exptime*w*im
        wsum += exptime*exptime*w

    resimg = jnp.where(jnp.logical_and(wsum == 0.0, maxti > -1.0e6), r_min, resimg)
    wsum = jnp.where(jnp.logical_and(wsum == 0.0, maxti > -1.0e6), maxti, wsum)
    resimg = jnp.where(jnp.logical_and(wsum == 0.0, minti < 1.0e6), r_max, resimg)
    wsum = jnp.where(jnp.logical_and(wsum == 0.0, minti < 1.0e6), minti, wsum)
        #print("Merged image " + str(time_idx))
    return jnp.where(wsum > 0.0, resimg/wsum, 0.0)

def calibrate_robertson(images, times, init_response, maxiter = 30, threshold = 0.01):
    print("Calculating cardinality")

    card = jnp.zeros(len(init_response)).astype(jnp.uint32)
    for img in images:
        card += jnp.zeros(len(init_response)).astype(jnp.uint32).at[img].add(1)

    response = init_response

    print("Calculated cardinality")
    
    minv = np.argmax(card[:128])
    maxv = np.argmax(card[128:])+128
    print(minv)
    print(maxv)
    for j in jnp.arange(maxiter):
        weights = robertson_weights(minv, maxv, len(response))
        #plt.plot(weights)
        radiance = merge_robertson(images, times, response, minv, maxv, weights)
        print("calculated radiance for iteration " + str(j+1))
        new_response = jnp.zeros(len(response))

        time_idx = 0
        for img in images:
            new_response += times[time_idx]*jnp.zeros(len(init_response)).at[img].add(radiance)
            time_idx += 1
            #print("Processed image " + str(time_idx))
        new_response /= card
        #iso_reg = IsotonicRegression().fit(np.arange(len(new_response)), new_response)
        #new_response = iso_reg.predict(np.arange(len(new_response)))
        #new_response = np.maximum.accumulate(new_response)

        r_spline = splrep(np.arange(len(new_response[minv:maxv+1])), np.log2(new_response[minv:maxv+1]), s=np.sqrt((maxv-minv)/2))
        smoothed = splev(np.arange(len(new_response[minv:maxv+1])), r_spline, der=0)
        #Attempting to use the updated weights from Robertson's 2003 paper causes weird banding with s-log2...  Kill it for now
        #The paper doesn't really discuss how the weights are properly normalized from the derivative, given that the weights are 0
        #at both the beginning and end of the response curve, but the derivative is nonzero...  (see fig. 7)
        if(0):
            smth_der = splev(np.arange(len(new_response[minv:maxv+1])), r_spline, der=1)
            inv_smth_der = np.power(1.0/smth_der, 2.0)
            inv_smth_der -= np.amin(inv_smth_der)
            inv_smth_der /= np.amax(inv_smth_der)

            weights = np.zeros(len(new_response))
            weights[minv:maxv] = inv_smth_der[:-1]
        new_response = np.zeros(len(new_response))
        new_response[minv:maxv+1] = np.power(2,smoothed)
        new_response[:minv] = new_response[minv]
        new_response[maxv:] = new_response[maxv]
        #interpolator = scipy.interpolate.PchipInterpolator(np.arange(len(new_response)), np.log2(new_response))
        #new_response = np.power(2,interpolator(np.arange(len(new_response))))
        new_response /= jnp.amax(new_response)
        diff = jnp.sum(abs(np.log2(response/new_response)))
        print(diff)
        if(0):
            plt.plot(np.log2(new_response),weights)
            plt.figure()
            plt.semilogy(new_response)
            plt.semilogy(response)
            plt.show()
        response = new_response
        if(diff < threshold):
            break

    return (response, minv, maxv, weights)

nbits = 8
nsteps = 2**nbits

robertson_images = []
robertson_times = []

my_capture = 0
if(my_capture):
    jpegFiles = glob.glob('capture_*.jpg')
else:
    # to prep xphase files extracted with unpackORI, first
    # remove any 0-byte files (only applies to xphase scan)
    # then combined using ImageMagick's montage.  Eventually I'll find a good way to tile these here
    # for j in 0 1 2; do rm combined_$j.jpg; montage -border 0 -geometry +0+0 -tile 10x3 *_$j.jpg combined_$j.jpg; done
    jpegFiles = glob.glob('combined_?.jpg')

for fn in jpegFiles:
    (bn,ext) = fn.split('.')
    if(my_capture): #my tool
        (garbage, num, den) = bn.split('_')
        exptime = Fraction(int(num),int(den))
    else:
        (garbage, idx) = bn.split('_')
        exptime = np.power(2.0,(int(idx)-1)*2)  #Assuming 3-shot is still -2,0,+2 EV, absolute exptime doesn't matter only relative
    robertson_images.append(cv2.imread(fn))
    robertson_times.append(float(exptime))

robertson_images = np.array(robertson_images).astype(np.uint8)
robertson_times = np.array(robertson_times, dtype=np.float32)

response = from_srgb(np.linspace(0,1,nsteps))

print(robertson_images.shape)
#(card,edges) = np.histogram(robertson_images, bins=nsteps)



(response, minv, maxv, weights) = calibrate_robertson(robertson_images, robertson_times, response)
merged_img = merge_robertson(robertson_images, robertson_times, response, minv, maxv, weights)
print(np.amax(merged_img))
print(np.amin(merged_img))
np.save('response.npy', response)
plt.imshow(robertson_images[0])
plt.title("First image")
plt.figure()
plt.imshow(to_srgb(merged_img/np.amax(merged_img)))
plt.title("Merged image")
#plt.figure()
#plt.semilogy(np.arange(nsteps),card)
#plt.figure()
#plt.plot(robertson_weights(nsteps))
plt.figure()
plt.semilogy(response)
plt.show()

