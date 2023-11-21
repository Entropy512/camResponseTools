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

jpegFiles = glob.glob('capture_*.jpg')

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
def merge_robertson(images, times, response, weights):
    resimg = jnp.zeros(images.shape[1:])
    wsum = jnp.zeros(images.shape[1:])
    time_idx = 0
    for img in images:
        w = weights[img]
        im = response[img]
        resimg += times[time_idx]*w*im
        wsum += jnp.power(times[time_idx],2.0)*w
        time_idx += 1
        #print("Merged image " + str(time_idx))
    return resimg/(wsum + jnp.finfo(wsum.dtype).eps)

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
        radiance = merge_robertson(images, times, response, weights)
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

    return (response, weights)

nbits = 8
nsteps = 2**nbits

robertson_images = []
robertson_times = []
for fn in jpegFiles:
    (bn,ext) = fn.split('.')
    (garbage, num, den) = bn.split('_')
    exptime = Fraction(int(num),int(den))
    robertson_images.append(cv2.imread(fn))
    robertson_times.append(float(exptime))

robertson_images = np.array(robertson_images).astype(np.uint8)
robertson_times = np.array(robertson_times, dtype=np.float32)

response = from_srgb(np.linspace(0,1,nsteps))

print(robertson_images.shape)
#(card,edges) = np.histogram(robertson_images, bins=nsteps)



(response, weights) = calibrate_robertson(robertson_images, robertson_times, response)
merged_img = merge_robertson(robertson_images, robertson_times, response, weights)
print(np.amax(merged_img))
print(np.amin(merged_img))
np.save('response.npy', response)
plt.imshow(robertson_images[0])
plt.figure()
plt.imshow(to_srgb(merged_img/np.amax(merged_img)))
#plt.figure()
#plt.semilogy(np.arange(nsteps),card)
#plt.figure()
#plt.plot(robertson_weights(nsteps))
plt.figure()
plt.semilogy(response)
plt.show()
