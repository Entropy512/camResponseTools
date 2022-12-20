#!/usr/bin/env python3

import numpy as np
import matplotlib.pyplot as plt
import glob
from scipy import interpolate
import splines

npyFiles = glob.glob('*.npy')

hlg_trc = []
hlg_a = 0.17883277
hlg_b = 0.28466892
hlg_c = 0.55991073


lin_codes = np.linspace(0,1,256)
hlg_trc = np.where(lin_codes < 0.5, 4*lin_codes**2, np.exp((lin_codes-hlg_c)/hlg_a)+hlg_b)
hlg_trc /= np.amax(hlg_trc)

def from_srgb(x):
    return np.where(x < 0.04045, x/12.92, np.power((x+0.055)/1.055, 2.4))

def to_srgb(x):
    return np.where(x <= 0.0031308, x*12.92, 1.055*np.power(x, 1/2.4) - 0.055)

srgb_trc = from_srgb(lin_codes)

rec709_trc = np.where(lin_codes < 0.081, lin_codes/4.5, np.power((lin_codes+0.099)/1.099, 1/0.45))

responses = {}
for fn in npyFiles:
    (bn,ext) = fn.split('.')
    if(bn not in ['cine2', 'slog2']):
        responses[bn] = np.reshape(np.load(fn)[0:254,:,1],254)
        responses[bn] /= np.amax(responses[bn])
        responses[bn] *= srgb_trc[253]
        plt.plot(to_srgb(responses[bn]), lin_codes[0:254], label=bn)
    elif(bn == 'slog2'):
        responses[bn] = np.reshape(np.load(fn)[0:252,:,1],252)
        responses[bn] /= np.amax(responses[bn])
        responses[bn] *= srgb_trc[251]
        plt.plot(to_srgb(responses[bn]), lin_codes[0:252], label=bn)



splx = [0, 0.11, 0.32, 0.66, 1]
sply = [0, 0.09, 0.43, 0.87, 1]
#tck = interpolate.splrep(splx, sply, s=0)
x = np.linspace(0,1,255)
rt_interpolator = interpolate.PchipInterpolator(splx, sply)
rt_inv = interpolate.PchipInterpolator(sply, splx)

plt.plot(x, rt_interpolator(x), label='RawTherapee Standard Film Curve')

amtc_points = [(0, 0),
               (0.05, 0.0332268),
               (0.12, 0.119268),
               (0.218, 0.297577),
               (0.3552, 0.575063),
               (0.54728, 0.85685),
               (0.816192, 0.971065),
               (1, 1)]

amtc_spline = splines.CatmullRom(amtc_points)

splinex = amtc_spline.evaluate(x*7).T[0]
plt.plot(splinex, amtc_spline.evaluate(x*7).T[1], label='A7M3 RT AMTC')


plt.plot(to_srgb(rec709_trc), lin_codes, label='Rec. 709 Standard')
plt.plot(to_srgb(srgb_trc), lin_codes, label='sRGB Standard')
plt.plot(to_srgb(hlg_trc), lin_codes, label='HLG Standard')

plt.title('Effective Tone Curve applied after sRGB encoding')
plt.xlabel('Input value')
plt.ylabel('Output value')

plt.legend()
plt.show()
