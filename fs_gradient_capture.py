#!/usr/bin/env python3

import numpy as np
import cv2
import screeninfo
import gphoto2 as gp
import logging
import sys
import time
import matplotlib.pyplot as plt
import glob

from fractions import Fraction

if __name__ == '__main__':
    screen_id = 0

    # get the size of the screen
    screen = screeninfo.get_monitors()[screen_id]
    width, height = screen.width, screen.height

    # create image
    image = np.zeros((height, width, 3), dtype=np.float32)
    #print(image[:,:,0].shape)
    image[0:int(height/4),:,0] = np.tile(np.linspace(0,1,width), (int(height/4),1))
    image[0:int(height/4),:,1] = np.tile(np.linspace(0,1,width), (int(height/4),1))
    image[0:int(height/4),:,2] = np.tile(np.linspace(0,1,width), (int(height/4),1))
    image[int(height/4):int(height/2),:,0] = np.tile(np.linspace(0,1,width), (int(height/4),1))
    image[int(height/2):int(height*3/4),:,1] = np.tile(np.linspace(0,1,width), (int(height/4),1))
    image[int(height*3/4):height,:,2] = np.tile(np.linspace(0,1,width), (int(height/4),1))
    #image[:10, :10] = 0  # black at top-left corner
    #image[height - 10:, :10] = [1, 0, 0]  # blue at bottom-left
    #image[:10, width - 10:] = [0, 1, 0]  # green at top-right
    #image[height - 10:, width - 10:] = [0, 0, 1]  # red at bottom-right

    window_name = 'projector'
    cv2.namedWindow(window_name, cv2.WND_PROP_FULLSCREEN)
    cv2.moveWindow(window_name, screen.x - 1, screen.y - 1)
    cv2.setWindowProperty(window_name, cv2.WND_PROP_FULLSCREEN,
                          cv2.WINDOW_FULLSCREEN)
    cv2.imshow(window_name, image)

    cv2.waitKey(1000)

    logging.basicConfig(
        format='%(levelname)s: %(name)s: %(message)s', level=logging.WARNING)
    callback_obj = gp.check_result(gp.use_python_logging())

    camera = gp.Camera()
    camera.init()
    time.sleep(5)

    mintime = Fraction('15')
    for fn in glob.glob('capture_*.jpg'):
        (bn,ext) = fn.split('.')
        (garbage, num, den) = bn.split('_')
        exptime = Fraction(int(num), int(den))
        if(exptime < mintime):
            mintime = exptime

    # get configuration tree
    cfg = camera.get_config()
    capturetarget_cfg = cfg.get_child_by_name('capturetarget')
    capturetarget_cfg.set_value('sdram')
    camera.set_config(cfg)
    shutterspeed_cfg = cfg.get_child_by_name('shutterspeed')
    speeds_byname = []
    speeds = []
    for j in range(shutterspeed_cfg.count_choices()):
        choice = shutterspeed_cfg.get_choice(j)
        if choice != 'Bulb':
            speeds_byname.append(choice)
            speeds.append(Fraction(choice))

    start_idx = speeds.index(mintime)
    end_idx = speeds.index(Fraction('1/400'))

    for j in range(start_idx, end_idx + 1):
        if cv2.waitKey(1) & 0xFF == ord('q'): # wait for 1 millisecond
            break
        speed = speeds_byname[j]
        shutterspeed_cfg.set_value(speed)
        camera.set_config(cfg)
        speed_num = speeds[j].numerator
        speed_den = speeds[j].denominator
        speed_fmt = str(speed_num) + "_" + str(speed_den)
        tgt_file = 'capture_' + speed_fmt + '.jpg'
        path = camera.capture(gp.GP_CAPTURE_IMAGE)
        camera_file = camera.file_get(path.folder, path.name, gp.GP_FILE_TYPE_NORMAL)
        camera_file.save(tgt_file)
        time.sleep(0.1)
        camera.file_delete(path.folder, path.name)
    
    cv2.destroyAllWindows()
