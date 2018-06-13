import cv2
import os
import numpy as np
from skimage import io
ht_dir = './ht'
dm_dir = './dm'
dm_normalization_dir = './dm_norm'
def resize_dm():
    for root, dirs, files in os.walk(ht_dir):
            for file in files:
                dm = cv2.imread(os.path.join(ht_dir,file),0)
                dm = cv2.resize(dm,(64,64))
                cv2.imwrite(os.path.join(dm_dir,file),dm)

def test():
    heat1 = io.imread(os.path.join(dm_dir,'heat1.png'))
    print(np.max(heat1))

def normalization():
    for root, dirs, files in os.walk(dm_dir):
            for file in files:
                dm = cv2.imread(os.path.join(dm_dir,file),0)
                dm = dm/255.
                cv2.imwrite(os.path.join(dm_normalization_dir,file),dm)
                

if __name__ == "__main__":
    # test()
    normalization()

    
    


    



