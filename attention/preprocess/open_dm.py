import os
import cv2
import numpy as np

def open_dm(dm_dir, color='GRAY'):
        """ Open an dm
        Args:
            name	: Name of the sample
            color	: Color Mode (RGB/BGR/GRAY)
        """
        dm_list = []
        for root, dirs, files in os.walk(dm_dir):
            for file in files:
                
                dm = cv2.imread(os.path.join(dm_dir, file),0)
                # cv2.imshow(file,dm)
                # cv2.waitKey()
                # dm = np.array(dm)
                dm_list.append(dm)
        print(dm_list[1])
        
if __name__ == '__main__':
    open_dm('/home/hongyvsvxinlang/PycharmProjects/ubuntu/tcfkd/attention/preprocess/dm_norm')

