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
        for i in range(1, 25):
            dm = cv2.imread(os.path.join(dm_dir, 'heat'+str(i)+'.png'), 0)
            # cv2.imshow(file,dm)
            # cv2.waitKey()
            # dm = np.array(dm)
            dm_list.append(dm)
            print(dm_list)
        
if __name__ == '__main__':
    open_dm('/home/hongyvsvxinlang/PycharmProjects/ubuntu/tcfkd/attention/preprocess/dm_norm')

