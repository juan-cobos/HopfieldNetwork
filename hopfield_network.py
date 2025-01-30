import numpy as np

class HopfieldNetwork:
    def __init__(self, img_lst):
        self.mem = img_lst
        self.w = np.zeros((img_lst[0].size, img_lst[0].size))
        
        for i in range(len(img_lst)):
            self.w = np.add(self.w, np.outer(img_lst[i], img_lst[i].T))

    def retrieve(self, mask):
        s = mask.flatten()
        
        while True:
            ret_img = s.reshape(mask.shape)
            if any((ret_img == x).all() for x in self.mem):
                return ret_img
            s = np.sign(np.dot(self.w, s))
        
            
