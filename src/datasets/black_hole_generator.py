import numpy as np
from albumentations.core.transforms_interface import ImageOnlyTransform
import albumentations as A

class Black_hole_aug(ImageOnlyTransform):
    def __init__(self, safe_db_lists=[], prob=0.5) -> None:
        super(Black_hole_aug, self).__init__(p=prob)
        self.safe_db_lists = safe_db_lists
        self.prob = prob

    def insert_black_circles(self, img, num_circles=40, min_radius_perc=0.001, max_radius_perc=0.05):
        """
        Insert black circles into the image using numpy.
        """
        img_array = np.array(img)
        h, w = img_array.shape
        num_circles = np.random.randint(20, 60)
        for _ in range(num_circles):
            radius = np.random.randint(h * min_radius_perc, h * max_radius_perc)
            center = (np.random.randint(w * 0.2, w * 0.8), np.random.randint(0.2 * h, h * 0.8))
            y, x = np.ogrid[:h, :w]
            mask = (x - center[0])**2 + (y - center[1])**2 <= radius**2
            
            offset = np.random.uniform(-2,0)
            img_array[mask] += [float(offset)]  # Set the circle area to black for all channels
            
        return img_array.clip(-1,1)

    def apply(self, img, copy=True, **params):
        if copy:
            img = img.copy()
        img = self.insert_black_circles(img)
        return img