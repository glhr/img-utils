from skimage import io, measure
import numpy as np
from skimage.draw import polygon
from scipy import ndimage

from vision_utils.perspective_transform import apply_transform
from vision_utils.img import normalize_img, get_2d_image, save_image
from vision_utils.timing import get_timestamp

class Object:
    def __init__(self, box, image, method='polygon'):
        self.image = image
        self.img_shape = image[:,:,0].shape if image.ndim > 2 else image.shape
        self.box = box
        self.mask = self.get_mask_from_contour(method=method)
        self.shitty_contour = False
        self.area = self.get_mask_area()
        try:
            self.props = measure.regionprops(self.get_mask(type=np.uint8))[0]
            self.area = self.props.area
            self.extent = self.props.extent
            self.x = self.props.centroid[1]
            self.y = self.props.centroid[0]
            self.coords = np.array([
                [box[0], box[1]], # top-left point
                [box[2], box[3]]  # bottom-right point
              ])
            self.aspect_ratio = self.props.major_axis_length / self.props.minor_axis_length
        except Exception as e:  # wasn't able to compute regionprops, therefore contour isn't valid
            # print(e)
            self.shitty_contour = True

    def is_valid(self, constraints=['area']):
        valid = []
        if not self.shitty_contour:
            if constraints is not None:
                if 'area' in constraints and (self.area < 2000 or self.area > 100000):
                    valid.append(False)
                if 'extent' in constraints and (self.extent > 0.9 or self.extent < 0.2):
                    valid.append(False)
                if 'aspect_ratio' in constraints and (self.aspect_ratio > 4 or self.aspect_ratio < 0.25):
                    valid.append(False)
                return np.all(valid)
            return True
        return False

    def has_placement(self, placement='any'):

        if placement is None or placement == 'any':
            return True
        else:
            img_y = self.img_shape[0]/2
            img_x = self.img_shape[1]/2
            if placement == 'middle':
                print("Checking object placement ({},{}) relative to image size ({},{})".format(
                    self.x,
                    self.y,
                    img_x,
                    img_y
                ))
                area_w = 300
                area_h = 300
                if (img_x - area_w <= self.x <= img_x + area_w) and (img_y - area_h <= self.y <= img_y + area_h):
                    print("--> Middle? True")
                    return True
                else:
                    print("--> Middle? False")
                    return False
            elif placement == 'left' and (self.x < img_x):
                print("--> Left? True")
                return True
            elif placement == 'right' and (self.x > img_x):
                print("--> Right? True")
                return True
            return False

    def get_image(self, type=np.uint8, range=255):
        return self.image.astype(type)*range

    def get_flattened_coords(self):
        return self.coords.flatten()

    def get_mask(self, type=bool, range=1):
        return self.mask.astype(type)*range if not type == bool else self.mask.astype(type)

    def get_mask_area(self):
        return np.sum(self.get_mask(type=np.uint8))

    def get_mask_from_contour(self, method):

        binary_mask = np.zeros(self.img_shape, dtype='bool')
        x0,y0,x1,y1 = self.box
        binary_mask[y0:y1,x0:x1] = 1

        return binary_mask

    def get_masked_image(self, range=255):
        image = self.image
        assert image.ndim > 2, "get_masked_image: expecting 3-channel image"
        masked = (image.copy()*range).astype(np.uint8)
        masked[~self.mask,:] = 0
        assert masked.ndim > 2, "get_masked_image: expecting 3-channel masked output"

        return masked

    def get_crop(self, binary=False, range=255):
        if not binary:
            image = self.image
        else:
            mask = self.get_mask(type=np.uint8, range=range)
            image = np.repeat(mask[:, :, np.newaxis], 3, axis=2)
        return image[self.coords[0][1]:self.coords[1][1], self.coords[0][0]:self.coords[1][0]]
