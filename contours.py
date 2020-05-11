from skimage import io
import numpy as np

from skimage import measure
try:
    from perspective_transform import apply_transform
    from img import normalize_img, get_2d_image
except ImportError:
    from utils.perspective_transform import apply_transform
    from utils.img import normalize_img, get_2d_image

from skimage.draw import polygon
from scipy import ndimage


class Object:
    def __init__(self, contour, image, method='polygon'):
        self.image = image
        self.img_shape = image[:,:,0].shape if image.ndim > 2 else image.shape
        self.contour = contour
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
                [self.props.bbox[1], self.props.bbox[0]], # top-left point
                [self.props.bbox[3], self.props.bbox[2]]  # bottom-right point
              ])
        except Exception as e:
            self.shitty_contour = True

    def is_valid(self, constraints=['area']):
        if not self.shitty_contour:
            if constraints is not None and 'area' in constraints and (self.area < 2000 or self.area > 100000):
                return False
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

        if not (self.contour[0] == self.contour[-1]).all():  # close open contours (on the edges)
            self.contour = np.concatenate((self.contour, [self.contour[0]]), axis=0)

        # Create an empty image to store the masked array
        mask = np.zeros(self.img_shape, dtype='bool')

        if method == 'binary_fill':
            # Create a contour image by using the contour coordinates rounded to their nearest integer value
            mask[np.round(self.contour[:, 0]).astype('int'), np.round(self.contour[:, 1]).astype('int')] = 1
            # Fill in the hole created by the contour boundary
            mask = ndimage.binary_fill_holes(mask)
        # mask = polygon2mask(image[:,:,0].shape, contour)
        elif method == 'polygon':
            rr, cc = polygon(self.contour[:, 0], self.contour[:, 1], mask.shape)
            mask[rr, cc] = 1
        return mask

    def get_masked_image(self, range=255):
        image = self.image
        assert image.ndim > 2, "get_masked_image: expecting 3-channel image"
        masked = (image.copy()*range).astype(np.uint8)
        masked[~self.mask,:] = range
        assert masked.ndim > 2, "get_masked_image: expecting 3-channel masked output"

        return masked

    def get_crop(self, binary=False, range=255):
        if not binary:
            image = self.image
        else:
            mask = self.get_mask(type=np.uint8, range=range)
            image = np.repeat(mask[:, :, np.newaxis], 3, axis=2)
        return image[self.coords[0][1]:self.coords[1][1], self.coords[0][0]:self.coords[1][0]]

    def get_binary_contour(self):
        mask = np.zeros(self.img_shape, dtype='bool')
        mask[np.round(self.contour[:, 0]).astype('int'), np.round(self.contour[:, 1]).astype('int')] = 1
        return mask


def get_contours(image, level=0.8):
    """Find iso-valued contours in a 2D array for a given level value, using skimage's find_contours() method.

    Parameters
    ----------
    image : 2D ndarray of double
        Input data in which to find contours.
    level : float
        Value along which to find contours in the array.

    Returns
    -------
    list of (n,2)-ndarrays
        Each contour is an ndarray of shape (n, 2), consisting of n (x, y) coordinates along the contour.
        Output contours are not guaranteed to be closed: contours which intersect the array edge will be left open.
        The order of the contours in the output list is determined by the position of the smallest x,y coordinate.
    """
    return measure.find_contours(image,
                                 level=level,
                                 fully_connected='high',
                                 positive_orientation='high')


def select_best_object(objects, constraints=None, placement='any'):
    """Given a list of Objects, try to find the object with the largest area.
    If constraints are provided, only objects considered "valid" are considered, therefore the function may return None.

    Parameters
    ----------
    objects : list of Object
        List of Objects to search in.
    constraints : list of string
        Constraints to use when checking for object validity. eg. constraints=['area']
        See the is_valid() method in the Object class.
    placement : String
        'any', 'left', 'right' or 'middle'
        If 'any', then all valid objects are returned.
        Otherwise the function only returns objects located in the designated area.
        See the has_placement() method in the Object class.

    Returns
    -------
    Object or None
        Returns largest object, or returns None if constraints are provided and all the objects are invalid.

    """
    # only consider valid objects if constraints provided
    if (constraints is not None) or (placement is not None):
        objects = filter_objects(objects, constraints, placement)

    if len(objects):
        # find index of object with largest mask area
        i = np.argmax([object.area for object in objects])
        # return largest object
        return objects[i]

    return None


def filter_objects(objects, constraints=['area'], placement='any'):
    """Given a list of Objects, return a subset of the list containing only valid objects.

    Parameters
    ----------
    objects : list of Object
        Description of parameter `objects`.
    constraints : list of string
        Constraints to use when checking for object validity. eg. constraints=['area']
        See the is_valid() method in the Object class.
    placement : String
        'any', 'left', 'right' or 'middle'
        If 'any', then all valid objects are returned.
        Otherwise the function only returns objects located in the designated area.
        See the has_placement() method in the Object class.

    Returns
    -------
    list of Object
        list of valid Objects

    """
    return [object for object in objects if object.is_valid(constraints) and object.has_placement(placement)]


def get_object_crops(image_orig, transform=True, placement='any'):
    """Performs object detection and segmentation.

    Parameters
    ----------
    image_orig : 2D or 3D ndarray of int, float, or double
        Image used for finding contours.
    transform : boolean
        If True, the image will first be rectified using the DEFAULT_TRANSFORM defined in perspective_transform.py.
        The resulting objects will then be transformed back to their original reference frame.
        If False, the original image will be used for object detection.
    placement : String
        'any', 'left', 'right' or 'middle'
        If 'any', then all valid objects are returned.
        Otherwise the function only returns objects located in the designated area.
        See the has_placement() method in the Object class.

    Returns
    -------
    list of Objects
        A list of "valid" Objects found in the input image. See the Object class to see object properties.
    """
    if transform:
        image_tf = apply_transform(image_orig)  # geometric transform
        image_tf = normalize_img(image_tf)  # make sure it's RGB & in range (0,1)
    else:
        image_tf = normalize_img(image_orig)

    image_value = get_2d_image(image_tf)  # select a single channel for contouring

    # Find contours, select good ones and get crop coordinates
    contours = get_contours(image_value)
    objects = filter_objects([Object(contour, image_tf) for contour in contours], placement=placement)

    if transform:
        for object in objects:
            object.image = image_orig
            object.mask = apply_transform(object.get_mask(type=np.uint8), inverse=True).astype(bool)
            object.coords = apply_transform(object.coords, coords=True)
            object.img_cropped = object.get_crop(binary=False)
    return objects


if __name__ == '__main__':
    import matplotlib.pyplot as plt

    image = io.imread("test/1588369634 test-orig.png")
    image = apply_transform(image)
    image = normalize_img(image)
    image_value = get_2d_image(image)

    # Find contours at a constant value of 0.8
    contours = get_contours(image_value)
    print("Found {} contours".format(len(contours)))

    # Display the image and plot all contours found
    fig, ax = plt.subplots(ncols=2, figsize=(8, 3))
    ax[0].imshow(image)
    for n, contour in enumerate(contours):
        ax[0].plot(contour[:, 1], contour[:, 0], linewidth=2)

    objects = [Object(contour, image) for contour in contours]

    masked = select_best_object(objects, constraints=["area"]).get_masked_image()

    io.imsave("test/mask.png", masked)
    ax[1].imshow(masked)
    for a in ax:
        a.axis('image')
        a.set_xticks([])
        a.set_yticks([])

    plt.tight_layout()
    # plt.savefig('test/contours_plot.png', dpi=500)
    plt.show()

    # GET CROP

    image_orig = io.imread("test/1588369634 test-orig.png")
    objects = get_object_crops(image_orig, transform=True)
    print([object.coords for object in objects])