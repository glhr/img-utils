from sklearn import preprocessing
from skimage.color import gray2rgb, rgb2hsv, rgb2gray, rgb2ycbcr
from skimage.io import imsave, imread
from skimage.exposure import equalize_hist
from skimage.transform import resize
import numpy as np
import cv2
from sensor_msgs.msg import Image
import sys

WHITE = (255, 255, 255)
RED = (255, 0, 0)
BLACK = (0, 0, 0)

def load_image(path):
    return imread(path)

def save_image(image, path):
    imsave(path, image, check_contrast=False)

def bgr_to_rgb(image):
    rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
    # print(type(rgb),np.min(rgb),np.max(rgb))
    return rgb

def rgb_to_bgr(image):
    bgr = cv2.cvtColor(image, cv2.COLOR_RGB2BGR)
    # print(type(rgb),np.min(rgb),np.max(rgb))
    return bgr

def adjust_image_range(image, max):
    if max == 1:
        if np.max(image) > 1:
            image = image/255
    elif max == 255:
        if np.max(image) <= 1:
            image = image*255
    return image

name_to_dtypes = {
	"rgb8":    (np.uint8,  3),
	"rgba8":   (np.uint8,  4),
	"rgb16":   (np.uint16, 3),
	"rgba16":  (np.uint16, 4),
	"bgr8":    (np.uint8,  3),
	"bgra8":   (np.uint8,  4),
	"bgr16":   (np.uint16, 3),
	"bgra16":  (np.uint16, 4),
	"mono8":   (np.uint8,  1),
	"mono16":  (np.uint16, 1),

    # for bayer image (based on cv_bridge.cpp)
	"bayer_rggb8":	(np.uint8,  1),
	"bayer_bggr8":	(np.uint8,  1),
	"bayer_gbrg8":	(np.uint8,  1),
	"bayer_grbg8":	(np.uint8,  1),
	"bayer_rggb16":	(np.uint16, 1),
	"bayer_bggr16":	(np.uint16, 1),
	"bayer_gbrg16":	(np.uint16, 1),
	"bayer_grbg16":	(np.uint16, 1),

    # OpenCV CvMat types
	"8UC1":    (np.uint8,   1),
	"8UC2":    (np.uint8,   2),
	"8UC3":    (np.uint8,   3),
	"8UC4":    (np.uint8,   4),
	"8SC1":    (np.int8,    1),
	"8SC2":    (np.int8,    2),
	"8SC3":    (np.int8,    3),
	"8SC4":    (np.int8,    4),
	"16UC1":   (np.uint16,   1),
	"16UC2":   (np.uint16,   2),
	"16UC3":   (np.uint16,   3),
	"16UC4":   (np.uint16,   4),
	"16SC1":   (np.int16,  1),
	"16SC2":   (np.int16,  2),
	"16SC3":   (np.int16,  3),
	"16SC4":   (np.int16,  4),
	"32SC1":   (np.int32,   1),
	"32SC2":   (np.int32,   2),
	"32SC3":   (np.int32,   3),
	"32SC4":   (np.int32,   4),
	"32FC1":   (np.float32, 1),
	"32FC2":   (np.float32, 2),
	"32FC3":   (np.float32, 3),
	"32FC4":   (np.float32, 4),
	"64FC1":   (np.float64, 1),
	"64FC2":   (np.float64, 2),
	"64FC3":   (np.float64, 3),
	"64FC4":   (np.float64, 4)
}

def image_to_numpy(msg):
	if not msg.encoding in name_to_dtypes:
		raise TypeError('Unrecognized encoding {}'.format(msg.encoding))

	dtype_class, channels = name_to_dtypes[msg.encoding]
	dtype = np.dtype(dtype_class)
	dtype = dtype.newbyteorder('>' if msg.is_bigendian else '<')
	shape = (msg.height, msg.width, channels)

	data = np.fromstring(msg.data, dtype=dtype).reshape(shape)
	data.strides = (
		msg.step,
		dtype.itemsize * channels,
		dtype.itemsize
	)

	if channels == 1:
		data = data[...,0]
	return data


def numpy_to_image(arr, encoding):
        if not encoding in name_to_dtypes:
                raise TypeError('Unrecognized encoding {}'.format(encoding))

        im = Image(encoding=encoding)

        # extract width, height, and channels
        dtype_class, exp_channels = name_to_dtypes[encoding]
        dtype = np.dtype(dtype_class)
        if len(arr.shape) == 2:
                im.height, im.width, channels = arr.shape + (1,)
        elif len(arr.shape) == 3:
                im.height, im.width, channels = arr.shape
        else:
                raise TypeError("Array must be two or three dimensional")

        # check type and channels
        if exp_channels != channels:
                raise TypeError("Array has {} channels, {} requires {}".format(
                        channels, encoding, exp_channels
                ))
        if dtype_class != arr.dtype.type:
                raise TypeError("Array is {}, {} requires {}".format(
                        arr.dtype.type, encoding, dtype_class
                ))

        # make the array contiguous in memory, as mostly required by the format
        contig = np.ascontiguousarray(arr)
        im.data = contig.tostring()
        im.step = contig.strides[0]
        im.is_bigendian = (
                arr.dtype.byteorder == '>' or
                arr.dtype.byteorder == '=' and sys.byteorder == 'big'
        )

        return im

def normalize_img(image, resize_shape=None):
    if resize_shape is not None:
        image = resize(image, output_shape=resize_shape)
    # print(image.shape)
    if len(image.shape) < 3:
        image = gray2rgb(image)
    elif image.shape[-1] > 3:
        image = image[:,:,:3]
    # image = resize(image, (100,100,3))
    if np.max(image) > 1:
        image = image.astype(np.float32)/255
    # print(image.shape)
    return image


def get_2d_image(image, equalize_histo=False):
    if equalize_histo:
        return equalize_hist(rgb2gray(image))
    return rgb2gray(image)
    # return image[:,:,2]


def get_histos(image, mask, bins, channels='hsv'):
    if channels == 'hsv':
        image = rgb2hsv(image)
    elif channels == 'ycbcr':
        image = rgb2ycbcr(image)
    if np.max(image) > 1:
        range = (0,255)
    else:
        range = (0,1)

    masked_array_h = np.ma.masked_array(image[:,:,0], mask=~mask)
    masked_array_s = np.ma.masked_array(image[:,:,1], mask=~mask)
    masked_array_v = np.ma.masked_array(image[:,:,2], mask=~mask)

    histo_h = np.histogram(masked_array_h.compressed(), range=range, bins=bins)
    histo_s = np.histogram(masked_array_s.compressed(), range=range, bins=bins)
    histo_v = np.histogram(masked_array_v.compressed(), range=range, bins=bins)

    return (histo_h, histo_s, histo_v)


def get_feature_vector(image, mask, bins=10, channels='hsv'):
    histos = get_histos(image, mask, bins, channels)
    histos = [histo[0] for histo in histos]
    vector = np.hstack(histos)
    vector = list(preprocessing.normalize([vector])[0])
    return vector

# from https://pypi.org/project/ImageMosaic/
class MosaicException(Exception):
    pass

def create_mosaic(
        images, nrows=None, ncols=None, border_val=0, border_size=0,
        rows_first=True):
    """
    Creates a mosaic of input images. 'images' should be an iterable of
    2D or 3D images. If they're 3D then they must have the shape
    (spatial, spatial, channels).

    Creates a square mosaic unless nrows and/or ncols is not 'None'. If both
    are not 'None' then ncols * nrows >= len(images)

    :param images: An iterable of 2D or 3D numpy arrays
    :param nrows: Custom number of rows
    :param ncols: Custom number of columns
    :param border_val: The value of the background and borders
    :param border_size: The size of the border between images
    :param rows_first: A boolean indicating if the mosaic should be populated
    by rows (True) or columns (False) first
    :return:
    """

    # Validate
    if border_size < 0:
        raise MosaicException("'border_size' must be >= 0")

    # Trivial cases
    if len(images) == 0:
        return None

    if len(images) == 1:
        return images[0]

    imshapes = [image.shape for image in images]
    imdims = [len(imshape) for imshape in imshapes]
    dtypes = [image.dtype for image in images]

    # Images must be all 2D or 3D
    if not set(imdims).issubset({2, 3}):
        raise MosaicException("All images must be 2D or 3D numpy arrays")

    # Images must all have the same shape
    if len(set(imdims)) != 1:
        raise MosaicException("Images cannot be mixed 2D and 3D")

    # If images are 3D then they must all have the same number of channels
    if imdims[0] == 3:
        num_channels = [imshape[2] for imshape in imshapes]
        if len(set(num_channels)) != 1:
            raise MosaicException(
                "3D images must have the same number of channels")

    # Images must all have the same dtype
    if len(set(dtypes)) != 1:
        raise MosaicException("All images must have the same dtype")

    # Embed images if they're differently sized
    if len(set(imshapes)) != 1:
        new_images = []
        max_dims = [
            max([imshape[i] for imshape in imshapes])
            for i in range(imdims[0])]
        for image in images:
            diff = [max_dims[i] - image.shape[i] for i in range(len(max_dims))]
            embed_box = border_val * np.ones(max_dims)
            start0 = diff[0] // 2
            end0 = start0 + image.shape[0]
            start1 = diff[1] // 2
            end1 = start1 + image.shape[1]
            embed_box[start0:end0, start1:end1] = image
            new_images.append(embed_box)
        images = new_images

    # Set up grid
    if ncols is None and nrows is None:
        n_cols = int(np.ceil(np.sqrt(len(images))))
        n_rows = int(np.ceil(len(images) / n_cols))
    elif ncols is None and nrows is not None:
        n_rows = nrows
        n_cols = int(np.ceil(len(images) / n_rows))
    elif ncols is not None and nrows is None:
        n_cols = ncols
        n_rows = int(np.ceil(len(images) / n_cols))
    else:
        n_cols = ncols
        n_rows = nrows

    if n_rows <= 0 or n_cols <= 0:
        raise MosaicException("'nrows' and 'ncols' must both be > 0")

    if n_cols * n_rows < len(images):
        raise MosaicException(
            "Mosaic grid too small: n_rows * n_cols < len(images)")

    size_0 = images[0].shape[0]
    size_1 = images[0].shape[1]

    if imdims[0] == 2:
        out_block = border_val * np.ones(
            (size_0 * n_rows + border_size * (n_rows - 1),
             size_1 * n_cols + border_size * (n_cols - 1)),
            dtype=dtypes[0])
    else:
        out_block = border_val * np.ones(
            (size_0 * n_rows + border_size * (n_rows - 1),
             size_1 * n_cols + border_size * (n_cols - 1),
             images[0].shape[2]), dtype=dtypes[0])
    for fld in range(len(images)):
        if rows_first:
            index_0 = int(np.floor(fld / n_cols))
            index_1 = fld - (n_cols * index_0)
        else:
            index_1 = int(np.floor(fld / n_rows))
            index_0 = fld - (n_rows * index_1)

        start0 = index_0 * (size_0 + border_size)
        end0 = start0 + size_0
        start1 = index_1 * (size_1 + border_size)
        end1 = start1 + size_1
        out_block[start0:end0, start1:end1] = images[fld]

    return out_block
