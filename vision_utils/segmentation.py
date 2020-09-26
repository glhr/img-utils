from skimage import io
from skimage.transform import resize
from skimage.color import rgb2hsv, gray2rgb, rgb2gray, label2rgb
from skimage.segmentation import slic
import numpy as np
import matplotlib.pyplot as plt
from skimage.exposure import equalize_adapthist

from vision_utils.img import rgb2hsv, rgb2gray, rgb2ycbcr, normalize_img


def get_segmentation_mask(image):
    image_slic = slic(image,
                      n_segments=3,
                      sigma=5,
                      compactness=0.1,
                      enforce_connectivity=False)
    image = label2rgb(image_slic, image, kind='avg')
    mask = np.ma.masked_equal(image, np.min(image))
    return mask


if __name__ == '__main__':
    image = io.imread("test/tformed.png")
    image = equalize_adapthist(image, nbins=5)
    image = normalize_img(image)

    mask = get_segmentation_mask(image)

    fig, ax = plt.subplots(ncols=2, figsize=(8, 3))

    ax[0].imshow(image, cmap=plt.cm.gray)
    ax[1].imshow(mask, cmap=plt.cm.gray)

    for a in ax:
        a.axis('image')
        a.set_xticks([])
        a.set_yticks([])

    plt.show()
