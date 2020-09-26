import numpy as np
import matplotlib.pyplot as plt

from skimage import transform as tf
from skimage import io

# updated camera position
DEFAULT_TRANSFORM = tf.ProjectiveTransform(matrix=np.array(
    [[ 6.90367250e-01, -4.40337497e-01,  3.93231306e+02],
     [ 2.42728954e-03,  5.05822868e-01,  1.74216401e+02],
     [ 3.39006919e-06, -6.85871534e-04,  1.24440287e+00]]))

# exclude notches
# DEFAULT_TRANSFORM = tf.ProjectiveTransform(matrix=np.array(
#     [[ 7.08657055e-01, -4.28303794e-01,  3.49380531e+02],
#      [ 2.03726067e-16,  5.16306357e-01,  1.73451327e+02],
#      [ 9.32158229e-20, -6.65565608e-04,  1.23893805e+00]]))

# DEFAULT_TRANSFORM = tf.ProjectiveTransform(matrix=np.array(
#     [[ 7.28598424e-01, -4.27356693e-01,  3.37246964e+02],
#      [-2.01770438e-16,  5.49445716e-01,  1.51265182e+02],
#      [ 0.00000000e+00, -6.68185355e-04,  1.23987854e+00]]))


def estimate_transform():
    src = np.array([[0, 0], [0, 719], [1279, 719], [1279, 0]])
    # dst = np.array([[272, 122], [40, 718], [1265, 718], [1023, 122]])
    # dst = np.array([[282, 140], [55, 715], [1245, 715], [1013, 140]])
    dst = np.array([[316, 140], [102, 716], [1270, 716], [1022, 142]])

    tform3 = tf.ProjectiveTransform()
    tform3.estimate(src, dst)
    # print(tform3)
    return tform3, dst


def apply_transform(image, tform=DEFAULT_TRANSFORM, coords=False, inverse=False):
    if not coords:
        if inverse:
            tform = tform.inverse
        warped = tf.warp(image, tform, output_shape=image.shape, preserve_range=True)
        if np.max(image) > 1:
            warped = warped.astype(np.uint8)
    else:
        xmin, ymin, xmax, ymax = image[0][0], image[0][1], image[1][0], image[1][1]
        # print(xmin, ymin, xmax, ymax)
        h = ymax-ymin
        w = xmax-xmin
        coords = [[xmin, ymin], [xmin+w, ymin], [xmin, ymin+h], [xmin+w, ymin+h]]
        # print(coords)
        if inverse:
            coords = tform(coords)
        else:
            coords = tform.inverse(coords)
        # print(coords)
        # print(warped)
        ymin = np.floor(np.min(coords[:,1]))
        ymax = np.ceil(np.max(coords[:,1]))
        xmin = np.floor((coords[0,0] + coords[2,0])/2)
        xmax = np.floor((coords[1,0] + coords[3,0])/2)
        warped = np.array([[xmin, ymin], [xmax, ymax]], dtype=int)
        # print(warped)
    return warped


if __name__ == '__main__':
    from file import get_working_directory
    image = io.imread(get_working_directory()+"/test/1588359112 test-orig.png")
    tform, dst = estimate_transform()
    print(tform)
    warped = apply_transform(image, tform)
    print(np.max(warped))
    fig, ax = plt.subplots(ncols=2, figsize=(8, 3))

    ax[0].imshow(image, cmap=plt.cm.gray)
    ax[0].plot(dst[:, 0], dst[:, 1], '.r')
    ax[1].imshow(warped, cmap=plt.cm.gray)

    io.imsave(get_working_directory()+"/test/tformed.png", warped)

    for a in ax:
        a.axis('image')
        a.set_xticks([])
        a.set_yticks([])

    plt.tight_layout()
    # plt.show()
    plt.savefig(get_working_directory()+'/test/transform.png', dpi=500)
