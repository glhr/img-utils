import matplotlib.pyplot as plt
import numpy as np
# from skimage.draw import rectangle_perimeter
from vision_utils.img import *
from skimage.color import gray2rgb, rgb2hsv
from vision_utils.contours import Object, get_contours, filter_objects
import skimage
import cv2

PATH_TO_CONTOURS_IMGS = '../color_classifier/contours/'


def plot_bounding_boxes(image,
                        objects,
                        classes=None,
                        desired=None,
                        show=False,
                        plot_boxes=False,
                        numbering=False,
                        correct_indexes = [],
                        ignore_indexes = []):
    n_objects = len(objects)
    # fig = plt.figure()

    for i in range(n_objects):
        box = objects[i].get_flattened_coords()

        # rr, cc = rectangle_perimeter(start=(box[1],box[0]), end=(box[3],box[2]), shape=image[:,:,0].shape)
        # image[rr, cc, :] = 1  #set color white
        if desired is None or classes is None or not len(classes) or classes[i] != desired:
            color = WHITE
        else:
            color = RED

        if color == RED:
            cv2.rectangle(image, (box[0],box[1]), (box[2],box[3]), color, 3)
        elif plot_boxes==True or i in correct_indexes:
            cv2.rectangle(image, (box[0],box[1]), (box[2],box[3]), BLACK, 2)
        if numbering:
            cv2.putText(image, str(i),
                        (textX-10, textY-10),
                        font,
                        fontScale=1,
                        color=color,
                        thickness=1,
                        lineType=cv2.LINE_AA)

        if classes is not None and len(classes) and not (i in ignore_indexes):
            width = box[2]-box[0]
            height = box[3]-box[1]
            # get boundary of this text
            font = cv2.FONT_HERSHEY_SIMPLEX
            textsize = cv2.getTextSize(classes[i], font, 1, 2)[0]
            # get coords based on boundary
            textX = box[0] + int((width - textsize[0]) / 2)
            textY = box[1] + int((height + textsize[1]) / 2)
            cv2.putText(image, classes[i],
                        (textX, textY),
                        font,
                        fontScale=1,
                        color=color,
                        thickness=2,
                        lineType=cv2.LINE_AA)

    if show:
        plt.imshow(image.astype(int))
        a = plt.gca()
        a.axis('image')
        a.set_xticks([])
        a.set_yticks([])
        plt.tight_layout()
        # plt.savefig(get_working_directory()+'/test/labelled_objects.png', dpi=500)
        # plt.close()

        plt.draw()
        plt.pause(0.001)

    return image



def save_contours():
    for image_filename in glob.glob(PATH_TO_CONTOURS_IMGS+"/*"):
        print(image_filename)
        masked = io.imread(image_filename)

        image = io.imread('dataset/'+get_filename_from_path(image_filename))
        image = normalize_img(image)
        fig, ax = plt.subplots(ncols=2, figsize=(8, 3))
        ax[0].imshow(image)
        ax[1].imshow(masked)
        for a in ax:
            a.axis('image')
            a.set_xticks([])
            a.set_yticks([])

        plt.tight_layout()
        plt.savefig('plots_contours/{}.png'.format(get_filename_from_path(image_filename, extension=False)), dpi=300)



def get_image_axis(ax, image):
    ax.imshow(image)
    ax.axis('image')
    ax.set_xticks([])
    ax.set_yticks([])
    return ax


def plot_hsv_histograms(imagepath):
    image_orig = skimage.io.imread(imagepath)

    # image_tf = apply_transform(image_orig)  # geometric transform
    # image_tf = skimage.transform.resize(image_orig, output_shape=(500,500))
    image_tf = normalize_img(image_orig)  # make sure it's RGB & in range (0,1)

    image_value = rgb2hsv(image_tf)[:,:,1]
    contours = get_contours(image_value, level=0.4)
    objects = filter_objects([Object(contour, image_tf) for contour in contours])
    # object = objects[0]
    nrows = 4
    ncols = len(objects)
    print(ncols, "objects found")
    fig, ax = plt.subplots(nrows=nrows, ncols=ncols, figsize=(2*ncols, 5), gridspec_kw={'height_ratios': [3, 1, 1, 1]})
    y_titles = ['Hue', 'Saturation', 'Value']

    for col, object in enumerate(objects):
        mask = object.get_mask(type=bool)
        histo = get_histos(image_tf, mask, bins=1)

        img_combined = np.vstack((object.get_crop(binary=False), object.get_crop(binary=True, range=1)))

        ax[0][col] = get_image_axis(ax[0][col], img_combined)

        for i, channel in enumerate(histo):
            bins = histo[i][1][:-1]
            values = histo[i][0]
            ax[i+1][col].bar(x=bins, height=values, align='edge', width=0.1)
            # ax[i+1][col].set_ylim(0, 10)
            if col == 0:
                ax[i+1][col].set_ylabel(y_titles[i])
    # plt.tight_layout()
    plt.show()


if __name__ == '__main__':
    # plot_hsv_histograms('test/mask.png')
    plot_hsv_histograms('test/tformed.png')
