import numpy as np
import matplotlib.pyplot as plt
import skimage  # .io.imread as imread


def img_proc(file):
    image_data = skimage.io.imread(file).astype(np.float32)
    print("Shape: {}".format(image_data.shape))
    print("Size: {}".format(image_data.size))

    scaled_image_data = image_data # / 255.
    plt.imshow(scaled_image_data)
    plt.show()
    print(scaled_image_data)


def img_proc_rgb(file):
    image_data = skimage.io.imread(file).astype(np.float32)
    print("Shape: {}".format(image_data.shape))
    print("Size: {}".format(image_data.size))

    scaled_image_data = image_data / 255.
    plt.imshow(scaled_image_data)
    plt.show()

    # image_slice_red = scaled_image_data[:, :, 0]
    # image_slice_green = scaled_image_data[:, :, 1]
    # image_slice_blue = scaled_image_data[:, :, 2]
    image_slice_red = scaled_image_data.copy()
    image_slice_red[:,:,1] = 0
    image_slice_red[:, :, 2] = 0
    image_slice_green = scaled_image_data.copy()
    image_slice_green[:,:,0] = 0
    image_slice_green[:, :, 2] = 0
    image_slice_blue = scaled_image_data.copy()
    image_slice_blue[:,:,0] = 0
    image_slice_blue[:, :, 1] = 0

    plt.subplot(221)
    plt.imshow(image_slice_red)
    plt.subplot(222)
    plt.imshow(image_slice_green)
    plt.subplot(223)
    plt.imshow(image_slice_blue)
    plt.subplot(224)
    # scaled_image_data_ng = scaled_image_data[:,:,:] = scaled_image_data[:,:,:] * -1
    plt.imshow(scaled_image_data)
    plt.show()
    exit()

