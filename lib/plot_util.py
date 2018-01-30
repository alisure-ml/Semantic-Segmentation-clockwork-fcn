import numpy as np
import matplotlib
import matplotlib.pyplot as plt

# random color map for segmentation
segment_color_map = matplotlib.colors.ListedColormap(np.random.rand(256, 3))


def segshow(im, label, out, n_cl=None):
    plt.figure()
    plt.subplot(1, 3, 1)
    plt.imshow(im)
    plt.axis('off')
    plt.tight_layout()
    plt.subplot(1, 3, 2)
    if n_cl:
        plt.imshow(label, cmap=segment_color_map, vmin=0, vmax=n_cl - 1)
    else:
        plt.imshow(label)
    plt.axis('off')
    plt.tight_layout()
    plt.subplot(1, 3, 3)
    if n_cl:
        plt.imshow(out, cmap=segment_color_map, vmin=0, vmax=n_cl - 1)
    else:
        plt.imshow(out)
    plt.axis('off')
    plt.tight_layout()
