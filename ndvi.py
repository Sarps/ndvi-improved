import cv2
import numpy as np


def disp_multiple(im1=None, im2=None, im3=None):
    height, width, _ = im1.shape

    combined = np.zeros((height, 3 * width, 3), dtype=np.uint8)

    combined[:, 0:width, :] = im1
    combined[:, width:width*2, :] = cv2.cvtColor(im2, cv2.COLOR_GRAY2RGB)
    combined[:, width*2:, :] = cv2.cvtColor(im3, cv2.COLOR_GRAY2RGB)

    return combined


def label(image, text):
    return cv2.putText(image, text, (0, 50), cv2.FONT_HERSHEY_SIMPLEX, 2, 255)


def contrast_stretch(im):
    in_min = np.percentile(im, 5)
    in_max = np.percentile(im, 95)

    out_min = 0.0
    out_max = 255.0

    out = im - in_min
    out *= ((out_min - out_max) / (in_min - in_max))
    out += in_min

    return out

if __name__ == '__main__':

    cv2.namedWindow('image', cv2.WINDOW_NORMAL)

    image = cv2.imread('assets/leaf.jpg')
    b, _, r = cv2.split(image)

    bottom = (r.astype(float) + b.astype(float))
    bottom[bottom == 0] = 0.01

    ndvi = (r.astype(float) - b) / bottom
    ndvi = contrast_stretch(ndvi)
    ndvi = ndvi.astype(np.uint8)

    print(image.shape)

    label(image, 'original')
    label(r, 'NIR')
    label(ndvi, 'NDVI')

    combined = disp_multiple(image, r, ndvi)

    cv2.imshow('image', combined)

    cv2.waitKey(0)
    cv2.destroyAllWindows()
