import numpy as np
from time import time


class SIFT:
    """
    SIFT class to preprocess the data
    """

    def __init__(self, gs = 8, ps = 16, gaussian_thres = 1.0, gaussian_sigma = 0.8, sift_thres = 0.2, \
                 num_angles = 12, num_bins = 5, alpha = 9.0):
        self.num_angles = num_angles
        self.num_bins = num_bins
        self.alpha = alpha
        self.angle_list = np.array(range(num_angles))*2.0*np.pi/num_angles
        self.gs = gs
        self.ps = ps
        self.gaussian_thres = gaussian_thres
        self.gaussian_sigma = gaussian_sigma
        self.sift_thres = sift_thres
        self.weights = self._get_weights(num_bins)

    def get_params_image(self, image):
        image = image.astype(np.double)
        if image.ndim == 3:
            image = np.mean(image, axis=2)
        H, W = image.shape
        gS = self.gs
        pS = self.ps
        remH = np.mod(H-pS, gS)
        remW = np.mod(W-pS, gS)
        offsetH = remH//2
        offsetW = remW//2
        gridH, gridW = np.meshgrid(range(offsetH, H-pS+1, gS), range(offsetW, W-pS+1, gS))
        gridH = gridH.flatten()
        gridW = gridW.flatten()
        features = self._calculate_sift_grid(image, gridH, gridW)
        features = self._normalize_sift(features)
        positions = np.vstack((gridH / np.double(H), gridW / np.double(W)))
        return features, positions

    def get_X(self, data):
        out = []
        start = time()
        finish = len(data)
        for idx, dt in enumerate(data):
            # Helper.log_process('SIFT', idx, finish_cursor=finish, start_time = start)
            out.append(self.get_params_image(np.mean(np.double(dt), axis=2))[0][0])
        return np.array(out)



    def _get_weights(self, num_bins):
        size_unit = np.array(range(self.ps))
        sph, spw = np.meshgrid(size_unit, size_unit)
        sph.resize(sph.size)
        spw.resize(spw.size)
        bincenter = np.array(range(1, num_bins*2, 2)) / 2.0 / num_bins * self.ps - 0.5
        bincenter_h, bincenter_w = np.meshgrid(bincenter, bincenter)
        bincenter_h.resize((bincenter_h.size, 1))
        bincenter_w.resize((bincenter_w.size, 1))
        dist_ph = abs(sph - bincenter_h)
        dist_pw = abs(spw - bincenter_w)
        weights_h = dist_ph / (self.ps / np.double(num_bins))
        weights_w = dist_pw / (self.ps / np.double(num_bins))
        weights_h = (1-weights_h) * (weights_h <= 1)
        weights_w = (1-weights_w) * (weights_w <= 1)
        return weights_h * weights_w

    def _calculate_sift_grid(self, image, gridH, gridW):
        H, W = image.shape
        Npatches = gridH.size
        features = np.zeros((Npatches, self.num_bins * self.num_bins * self.num_angles))
        gaussian_height, gaussian_width = self._get_gauss_filter(self.gaussian_sigma)
        IH = self._convolution2D(image, gaussian_height)
        IW = self._convolution2D(image, gaussian_width)
        Imag = np.sqrt(IH**2 + IW**2)
        Itheta = np.arctan2(IH,IW)
        Iorient = np.zeros((self.num_angles, H, W))
        for i in range(self.num_angles):
            Iorient[i] = Imag * np.maximum(np.cos(Itheta - self.angle_list[i])**self.alpha, 0)
        for i in range(Npatches):
            currFeature = np.zeros((self.num_angles, self.num_bins**2))
            for j in range(self.num_angles):
                currFeature[j] = np.dot(self.weights,\
                        Iorient[j,gridH[i]:gridH[i]+self.ps, gridW[i]:gridW[i]+self.ps].flatten())
            features[i] = currFeature.flatten()
        return features

    def _normalize_sift(self, features):
        siftlen = np.sqrt(np.sum(features**2, axis=1))
        hcontrast = (siftlen >= self.gaussian_thres)
        siftlen[siftlen < self.gaussian_thres] = self.gaussian_thres
        features /= siftlen.reshape((siftlen.size, 1))
        features[features>self.sift_thres] = self.sift_thres
        features[hcontrast] /= np.sqrt(np.sum(features[hcontrast]**2, axis=1)).\
                reshape((features[hcontrast].shape[0], 1))
        return features


    def _get_gauss_filter(self, sigma):
        gaussian_filter_amp = int(2*np.ceil(sigma))
        gaussian_filter = np.array(range(-gaussian_filter_amp, gaussian_filter_amp+1))**2
        gaussian_filter = gaussian_filter[:, np.newaxis] + gaussian_filter
        gaussian_filter = np.exp(- gaussian_filter / (2.0 * sigma**2))
        gaussian_filter /= np.sum(gaussian_filter)
        gaussian_height, gaussian_width = np.gradient(gaussian_filter)
        gaussian_height *= 2.0/np.sum(np.abs(gaussian_height))
        gaussian_width  *= 2.0/np.sum(np.abs(gaussian_width))
        return gaussian_height, gaussian_width

    def _convolution2D(self, image, kernel):
        imRows, imCols = image.shape
        kRows, kCols = kernel.shape

        y = np.zeros((imRows,imCols))

        kcenterX = kCols//2
        kcenterY = kRows//2

        for i in range(imRows):
            for j in range(imCols):
                for m in range(kRows):
                    mm = kRows - 1 - m
                    for n in range(kCols):
                        nn = kCols - 1 - n

                        ii = i + (m - kcenterY)
                        jj = j + (n - kcenterX)

                        if ii >= 0 and ii < imRows and jj >= 0 and jj < imCols :
                            y[i][j] += image[ii][jj] * kernel[mm][nn]

        return y
    

class ImageTransformation:
    """
    Class to do Image Transformation
    """

    @staticmethod
    def flip_image_horizontal(image):
        result = image.copy()
        for channel in range(3):
            aux = image[:, :, channel]
            for column in range(len(aux)):
                result[:, column, channel] = aux[:, len(aux) - column - 1]
        return result