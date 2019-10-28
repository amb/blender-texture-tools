# -*- coding:utf-8 -*-

# This program is free software: you can redistribute it and/or modify
# it under the terms of the GNU General Public License as published by
# the Free Software Foundation, either version 3 of the License, or
# (at your option) any later version.
#
# This program is distributed in the hope that it will be useful,
# but WITHOUT ANY WARRANTY; without even the implied warranty of
# MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
# GNU General Public License for more details.
#
# You should have received a copy of the GNU General Public License
# along with this program.  If not, see <https://www.gnu.org/licenses/>.
#
# Created Date: Monday, July 8th 2019, 8:27:07 am
# Copyright: Tommi Hyppänen


bl_info = {
    "name": "Image Edit Operations",
    "category": "Paint",
    "description": "Various image processing filters and operations",
    "author": "Tommi Hyppänen (ambi)",
    "location": "Image Editor > Side Panel > Image",
    "documentation": "http://blenderartists.org/forum/"
    "showthread.php?364409-WIP-Seamless-texture-patching-addon",
    "version": (0, 1, 18),
    "blender": (2, 81, 0),
}

import numpy

np = numpy
import bpy
from . import image_ops
import importlib

importlib.reload(image_ops)


def convolution(ssp, intens, sfil):
    # source, intensity, convolution matrix
    tpx = numpy.zeros(ssp.shape, dtype=float)
    ysz, xsz = sfil.shape[0], sfil.shape[1]
    ystep = int(4 * ssp.shape[1])
    for y in range(ysz):
        for x in range(xsz):
            tpx += numpy.roll(ssp, (x - xsz // 2) * 4 + (y - ysz // 2) * ystep) * sfil[y, x]
    return tpx


def blur(pix, s, intensity):
    return convolution(pix, intensity, numpy.ones((1 + s * 2, 1 + s * 2), dtype=float))


def sharpen(pix, intensity):
    return convolution(pix, intensity, numpy.array([[-1, -1, -1], [-1, 9, -1], [-1, -1, -1]]))


def separate_values(pix, s, intensity):
    retarr = numpy.copy(pix)
    retarr[..., :3] = pix[..., :3] ** intensity
    return retarr


def grayscale(ssp):
    r, g, b = ssp[:, :, 0], ssp[:, :, 1], ssp[:, :, 2]
    gray = 0.2989 * r + 0.5870 * g + 0.1140 * b
    ssp[..., 0] = gray
    ssp[..., 1] = gray
    ssp[..., 2] = gray
    return ssp


def normalize(pix):
    t = pix - numpy.min(pix)
    return t / numpy.max(t)


def sobel_x(pix, intensity):
    gx = numpy.array([[-1, 0, 1], [-2, 0, 2], [-1, 0, 1]])
    return convolution(pix, intensity, gx)


def sobel_y(pix, intensity):
    gy = numpy.array([[1, 2, 1], [0, 0, 0], [-1, -2, -1]])
    return convolution(pix, intensity, gy)


def sobel(pix, s, intensity):
    retarr = numpy.zeros(pix.shape)
    retarr = sobel_x(pix, 1.0)
    retarr += sobel_y(pix, 1.0)
    retarr = (retarr * intensity) * 0.5 + 0.5
    retarr[..., 3] = pix[..., 3]
    return retarr


def edgedetect(pix, s, intensity):
    k = numpy.array([[0, 1, 0], [1, -4, 1], [0, 1, 0]])
    return convolution(pix, intensity, k) * 0.5 + 0.5


def gaussian(pix, s, intensity):
    sval = 1 + s * 2
    # TODO: actual gaussian, not just box
    krn = np.ones(sval) / sval
    f_krn = lambda m: np.convolve(m, krn, mode="same")
    pix = np.apply_along_axis(f_krn, axis=1, arr=pix)
    pix = np.apply_along_axis(f_krn, axis=0, arr=pix)
    return pix


def hi_pass(pix, s, intensity):
    bg = pix.copy()
    pix = (bg - gaussian(pix, s, 1.0)) * 0.5 + 0.5
    pix[:, :, 3] = bg[:, :, 3]
    return pix


def hgram_equalize(pix, intensity, atest):
    old = pix.copy()
    aw = np.argwhere(pix[..., 3] > atest)
    r = pix[..., 0][aw[:, 0], aw[:, 1]]
    g = pix[..., 1][aw[:, 0], aw[:, 1]]
    b = pix[..., 2][aw[:, 0], aw[:, 1]]
    pix[..., 0][aw[:, 0], aw[:, 1]] = np.sort(r).searchsorted(r)
    pix[..., 1][aw[:, 0], aw[:, 1]] = np.sort(g).searchsorted(g)
    pix[..., 2][aw[:, 0], aw[:, 1]] = np.sort(b).searchsorted(b)
    npm = np.max(pix[..., :3])
    print(npm)
    pix[..., :3] /= npm
    return old * (1.0 - intensity) + pix * intensity


def bilateral(img_in, sigma_s, sigma_v, eps=1e-8):
    # gaussian
    gsi = lambda r2, sigma: numpy.exp(-0.5 * r2 / sigma ** 2)
    win_width = int(np.ceil(3 * sigma_s))
    wgt_sum = numpy.ones(img_in.shape) * eps
    result = img_in * eps

    # TODO: mix the steps to remove artifacts
    for shft_x in range(-win_width, win_width + 1):
        shft_y = 0
        off = numpy.roll(img_in, [shft_y, shft_x], axis=[0, 1])
        w = gsi(shft_x ** 2 + shft_y ** 2, sigma_s)
        tw = w * gsi((off - img_in) ** 2, sigma_v)
        result += off * tw
        wgt_sum += tw

    for shft_y in range(-win_width, win_width + 1):
        shft_x = 0
        off = numpy.roll(img_in, [shft_y, shft_x], axis=[0, 1])
        w = gsi(shft_x ** 2 + shft_y ** 2, sigma_s)
        tw = w * gsi((off - img_in) ** 2, sigma_v)
        result += off * tw
        wgt_sum += tw

    # normalize the result and return
    return result / wgt_sum


def bilateral_filter(pix, s, intensity):
    # multiply by alpha
    pix[..., 0] *= pix[..., 3]
    pix[..., 1] *= pix[..., 3]
    pix[..., 2] *= pix[..., 3]

    print("R")
    # image, spatial, range
    pix[..., 0] = bilateral(pix[..., 0], s, intensity)
    print("G")
    pix[..., 1] = bilateral(pix[..., 1], s, intensity)
    print("B")
    pix[..., 2] = bilateral(pix[..., 2], s, intensity)

    return pix


def normals_simple(pix, s, intensity):
    pix = normalize(gaussian(grayscale(pix), s, 1.0))
    sshape = pix.shape

    # extract x and y deltas
    px = sobel_x(pix, 1.0)
    px[:, :, 2] = px[:, :, 2] * intensity
    px[:, :, 1] = 0
    px[:, :, 0] = 1

    py = sobel_y(pix, 1.0)
    py[:, :, 2] = py[:, :, 2] * intensity
    py[:, :, 1] = 1
    py[:, :, 0] = 0

    # find the imagined approximate surface normal
    arr = numpy.cross(px[:, :, :3], py[:, :, :3])

    # normalization: vec *= 1/len(vec)
    m = 1.0 / numpy.sqrt(arr[:, :, 0] ** 2 + arr[:, :, 1] ** 2 + arr[:, :, 2] ** 2)
    arr[..., 0] *= m
    arr[..., 1] *= m
    arr[..., 2] *= m
    vectors = arr

    # normals format
    retarr = numpy.zeros(sshape)
    retarr[:, :, 0] = 0.5 - vectors[:, :, 0]
    retarr[:, :, 1] = vectors[:, :, 1] + 0.5
    retarr[:, :, 2] = vectors[:, :, 2]
    retarr[:, :, 3] = 1.0
    return retarr


class Normals_IOP(image_ops.ImageOperatorGenerator):
    def generate(self):
        self.props["width"] = bpy.props.IntProperty(name="Width", min=0, default=2)
        self.props["intensity"] = bpy.props.FloatProperty(name="Intensity", min=0.0, default=1.0)
        self.prefix = "normals"
        self.info = "(Very rough estimate) normal map from RGB"
        self.category = "Filter"
        self.payload = lambda self, image, context: normals_simple(
            image, self.width, self.intensity
        )


class Grayscale_IOP(image_ops.ImageOperatorGenerator):
    def generate(self):
        self.prefix = "grayscale"
        self.info = "Grayscale from RGB"
        self.category = "Filter"
        self.payload = lambda self, image, context: grayscale(image)


class Sharpen_IOP(image_ops.ImageOperatorGenerator):
    def generate(self):
        self.props["intensity"] = bpy.props.FloatProperty(name="Intensity", min=0.0, default=1.0)
        self.prefix = "sharpen"
        self.info = "Simple sharpen"
        self.category = "Filter"
        self.payload = lambda self, image, context: sharpen(image, self.intensity)


class GaussianBlur_IOP(image_ops.ImageOperatorGenerator):
    def generate(self):
        self.props["width"] = bpy.props.IntProperty(name="Width", min=1, default=2)
        self.props["intensity"] = bpy.props.FloatProperty(name="Intensity", min=0.0, default=1.0)
        self.prefix = "gaussian_blur"
        self.info = "Does a Gaussian blur"
        self.category = "Filter"
        self.payload = lambda self, image, context: gaussian(image, self.width, self.intensity)


class HiPass_IOP(image_ops.ImageOperatorGenerator):
    def generate(self):
        self.props["width"] = bpy.props.IntProperty(name="Width", min=1, default=2)
        self.props["intensity"] = bpy.props.FloatProperty(name="Intensity", min=0.0, default=1.0)
        self.prefix = "high_pass"
        self.info = "High pass"
        self.category = "Filter"
        self.payload = lambda self, image, context: hi_pass(image, self.width, self.intensity)


class Bilateral_IOP(image_ops.ImageOperatorGenerator):
    def generate(self):
        self.props["sigma_a"] = bpy.props.FloatProperty(name="Sigma A", min=0.01, default=3.0)
        self.props["sigma_b"] = bpy.props.FloatProperty(name="Sigma B", min=0.01, default=0.1)
        self.prefix = "bilateral_filter"
        self.info = "Bilateral"
        self.category = "Filter"
        self.payload = lambda self, image, context: bilateral_filter(
            image, self.sigma_a, self.sigma_b
        )


class HistogramEQ_IOP(image_ops.ImageOperatorGenerator):
    def generate(self):
        self.props["intensity"] = bpy.props.FloatProperty(
            name="Intensity", min=0.0, max=1.0, default=1.0
        )
        self.prefix = "histogram_eq"
        self.info = "Histogram equalization"
        self.category = "Filter"
        self.payload = lambda self, image, context: hgram_equalize(image, self.intensity, 0.5)


class GimpSeamless_IOP(image_ops.ImageOperatorGenerator):
    """Image seamless generator operator"""

    # TODO: the smoothing is not complete, it goes only one way
    def generate(self):
        self.prefix = "gimp_seamless"
        self.info = "Gimp style seamless image operation"
        self.category = "Filter"

        def gimpify(self, image, context):
            pixels = numpy.copy(image)
            xs, ys = image.shape[1], image.shape[0]
            image = numpy.roll(image, xs * 2 + xs * 4 * (ys // 2))

            sxs = xs // 2
            sys = ys // 2

            # generate the mask
            imask = numpy.zeros((pixels.shape[0], pixels.shape[1]), dtype=float)
            for y in range(0, sys):
                zy0 = y / sys + 0.001
                zy1 = 1 - y / sys + 0.001
                for x in range(0, sxs):
                    zx0 = 1 - x / sxs + 0.001
                    imask[y, x] = 1 - zy0 / zx0
                    zx1 = x / sxs + 0.001
                    imask[y, x] = numpy.maximum((1 - zx1 / zy1), imask[y, x])

            imask[imask < 0] = 0

            # copy the data into the three remaining corners
            imask[0 : sys + 1, sxs : xs - 1] = numpy.fliplr(imask[0 : sys + 1, 0 : sxs - 1])
            imask[-sys:ys, 0:sxs] = numpy.flipud(imask[0:sys, 0:sxs])
            imask[-sys:ys, sxs : xs - 1] = numpy.flipud(imask[0:sys, sxs : xs - 1])
            imask[sys, :] = imask[sys - 1, :]  # center line

            # apply mask
            amask = numpy.zeros(pixels.shape, dtype=float)
            amask[:, :, 0] = imask
            amask[:, :, 1] = imask
            amask[:, :, 2] = imask
            amask[:, :, 3] = imask

            return amask * image + (numpy.ones(amask.shape) - amask) * pixels

        self.payload = gimpify


# class LaplacianBlend_IOP(image_ops.ImageOperatorGenerator):
#     def generate(self):
#         self.prefix = "laplacian_blend"
#         self.info = "Blends two images with Laplacian pyramids"
#         self.category = "Filter"

#         def _pl(self, image, context):
#             return image

#         self.payload = _pl


register, unregister = image_ops.create(locals())
