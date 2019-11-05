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
    "documentation": "https://blenderartists.org/t/seamless-texture-patching-and-filtering-addon",
    "version": (0, 1, 22),
    "blender": (2, 81, 0),
}

import numpy

np = numpy
import bpy
from . import image_ops
import importlib

importlib.reload(image_ops)


def gauss_curve(x):
    # gaussian with 0.01831 at last
    res = np.array([np.exp(-((i * (2 / x)) ** 2)) for i in range(-x, x + 1)])
    res /= np.sum(res)
    return res


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


def sobel(pix, intensity):
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
    s = int(s)
    sa = pix[..., 3]
    # sval = 1 + s * 2
    # krn = np.ones(sval) / sval
    krn = gauss_curve(s)
    f_krn = lambda m: np.convolve(m, krn, mode="same")
    pix = np.apply_along_axis(f_krn, axis=1, arr=pix)
    pix = np.apply_along_axis(f_krn, axis=0, arr=pix)
    pix[..., 3] = sa
    return pix


def hi_pass(pix, s, intensity):
    bg = pix.copy()
    pix = (bg - gaussian(pix, s, 1.0)) * 0.5 + 0.5
    pix[:, :, 3] = bg[:, :, 3]
    return pix


def gaussian_repeat(pix, s):
    res = np.zeros_like(pix)
    gcr = gauss_curve(s)
    for i in range(-s, s + 1):
        if i != 0:
            res[:-i, ...] += pix[i:, ...] * gcr[i + s]
            res[-i:, ...] += pix[:i, ...] * gcr[i + s]
        else:
            res += pix * gcr[s]
    pix2 = res.copy()
    res *= 0.0
    for i in range(-s, s + 1):
        if i != 0:
            res[:, :-i, :] += pix2[:, i:, :] * gcr[i + s]
            res[:, -i:, :] += pix2[:, :i, :] * gcr[i + s]
        else:
            res += pix2 * gcr[s]
    return res


def gaussian_repeat_fit(pix, s):
    rf = s

    pix[0, :] = (pix[0, :] + pix[-1, :]) * 0.5
    pix[-1, :] = pix[0, :]
    for i in range(1, rf):
        factor = ((rf - i)) / rf
        pix[i, :] = pix[0, :] * factor + pix[i, :] * (1 - factor)
        pix[-i, :] = pix[0, :] * factor + pix[-i, :] * (1 - factor)

    pix[:, 0] = (pix[:, 0] + pix[:, -1]) * 0.5
    pix[:, -1] = pix[:, 0]
    for i in range(1, rf):
        factor = ((rf - i)) / rf
        pix[:, i] = pix[:, 0] * factor + pix[:, i] * (1 - factor)
        pix[:, -i] = pix[:, 0] * factor + pix[:, -i] * (1 - factor)

    return gaussian_repeat(pix, s)


# https://stackoverflow.com/questions/32655686/histogram-matching-of-two-images-in-python-2-x
def hist_match(source, template):
    """
    Adjust the pixel values of a grayscale image such that its histogram
    matches that of a target image

    Arguments:
    -----------
        source: np.ndarray
            Image to transform; the histogram is computed over the flattened
            array
        template: np.ndarray
            Template image; can have different dimensions to source
    Returns:
    -----------
        matched: np.ndarray
            The transformed output image
    """

    oldshape = source.shape
    source = source.ravel()
    template = template.ravel()

    # get the set of unique pixel values and their corresponding indices and
    # counts
    s_values, bin_idx, s_counts = np.unique(source, return_inverse=True, return_counts=True)
    t_values, t_counts = np.unique(template, return_counts=True)

    # take the cumsum of the counts and normalize by the number of pixels to
    # get the empirical cumulative distribution functions for the source and
    # template images (maps pixel value --> quantile)
    s_quantiles = np.cumsum(s_counts).astype(np.float64)
    s_quantiles /= s_quantiles[-1]
    t_quantiles = np.cumsum(t_counts).astype(np.float64)
    t_quantiles /= t_quantiles[-1]

    # interpolate linearly to find the pixel values in the template image
    # that correspond most closely to the quantiles in the source image
    interp_t_values = np.interp(s_quantiles, t_quantiles, t_values)

    return interp_t_values[bin_idx].reshape(oldshape)


def gaussianize(source, NG=1000):
    oldshape = source.shape
    output = source.copy()
    transforms = []

    t_values = np.arange(NG * 8 + 1) / (NG * 8)
    t_counts = gauss_curve(NG * 4)
    t_quantiles = np.cumsum(t_counts).astype(np.float64)

    t_max = 0.0
    for i in range(3):
        s_values, bin_idx, s_counts = np.lib.arraysetops.unique(
            source[..., i].ravel(), return_inverse=True, return_counts=True
        )

        s_quantiles = np.cumsum(s_counts).astype(np.float64)
        s_quantiles /= s_quantiles[-1]
        s_max = s_quantiles[-1]
        if s_max > t_max:
            t_max = s_max
        transforms.append([s_values, s_quantiles, s_max])

        tv = np.interp(s_quantiles, t_quantiles, t_values)[bin_idx]
        output[..., i] = tv.reshape(oldshape[:2])

    return output, transforms


def degaussianize(source, transforms):
    oldshape = source.shape
    output = source.copy()

    for i in range(3):
        s_values, bin_idx, s_counts = np.lib.arraysetops.unique(
            output[..., i].ravel(), return_inverse=True, return_counts=True
        )
        t_values, t_quantiles, _ = transforms[i]

        s_quantiles = np.cumsum(s_counts).astype(np.float64)
        s_quantiles /= s_quantiles[-1]
        # t_quantiles = np.cumsum(t_counts).astype(np.float64)
        # t_quantiles /= t_quantiles[-1]

        tv = np.interp(s_quantiles, t_quantiles, t_values)[bin_idx]
        output[..., i] = tv.reshape(oldshape[:2])

    return output


def cumulative_distribution(data, bins):
    assert np.min(data) >= 0.0 and np.max(data) <= 1.0
    hg_av, hg_a = np.unique(np.floor(data * (bins - 1)), return_index=True)
    hg_a = np.float32(hg_a)
    hgs = np.sum(hg_a)
    hg_a /= hgs
    res = np.zeros((bins,))
    res[np.int64(hg_av)] = hg_a
    return np.cumsum(res)


def ecdf(x):
    """ empirical CDF """
    vals, counts = np.unique(x, return_counts=True)
    ecdf = np.cumsum(counts).astype(np.float64)
    ecdf /= ecdf[-1]
    return vals, ecdf


def hi_pass_balance(pix, s, zoom):
    bg = pix.copy()

    yzm = pix.shape[0] // 2
    xzm = pix.shape[1] // 2

    yzoom = zoom if zoom < yzm else yzm
    xzoom = zoom if zoom < xzm else xzm

    pixmin = np.min(pix)
    pixmax = np.max(pix)
    med = (pixmin + pixmax) / 2
    gas = gaussian_repeat(pix - med, s) + med
    pix = (pix - gas) * 0.5 + 0.5
    for c in range(3):
        pix[..., c] = hist_match(
            pix[..., c], bg[yzm - yzoom : yzm + yzoom, xzm - xzoom : xzm + xzoom, c]
        )
    pix[..., 3] = bg[..., 3]
    return pix


def hgram_equalize(pix, intensity, atest):
    old = pix.copy()
    aw = np.argwhere(pix[..., 3] > atest)
    aws = (aw[:, 0], aw[:, 1])
    for c in range(3):
        t = pix[..., c][aws]
        pix[..., c][aws] = np.sort(t).searchsorted(t)
    pix[..., :3] /= np.max(pix[..., :3])
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


def bilateral_filter(pix, s, intensity, source):
    # multiply by alpha
    pix[..., 0] *= pix[..., 3]
    pix[..., 1] *= pix[..., 3]
    pix[..., 2] *= pix[..., 3]

    if source == "SOBEL":
        sb = sobel(pix, 1.0)
    else:
        sb = pix

    print("R")
    # image, spatial, range
    pix[..., 0] = bilateral(sb[..., 0], s, intensity)
    print("G")
    pix[..., 1] = bilateral(sb[..., 1], s, intensity)
    print("B")
    pix[..., 2] = bilateral(sb[..., 2], s, intensity)

    return pix


def normals_simple(pix, s, intensity, source):
    if s > 0:
        pix = gaussian_repeat(pix, s)

    if source == "SOBEL":
        pix = sobel(pix, 1.0)
    else:
        pix = grayscale(pix)

    pix = normalize(pix)
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


def normals_to_curvature(pix, intensity):
    curve = np.zeros((pix.shape[0], pix.shape[1]), dtype=np.float32)
    vectors = np.zeros((pix.shape[0], pix.shape[1], 3), dtype=np.float32)

    vectors[..., 0] = 0.5 - pix[..., 0]
    vectors[..., 1] = pix[..., 1] - 0.5
    vectors[..., 2] = pix[..., 2]

    y_vec = np.array([1, 0, 0], dtype=np.float32)
    x_vec = np.array([0, 1, 0], dtype=np.float32)

    yd = vectors.dot(x_vec)
    xd = vectors.dot(y_vec)

    # curve[0,0] = yd[1,0]
    curve[:-1, :] += yd[1:, :]
    curve[-1, :] += yd[0, :]

    # curve[0,0] = yd[-1,0]
    curve[1:, :] -= yd[:-1, :]
    curve[0, :] -= yd[-1, :]

    # curve[0,0] = xd[1,0]
    curve[:, :-1] -= xd[:, 1:]
    curve[:, -1] -= xd[:, 0]

    # curve[0,0] = xd[-1,0]
    curve[:, 1:] += xd[:, :-1]
    curve[:, 0] += xd[:, -1]

    curve = curve * intensity + 0.5

    pix[..., 0] = curve
    pix[..., 1] = curve
    pix[..., 2] = curve
    return pix


def dog(pix, a, b, mp):
    pixb = pix.copy()
    pix[..., :3] = np.abs(gaussian_repeat(pix, a) - gaussian_repeat(pixb, b))[..., :3]
    pix[pix < mp][..., :3] = 0.0
    return pix


def gimpify(image):
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
    imask[0 : sys + 1, sxs:xs] = numpy.fliplr(imask[0 : sys + 1, 0:sxs])
    imask[-sys:ys, 0:sxs] = numpy.flipud(imask[0:sys, 0:sxs])
    imask[-sys:ys, sxs:xs] = numpy.flipud(imask[0:sys, sxs:xs])
    imask[sys, :] = imask[sys - 1, :]  # center line

    # apply mask
    amask = numpy.empty(pixels.shape, dtype=float)
    amask[:, :, 0] = imask
    amask[:, :, 1] = imask
    amask[:, :, 2] = imask
    amask[:, :, 3] = imask

    return amask * image + (1.0 - amask) * pixels


class Grayscale_IOP(image_ops.ImageOperatorGenerator):
    def generate(self):
        self.prefix = "grayscale"
        self.info = "Grayscale from RGB"
        self.category = "Basic"
        self.payload = lambda self, image, context: grayscale(image)


class Swizzle_IOP(image_ops.ImageOperatorGenerator):
    def generate(self):
        self.props["order_a"] = bpy.props.StringProperty(name="Order A", default="RGBA")
        self.props["order_b"] = bpy.props.StringProperty(name="Order B", default="RBGa")
        self.props["direction"] = bpy.props.EnumProperty(
            name="Direction", items=[("ATOB", "A to B", "", 1), ("BTOA", "B to A", "", 2)]
        )
        self.prefix = "swizzle"
        self.info = "Channel swizzle"
        self.category = "Basic"

        def _pl(self, image, context):
            test_a = self.order_a.upper()
            test_b = self.order_b.upper()

            if len(test_a) != 4 or len(test_b) != 4:
                self.report({"INFO"}, "Swizzle channel count must be 4")
                return image

            if set(test_a) != set(test_b):
                self.report({"INFO"}, "Swizzle channels must have same names")
                return image

            first = self.order_a
            second = self.order_b

            if self.direction == "BTOA":
                first, second = second, first

            temp = image.copy()

            for i in range(4):
                fl = first[i].upper()
                t = second.upper().index(fl)
                if second[t] != first[i]:
                    temp[..., t] = 1.0 - image[..., i]
                else:
                    temp[..., t] = image[..., i]

            return temp

        self.payload = _pl


class Normalize_IOP(image_ops.ImageOperatorGenerator):
    def generate(self):
        self.prefix = "normalize"
        self.info = "Normalize"
        self.category = "Basic"

        def _pl(self, image, context):
            tmp = image[..., 3]
            res = normalize(image)
            res[..., 3] = tmp
            return res

        self.payload = _pl


class Sharpen_IOP(image_ops.ImageOperatorGenerator):
    def generate(self):
        self.props["intensity"] = bpy.props.FloatProperty(name="Intensity", min=0.0, default=1.0)
        self.prefix = "sharpen"
        self.info = "Simple sharpen"
        self.category = "Basic"
        self.payload = lambda self, image, context: sharpen(image, self.intensity)


class GaussianBlur_IOP(image_ops.ImageOperatorGenerator):
    def generate(self):
        self.props["width"] = bpy.props.IntProperty(name="Width", min=1, default=2)
        self.props["intensity"] = bpy.props.FloatProperty(name="Intensity", min=0.0, default=1.0)
        self.prefix = "gaussian_blur"
        self.info = "Does a Gaussian blur"
        self.category = "Basic"
        self.payload = lambda self, image, context: gaussian(image, self.width, self.intensity)


class Bilateral_IOP(image_ops.ImageOperatorGenerator):
    def generate(self):
        self.props["source"] = bpy.props.EnumProperty(
            name="Source", items=[("LUMINANCE", "Luminance", "", 1), ("SOBEL", "Sobel", "", 2)]
        )
        self.props["sigma_a"] = bpy.props.FloatProperty(name="Sigma A", min=0.01, default=3.0)
        self.props["sigma_b"] = bpy.props.FloatProperty(name="Sigma B", min=0.01, default=0.1)
        self.prefix = "bilateral_filter"
        self.info = "Bilateral"
        self.category = "Basic"
        self.payload = lambda self, image, context: bilateral_filter(
            image, self.sigma_a, self.sigma_b, self.source
        )


class HiPass_IOP(image_ops.ImageOperatorGenerator):
    def generate(self):
        self.props["width"] = bpy.props.IntProperty(name="Width", min=1, default=2)
        self.props["intensity"] = bpy.props.FloatProperty(name="Intensity", min=0.0, default=1.0)
        self.prefix = "high_pass"
        self.info = "High pass"
        self.category = "Basic"
        self.payload = lambda self, image, context: hi_pass(image, self.width, self.intensity)


class HiPassBalance_IOP(image_ops.ImageOperatorGenerator):
    def generate(self):
        self.props["width"] = bpy.props.IntProperty(name="Width", min=1, default=2)
        self.props["zoom"] = bpy.props.IntProperty(name="Center slice", min=5, default=1000)
        # self.props["intensity"] = bpy.props.FloatProperty(name="Intensity", min=0.0, default=1.0)
        self.prefix = "hipass_balance"
        self.info = "Remove low frequencies from the image"
        self.category = "Balance"
        self.payload = lambda self, image, context: hi_pass_balance(image, self.width, self.zoom)


class ContrastBalance_IOP(image_ops.ImageOperatorGenerator):
    def generate(self):
        self.prefix = "contrast_balance"
        self.info = "Balance contrast"
        self.category = "Balance"

        self.props["gA"] = bpy.props.IntProperty(name="Range", min=1, max=256, default=20)
        self.props["gB"] = bpy.props.IntProperty(name="Error", min=1, max=256, default=40)
        self.props["strength"] = bpy.props.FloatProperty(name="Strength", min=0.0, default=1.0)

        def _pl(self, image, context):
            tmp = image.copy()

            # squared error
            gcr = gaussian_repeat(tmp, self.gA)
            error = (tmp - gcr) ** 2
            mask = -gaussian_repeat(error, self.gB)
            mask -= np.min(mask)
            mask /= np.max(mask)
            mask = (mask - 0.5) * self.strength + 1.0
            res = gcr + mask * (tmp - gcr)

            res[..., 3] = tmp[..., 3]
            return res

        self.payload = _pl


class HistogramEQ_IOP(image_ops.ImageOperatorGenerator):
    def generate(self):
        self.props["intensity"] = bpy.props.FloatProperty(
            name="Intensity", min=0.0, max=1.0, default=1.0
        )
        self.prefix = "histogram_eq"
        self.info = "Histogram equalization"
        self.category = "Advanced"
        self.payload = lambda self, image, context: hgram_equalize(image, self.intensity, 0.5)


class Gaussianize_IOP(image_ops.ImageOperatorGenerator):
    def generate(self):
        self.props["count"] = bpy.props.IntProperty(name="Count", min=10, max=100000, default=1000)
        self.prefix = "gaussianize"
        self.info = "Gaussianize histogram"
        self.category = "Advanced"
        self.payload = lambda self, image, context: gaussianize(image, NG=self.count)[0]


class GimpSeamless_IOP(image_ops.ImageOperatorGenerator):
    """Image seamless generator operator"""

    # TODO: the smoothing is not complete, it goes only one way
    def generate(self):
        self.prefix = "gimp_seamless"
        self.info = "Gimp style seamless image operation"
        self.category = "Advanced"
        self.payload = lambda self, image, context: gimpify(image)


class HistogramSeamless_IOP(image_ops.ImageOperatorGenerator):
    def generate(self):
        self.prefix = "histogram_seamless"
        self.info = "Seamless histogram blending"
        self.category = "Advanced"

        def _pl(self, image, context):
            gimg, transforms = gaussianize(image)
            blended = gimpify(gimg)
            return degaussianize(blended, transforms)

        self.payload = _pl


class Normals_IOP(image_ops.ImageOperatorGenerator):
    def generate(self):
        self.props["source"] = bpy.props.EnumProperty(
            name="Source", items=[("LUMINANCE", "Luminance", "", 1), ("SOBEL", "Sobel", "", 2)]
        )
        self.props["width"] = bpy.props.IntProperty(name="Width", min=0, default=2)
        self.props["intensity"] = bpy.props.FloatProperty(name="Intensity", min=0.0, default=1.0)
        self.prefix = "height_to_normals"
        self.info = "(Very rough estimate) normal map from RGB"
        self.category = "Normals"
        self.payload = lambda self, image, context: normals_simple(
            image, self.width, self.intensity, self.source
        )


class NormalsToCurvature_IOP(image_ops.ImageOperatorGenerator):
    def generate(self):
        # self.props["width"] = bpy.props.IntProperty(name="Width", min=0, default=2)
        self.props["intensity"] = bpy.props.FloatProperty(name="Intensity", min=0.0, default=1.0)
        self.prefix = "normals_to_curvature"
        self.info = "Curvature map from tangent normal map"
        self.category = "Normals"
        self.payload = lambda self, image, context: normals_to_curvature(image, self.intensity)


# class DoG_IOP(image_ops.ImageOperatorGenerator):
#     def generate(self):
#         self.props["a"] = bpy.props.IntProperty(name="Width A", min=1, default=20)
#         self.props["b"] = bpy.props.IntProperty(name="Width B", min=1, default=100)
#         self.props["mp"] = bpy.props.FloatProperty(name="Treshold", min=0.0, default=1.0)
#         self.prefix = "dog"
#         self.info = "Difference of gaussians"
#         self.category = "Filter"
#         self.payload = lambda self, image, context: dog(image, self.a, self.b, self.mp)

# class LaplacianBlend_IOP(image_ops.ImageOperatorGenerator):
#     def generate(self):
#         self.prefix = "laplacian_blend"
#         self.info = "Blends two images with Laplacian pyramids"
#         self.category = "Filter"

#         def _pl(self, image, context):
#             return image

#         self.payload = _pl


register, unregister = image_ops.create(locals())
