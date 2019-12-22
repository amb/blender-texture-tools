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
    "version": (0, 1, 25),
    "blender": (2, 81, 0),
}

import numpy as np

CUDA_ACTIVE = False
try:
    import cupy as cup

    CUDA_ACTIVE = True
except Exception:
    CUDA_ACTIVE = False
    cup = np

import bpy

# from . import pycl
from . import image_ops
import importlib

importlib.reload(image_ops)


class BTT_InstallLibraries(bpy.types.Operator):
    bl_idname = "image.ied_install_libraries"
    bl_label = "Install CUDA support (cupy-cuda100 library)"

    def execute(self, context):
        from subprocess import call

        pp = bpy.app.binary_path_python

        call([pp, "-m", "ensurepip", "--user"])
        call([pp, "-m", "pip", "install", "--user", "cupy-cuda100"])

        global CUDA_ACTIVE
        CUDA_ACTIVE = True

        import cupy

        global cup
        cup = cupy

        return {"FINISHED"}


class BTT_AddonPreferences(bpy.types.AddonPreferences):
    bl_idname = __name__

    def draw(self, context):

        if CUDA_ACTIVE is False:
            info_text = (
                "The button below should automatically install required CUDA libs.\n"
                "You need to run the reload scripts command in Blender to activate the\n"
                " functionality after the installation finishes, or restart Blender."
            )
            col = self.layout.box().column(align=True)
            for l in info_text.split("\n"):
                row = col.row()
                row.label(text=l)
            # col.separator()
            row = self.layout.row()
            row.operator(BTT_InstallLibraries.bl_idname, text="Install CUDA acceleration library")
        else:
            row = self.layout.row()
            row.label(text="All optional libraries installed")


def gauss_curve(x):
    # gaussian with 0.01831 at last
    res = cup.array([cup.exp(-((i * (2 / x)) ** 2)) for i in range(-x, x + 1)], dtype=cup.float32)
    res /= cup.sum(res)
    return res


# def gauss_curve_np(x):
#     # gaussian with 0.01831 at last
#     res = np.array([np.exp(-((i * (2 / x)) ** 2)) for i in range(-x, x + 1)], dtype=np.float32)
#     res /= np.sum(res)
#     return res


def vectors_to_nmap(vectors, nmap):
    vectors *= 0.5
    nmap[:, :, 0] = vectors[:, :, 0] + 0.5
    nmap[:, :, 1] = vectors[:, :, 1] + 0.5
    nmap[:, :, 2] = vectors[:, :, 2] + 0.5


def nmap_to_vectors(nmap):
    vectors = cup.empty((nmap.shape[0], nmap.shape[1], 3), dtype=cup.float32)
    vectors[..., 0] = nmap[..., 0] - 0.5
    vectors[..., 1] = nmap[..., 1] - 0.5
    vectors[..., 2] = nmap[..., 2] - 0.5
    vectors *= 2.0
    return vectors


def neighbour_average(ig):
    return (ig[1:-1, :-2] + ig[1:-1, 2:] + ig[:-2, 1:-1] + ig[:-2, 1:-1]) * 0.25


def explicit_cross(a, b):
    x = a[..., 1] * b[..., 2] - a[..., 2] * b[..., 1]
    y = a[..., 2] * b[..., 0] - a[..., 0] * b[..., 2]
    z = a[..., 0] * b[..., 1] - a[..., 1] * b[..., 0]
    return cup.dstack([x, y, z])


def aroll0(o, i, d):
    if d > 0:
        k = d
        o[:-k, :] = i[k:, :]
        o[-k:, :] = i[:k, :]
    elif d < 0:
        k = -d
        o[k:, :] = i[:-k, :]
        o[:k, :] = i[-k:, :]


def aroll1(o, i, d):
    if d > 0:
        k = d
        o[:, :-k] = i[:, k:]
        o[:, -k:] = i[:, :k]
    elif d < 0:
        k = -d
        o[:, k:] = i[:, :-k]
        o[:, :k] = i[:, -k:]


def addroll0(o, i, d):
    if d > 0:
        k = d
        o[:-k, :] += i[k:, :]
        o[-k:, :] += i[:k, :]
    elif d < 0:
        k = -d
        o[k:, :] += i[:-k, :]
        o[:k, :] += i[-k:, :]


def addroll1(o, i, d):
    if d > 0:
        k = d
        o[:, :-k] += i[:, k:]
        o[:, -k:] += i[:, :k]
    elif d < 0:
        k = -d
        o[:, k:] += i[:, :-k]
        o[:, :k] += i[:, -k:]


def convolution(ssp, intens, sfil):
    # source, intensity, convolution matrix
    tpx = cup.zeros(ssp.shape, dtype=float)
    ysz, xsz = sfil.shape[0], sfil.shape[1]
    ystep = int(4 * ssp.shape[1])
    for y in range(ysz):
        for x in range(xsz):
            tpx += cup.roll(ssp, (x - xsz // 2) * 4 + (y - ysz // 2) * ystep) * sfil[y, x]
    return tpx


def grayscale(ssp):
    r, g, b = ssp[:, :, 0], ssp[:, :, 1], ssp[:, :, 2]
    gray = 0.2989 * r + 0.5870 * g + 0.1140 * b
    ssp[..., 0] = gray
    ssp[..., 1] = gray
    ssp[..., 2] = gray
    return ssp


def normalize(pix, save_alpha=False):
    if save_alpha:
        A = pix[..., 3]
    t = pix - cup.min(pix)
    t = t / cup.max(t)
    if save_alpha:
        t[..., 3] = A
    return t


def sobel_x(pix, intensity):
    gx = cup.array([[-1, 0, 1], [-2, 0, 2], [-1, 0, 1]])
    return convolution(pix, intensity, gx)


def sobel_y(pix, intensity):
    gy = cup.array([[1, 2, 1], [0, 0, 0], [-1, -2, -1]])
    return convolution(pix, intensity, gy)


def sobel(pix, intensity):
    retarr = cup.zeros(pix.shape)
    retarr = sobel_x(pix, 1.0)
    retarr += sobel_y(pix, 1.0)
    retarr = (retarr * intensity) * 0.5 + 0.5
    retarr[..., 3] = pix[..., 3]
    return retarr


def gaussian_repeat(pix, s):
    res = cup.zeros(pix.shape, dtype=cup.float32)
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


def sharpen(pix, width, intensity):
    # return convolution(pix, intensity, cup.array([[-1, -1, -1], [-1, 9, -1], [-1, -1, -1]]))
    A = pix[..., 3]
    gas = gaussian_repeat(pix, width)
    pix += (pix - gas) * intensity
    pix[..., 3] = A
    return pix


def hi_pass(pix, s, intensity):
    bg = pix.copy()
    pix = (bg - gaussian_repeat(pix, s)) * 0.5 + 0.5
    pix[:, :, 3] = bg[:, :, 3]
    return pix


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
    t_counts = gauss_curve_np(NG * 4)
    t_quantiles = np.cumsum(t_counts).astype(np.float64)

    t_max = 0.0
    for i in range(3):
        # s_values, bin_idx, s_counts = cup.lib.arraysetops.unique(
        s_values, bin_idx, s_counts = np.unique(
            source[..., i].ravel(), return_inverse=True, return_counts=True
        )

        s_quantiles = np.cumsum(s_counts).astype(cup.float64)
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
        s_values, bin_idx, s_counts = np.unique(
            output[..., i].ravel(), return_inverse=True, return_counts=True
        )
        t_values, t_quantiles, _ = transforms[i]

        s_quantiles = np.cumsum(s_counts).astype(cup.float64)
        s_quantiles /= s_quantiles[-1]

        tv = np.interp(s_quantiles, t_quantiles, t_values)[bin_idx]
        output[..., i] = tv.reshape(oldshape[:2])

    return output


def cumulative_distribution(data, bins):
    assert cup.min(data) >= 0.0 and cup.max(data) <= 1.0
    hg_av, hg_a = cup.unique(cup.floor(data * (bins - 1)), return_index=True)
    hg_a = cup.float32(hg_a)
    hgs = cup.sum(hg_a)
    hg_a /= hgs
    res = cup.zeros((bins,))
    res[cup.int64(hg_av)] = hg_a
    return cup.cumsum(res)


def hi_pass_balance(pix, s, zoom):
    bg = pix.copy()

    yzm = pix.shape[0] // 2
    xzm = pix.shape[1] // 2

    yzoom = zoom if zoom < yzm else yzm
    xzoom = zoom if zoom < xzm else xzm

    pixmin = np.min(pix)
    pixmax = np.max(pix)
    med = (pixmin + pixmax) / 2
    # TODO: np.mean
    gas = gaussian_repeat_np(pix - med, s) + med
    pix = (pix - gas) * 0.5 + 0.5
    for c in range(3):
        pix[..., c] = hist_match(
            pix[..., c], bg[yzm - yzoom : yzm + yzoom, xzm - xzoom : xzm + xzoom, c]
        )
    pix[..., 3] = bg[..., 3]
    return pix


def hgram_equalize(pix, intensity, atest):
    old = pix.copy()
    # aw = cup.argwhere(pix[..., 3] > atest)
    aw = (pix[..., 3] > atest).nonzero()
    aws = (aw[0], aw[1])
    # aws = (aw[:, 0], aw[:, 1])
    for c in range(3):
        t = pix[..., c][aws]
        pix[..., c][aws] = np.sort(t).searchsorted(t)
        # pix[..., c][aws] = cup.argsort(t)
    pix[..., :3] /= np.max(pix[..., :3])
    return old * (1.0 - intensity) + pix * intensity


def bilateral(img_in, sigma_s, sigma_v, eps=1e-8):
    # gaussian
    gsi = lambda r2, sigma: cup.exp(-0.5 * r2 / sigma ** 2)
    win_width = int(cup.ceil(3 * sigma_s))
    wgt_sum = cup.ones(img_in.shape) * eps
    result = img_in * eps
    off = np.empty_like(img_in, dtype=np.float32)

    assert off.dtype == img_in.dtype
    assert off.shape == img_in.shape

    for shft_x in range(-win_width, win_width + 1):
        for shft_y in range(-win_width, win_width + 1):
            aroll0(off, img_in, shft_y)
            aroll1(off, off, shft_x)

            w = gsi(shft_x ** 2 + shft_y ** 2, sigma_s)
            tw = w * gsi((off - img_in) ** 2, sigma_v)
            result += off * tw
            wgt_sum += tw

    # normalize the result and return
    return result / wgt_sum


def bilateral_filter(pix, s, intensity, source):
    # multiply by alpha
    # pix[..., 0] *= pix[..., 3]
    # pix[..., 1] *= pix[..., 3]
    # pix[..., 2] *= pix[..., 3]

    # TODO: this
    # if source == "SOBEL":
    #     sb = sobel(pix, 1.0)
    # else:
    #     sb = pix

    sb = pix

    print("R")
    # image, spatial, range
    pix[..., 0] = bilateral(sb[..., 0], s, intensity)
    print("G")
    pix[..., 1] = bilateral(sb[..., 1], s, intensity)
    print("B")
    pix[..., 2] = bilateral(sb[..., 2], s, intensity)

    return pix


def median_filter_blobs(pix, s, picked="center"):
    ph, pw = pix.shape[0], pix.shape[1]

    pick = 0
    if picked == "center":
        pick = s
    if picked == "end":
        pick = s * 2 - 1

    temp = pix.copy()
    r = cup.zeros((ph, s * 2, 4), dtype=np.float32)
    for x in range(pw):
        if x - s >= 0 and x + s <= pw:
            r[:, :, :] = temp[:, x - s : x + s, :]

        if x - s < 0:
            dp = s - x
            r[:, dp:, :] = temp[:, : x + s, :]
            r[:, :dp, :] = temp[:, -dp:, :]

        if x + s > pw:
            dp = x + s - pw
            r[:, :-dp, :] = temp[:, x - s :, :]
            r[:, -dp:, :] = temp[:, :dp, :]

        pix[:, x, :] = cup.sort(r, axis=1)[:, pick, :]

    temp = pix.copy()
    r = cup.zeros((s * 2, pw, 4), dtype=np.float32)
    for y in range(ph):
        if y - s >= 0 and y + s <= ph:
            r[:, :, :] = temp[y - s : y + s, :, :]

        if y - s < 0:
            dp = s - y
            r[dp:, :, :] = temp[: y + s, :, :]
            r[:dp, :, :] = temp[-dp:, :, :]

        if y + s > pw:
            dp = y + s - pw
            r[:-dp, :, :] = temp[y - s :, :, :]
            r[-dp:, :, :] = temp[:dp, :, :]

        pix[y, :, :] = cup.sort(r, axis=0)[pick, :, :]

    return pix


def normals_simple(pix, source):
    pix = grayscale(pix)
    pix = normalize(pix)
    sshape = pix.shape

    # extract x and y deltas
    px = sobel_x(pix, 1.0)
    px[:, :, 2] = px[:, :, 2]
    px[:, :, 1] = 0
    px[:, :, 0] = 1

    py = sobel_y(pix, 1.0)
    py[:, :, 2] = py[:, :, 2]
    py[:, :, 1] = 1
    py[:, :, 0] = 0

    # normalize
    # dv = max(abs(cup.min(curve)), abs(cup.max(curve)))
    # curve /= dv

    # find the imagined approximate surface normal
    # arr = cup.cross(px[:, :, :3], py[:, :, :3])
    arr = explicit_cross(px[:, :, :3], py[:, :, :3])
    print(arr.shape)

    # normalization: vec *= 1/len(vec)
    m = 1.0 / cup.sqrt(arr[:, :, 0] ** 2 + arr[:, :, 1] ** 2 + arr[:, :, 2] ** 2)
    arr[..., 0] *= m
    arr[..., 1] *= m
    arr[..., 2] *= m
    arr[..., 0] = -arr[..., 0]

    # normals format
    retarr = cup.zeros(sshape)
    vectors_to_nmap(arr, retarr)
    retarr[:, :, 3] = pix[..., 3]
    return retarr


def normals_to_curvature(pix):
    intensity = 1.0
    curve = cup.zeros((pix.shape[0], pix.shape[1]), dtype=cup.float32)
    vectors = nmap_to_vectors(pix)

    # y_vec = cup.array([1, 0, 0], dtype=cup.float32)
    # x_vec = cup.array([0, 1, 0], dtype=cup.float32)

    # yd = vectors.dot(x_vec)
    # xd = vectors.dot(y_vec)

    xd = vectors[:, :, 0]
    yd = vectors[:, :, 1]

    # curve[0,0] = yd[1,0]
    curve[:-1, :] += yd[1:, :]
    curve[-1, :] += yd[0, :]

    # curve[0,0] = yd[-1,0]
    curve[1:, :] -= yd[:-1, :]
    curve[0, :] -= yd[-1, :]

    # curve[0,0] = xd[1,0]
    curve[:, :-1] += xd[:, 1:]
    curve[:, -1] += xd[:, 0]

    # curve[0,0] = xd[-1,0]
    curve[:, 1:] -= xd[:, :-1]
    curve[:, 0] -= xd[:, -1]

    # normalize
    dv = max(abs(cup.min(curve)), abs(cup.max(curve)))
    curve /= dv

    # 0 = 0.5 grey
    curve = curve * intensity + 0.5

    pix[..., 0] = curve
    pix[..., 1] = curve
    pix[..., 2] = curve
    return pix


def curvature_to_height(image, h2, iterations=2000):
    f = image[..., 0]
    A = image[..., 3]
    u = cup.ones_like(f) * 0.5

    k = 1
    t = np.empty_like(u, dtype=np.float32)

    # periodic gauss seidel iteration
    for ic in range(iterations):
        if ic % 100 == 0:
            print(ic)

        # roll k, axis=0
        t[:-k, :] = u[k:, :]
        t[-k:, :] = u[:k, :]
        # roll -k, axis=0
        t[k:, :] += u[:-k, :]
        t[:k, :] += u[-k:, :]
        # roll k, axis=1
        t[:, :-k] += u[:, k:]
        t[:, -k:] += u[:, :k]
        # roll -k, axis=1
        t[:, k:] += u[:, :-k]
        t[:, :k] += u[:, -k:]

        t -= h2 * f
        t *= 0.25
        u = t * A

    u = -u
    u -= cup.min(u)
    u /= cup.max(u)

    return cup.dstack([u, u, u, image[..., 3]])


def normals_to_height(image, grid_steps, iterations=2000, intensity=1.0):
    # A = image[..., 3]
    ih, iw = image.shape[0], image.shape[1]
    u = cup.ones((ih, iw), dtype=np.float32) * 0.5

    vectors = nmap_to_vectors(image)
    # vectors[..., 0] = 0.5 - image[..., 0]
    # vectors[..., 1] = image[..., 1] - 0.5

    vectors *= intensity

    t = np.empty_like(u, dtype=np.float32)

    for k in range(grid_steps, -1, -1):
        # multigrid
        k = 2 ** k
        print("grid step:", k)

        n = cup.roll(vectors[..., 0], k, axis=1)
        n -= cup.roll(vectors[..., 0], -k, axis=1)
        n += cup.roll(vectors[..., 1], k, axis=0)
        n -= cup.roll(vectors[..., 1], -k, axis=0)
        n *= 0.125

        for ic in range(iterations):
            if ic % 100 == 0:
                print(ic)

            # roll k, axis=0
            t[:-k, :] = u[k:, :]
            t[-k:, :] = u[:k, :]
            # roll -k, axis=0
            t[k:, :] += u[:-k, :]
            t[:k, :] += u[-k:, :]
            # roll k, axis=1
            t[:, :-k] += u[:, k:]
            t[:, -k:] += u[:, :k]
            # roll -k, axis=1
            t[:, k:] += u[:, :-k]
            t[:, :k] += u[:, -k:]

            t *= 0.25
            u = t + n
            # zero alpha = zero height
            # u = u * A + cup.max(u) * (1 - A)

    u = -u
    u -= cup.min(u)
    u /= cup.max(u)

    return cup.dstack([u, u, u, image[..., 3]])


def delight_simple(image, dd, iterations=500):
    A = image[..., 3]
    u = cup.ones_like(image[..., 0])

    grads = cup.zeros((image.shape[0], image.shape[1], 2), dtype=cup.float32)
    grads[..., 0] = (cup.roll(image[..., 0], 1, axis=0) - image[..., 0]) * dd
    grads[..., 1] = (image[..., 0] - cup.roll(image[..., 0], 1, axis=1)) * dd
    # grads[..., 0] = (image[..., 0] - 0.5) * (dd)
    # grads[..., 1] = (image[..., 0] - 0.5) * (dd)
    for k in range(5, -1, -1):
        # multigrid
        k = 2 ** k
        print("grid step:", k)

        n = cup.roll(grads[..., 0], k, axis=1)
        n -= cup.roll(grads[..., 0], -k, axis=1)
        n += cup.roll(grads[..., 1], k, axis=0)
        n -= cup.roll(grads[..., 1], -k, axis=0)
        n *= 0.125 * image[..., 3]

        for ic in range(iterations):
            if ic % 100 == 0:
                print(ic)
            t = cup.roll(u, -k, axis=0)
            t += cup.roll(u, k, axis=0)
            t += cup.roll(u, -k, axis=1)
            t += cup.roll(u, k, axis=1)
            t *= 0.25

            # zero alpha = zero height
            u = t + n
            u = u * A + cup.max(u) * (1 - A)

    u = -u
    u -= cup.min(u)
    u /= cup.max(u)

    # u *= image[..., 3]

    # u -= cup.mean(u)
    # u /= max(abs(cup.min(u)), abs(cup.max(u)))
    # u *= 0.5
    # u += 0.5
    # u = 1.0 - u

    # return cup.dstack([(u - image[..., 0]) * 0.5 + 0.5, u, u, image[..., 3]])
    u = (image[..., 0] - u) * 0.5 + 0.5
    return cup.dstack([u, u, u, image[..., 3]])


def fill_alpha(image, style="black"):
    if style == "black":
        for c in range(3):
            image[..., c] *= image[..., 3]
        image[..., 3] = 1.0
        return image
    else:
        cols = [0.5, 0.5, 1.0]
        A = image[..., 3]
        for c in range(3):
            image[..., c] = cols[c] * (1 - A) + image[..., c] * A
        image[..., 3] = 1.0
        return image


def dog(pix, a, b, mp):
    pixb = pix.copy()
    pix[..., :3] = cup.abs(gaussian_repeat(pix, a) - gaussian_repeat(pixb, b))[..., :3]
    pix[pix < mp][..., :3] = 0.0
    return pix


def gimpify(image):
    pixels = np.copy(image)
    xs, ys = image.shape[1], image.shape[0]
    image = np.roll(image, xs * 2 + xs * 4 * (ys // 2))

    sxs = xs // 2
    sys = ys // 2

    # generate the mask
    mask_pix = []
    for y in range(0, sys):
        zy0 = y / sys + 0.001
        zy1 = 1 - y / sys + 0.001
        for x in range(0, sxs):
            xp = x / sxs
            p = 1.0 - zy0 / (1.0 - xp + 0.001)
            t = 1.0 - xp / zy1
            mask_pix.append(t if t > p else p)
            # imask[y, x] = max(, imask[y, x])

    tmask = np.array(mask_pix, dtype=np.float32)
    tmask = tmask.reshape((sys, sxs))
    imask = np.zeros((pixels.shape[0], pixels.shape[1]), dtype=np.float32)
    imask[:sys, :sxs] = tmask

    imask[imask < 0] = 0

    # copy the data into the three remaining corners
    imask[0 : sys + 1, sxs:xs] = np.fliplr(imask[0 : sys + 1, 0:sxs])
    imask[-sys:ys, 0:sxs] = np.flipud(imask[0:sys, 0:sxs])
    imask[-sys:ys, sxs:xs] = np.flipud(imask[0:sys, sxs:xs])
    imask[sys, :] = imask[sys - 1, :]  # center line

    # apply mask
    amask = np.empty(pixels.shape, dtype=float)
    amask[:, :, 0] = imask
    amask[:, :, 1] = imask
    amask[:, :, 2] = imask
    amask[:, :, 3] = imask

    return amask * image + (1.0 - amask) * pixels


def inpaint_tangents(pixels, threshold):
    # invalid = pixels[:, :, 2] < 0.5 + (self.tolerance * 0.5)
    invalid = pixels[:, :, 2] < threshold
    # n2 = (
    #     ((pixels[:, :, 0] - 0.5) * 2) ** 2
    #     + ((pixels[:, :, 1] - 0.5) * 2) ** 2
    #     + ((pixels[:, :, 2] - 0.5) * 2) ** 2
    # )
    # invalid |= (n2 < 0.9) | (n2 > 1.1)

    # grow selection
    for _ in range(2):
        invalid[0, :] = False
        invalid[-1, :] = False
        invalid[:, 0] = False
        invalid[:, -1] = False

        invalid = (
            np.roll(invalid, 1, axis=0)
            | np.roll(invalid, -1, axis=0)
            | np.roll(invalid, 1, axis=1)
            | np.roll(invalid, -1, axis=1)
        )

    pixels[invalid] = np.array([0.5, 0.5, 1.0, 1.0])

    invalid[0, :] = False
    invalid[-1, :] = False
    invalid[:, 0] = False
    invalid[:, -1] = False

    # fill
    front = np.copy(invalid)
    locs = [(0, -1, 1), (0, 1, -1), (1, -1, 1), (1, 1, -1)]
    for i in range(4):
        print("fill step:", i)
        for l in locs:
            r = np.roll(front, l[1], axis=l[0])
            a = (r != front) & front
            pixels[a] = pixels[np.roll(a, l[2], axis=l[0])]
            front[a] = False

    cl = np.roll(invalid, -1, axis=0)
    cr = np.roll(invalid, 1, axis=0)
    uc = np.roll(invalid, -1, axis=1)
    bc = np.roll(invalid, 1, axis=1)

    # smooth
    for i in range(4):
        print("smooth step:", i)
        pixels[invalid] = (pixels[invalid] + pixels[cl] + pixels[cr] + pixels[uc] + pixels[bc]) / 5

    return pixels


def normalize_tangents(image):
    ih, iw = image.shape[0], image.shape[1]
    vectors = cup.zeros((ih, iw, 3), dtype=cup.float32)
    vectors[..., 0] = image[..., 0] - 0.5
    vectors[..., 1] = image[..., 1] - 0.5
    vectors[..., 2] = image[..., 2] - 0.5

    vectors = (vectors.T / np.linalg.norm(vectors, axis=2)).T * 0.5

    retarr = cup.empty_like(image)
    retarr[:, :, 0] = 0.5 + vectors[:, :, 0]
    retarr[:, :, 1] = 0.5 + vectors[:, :, 1]
    retarr[:, :, 2] = 0.5 + vectors[:, :, 2]
    retarr[:, :, 3] = image[..., 3]

    return retarr


def image_to_material(image):
    return image


class Grayscale_IOP(image_ops.ImageOperatorGenerator):
    def generate(self):
        self.prefix = "grayscale"
        self.info = "Grayscale from RGB"
        self.category = "Basic"
        self.payload = lambda self, image, context: grayscale(image)


class Random_IOP(image_ops.ImageOperatorGenerator):
    def generate(self):
        self.prefix = "random"
        self.info = "Random RGB pixels"
        self.category = "Basic"

        def _pl(self, image, context):
            t = np.random.random(image.shape)
            t[..., 3] = 1.0
            return t

        self.payload = _pl


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


class Fractal_IOP(image_ops.ImageOperatorGenerator):
    def generate(self):
        self.props["count"] = bpy.props.IntProperty(name="Count", min=1, default=2)
        self.props["style"] = bpy.props.EnumProperty(
            name="Style",
            items=[
                ("blend", "Blend", "", 1),
                ("multiply", "Multiply", "", 2),
                ("multiply_b", "Multiply B", "", 3),
            ],
        )
        self.prefix = "fractal"
        self.info = "Fractalize image"
        self.category = "Basic"

        def _pl(self, image, context):
            # A = image[..., 3]
            iw, ih = image.shape[1], image.shape[0]
            iwh, ihh = iw // 2, ih // 2

            pix = image.copy()
            for i in range(self.count):
                if self.style == "blend":
                    smol = pix[::2, ::2, :] * 0.5
                    pix *= 0.5
                    pix[:ihh, :iwh, :] += smol
                    pix[-ihh:, :iwh, :] += smol
                    pix[:ihh, -iwh:, :] += smol
                    pix[-ihh:, -iwh:, :] += smol
                else:
                    smol = pix[::2, ::2, :].copy() * 2.0
                    pix[:ihh, :iwh, :] *= smol
                    pix[ihh:, :iwh, :] *= smol
                    pix[:ihh, iwh:, :] *= smol
                    pix[ihh:, iwh:, :] *= smol

                    if self.style == "multiply":
                        pix *= 0.5
                    else:
                        pix = (pix - 0.5) * 0.5 + 0.5

            # pix[..., 3] = A
            return pix

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


class CropToP2_IOP(image_ops.ImageOperatorGenerator):
    def generate(self):
        self.prefix = "crop_to_power"
        self.info = "Crops the middle of the image to power of twos"
        self.category = "Basic"

        def _pl(self, image, context):
            h, w = image.shape[0], image.shape[1]

            offx = 0
            offy = 0

            wpow = int(cup.log2(w))
            hpow = int(cup.log2(h))

            offx = (w - 2 ** wpow) // 2
            offy = (h - 2 ** hpow) // 2

            if w > 2 ** wpow:
                w = 2 ** wpow
            if h > 2 ** hpow:
                h = 2 ** hpow
            # crop to center
            image = image[offy : offy + h, offx : offx + w]

            return image

        self.payload = _pl


class CropToSquare_IOP(image_ops.ImageOperatorGenerator):
    def generate(self):
        self.prefix = "crop_to_square"
        self.info = "Crop the middle to square with two divisible height and width"
        self.category = "Basic"

        def _pl(self, image, context):
            h, w = image.shape[0], image.shape[1]

            offx = w // 2
            offy = h // 2

            if h > w:
                h = w
            if w > h:
                w = h

            xt = w // 2 - 1
            yt = w // 2 - 1

            # crop to center
            image = image[offy - yt : offy + yt, offx - xt : offx + xt]

            return image

        self.payload = _pl


class Sharpen_IOP(image_ops.ImageOperatorGenerator):
    def generate(self):
        self.props["width"] = bpy.props.IntProperty(name="Width", min=2, default=5)
        self.props["intensity"] = bpy.props.FloatProperty(name="Intensity", min=0.0, default=0.6)
        self.prefix = "sharpen"
        self.info = "Simple sharpen"
        self.category = "Filter"
        self.payload = lambda self, image, context: sharpen(image, self.width, self.intensity)


class Sobel_IOP(image_ops.ImageOperatorGenerator):
    def generate(self):
        # self.props["intensity"] = bpy.props.FloatProperty(name="Intensity", min=0.0, default=1.0)
        self.prefix = "sobel"
        self.info = "Sobel"
        self.category = "Filter"
        self.payload = lambda self, image, context: normalize(
            sobel(grayscale(image), 1.0), save_alpha=True
        )


class FillAlpha_IOP(image_ops.ImageOperatorGenerator):
    def generate(self):
        self.props["style"] = bpy.props.EnumProperty(
            name="Style",
            items=[("black", "Black color", "", 1), ("tangent", "Neutral tangent", "", 2)],
        )
        self.prefix = "fill_alpha"
        self.info = "Fill alpha with color or normal"
        self.category = "Basic"
        self.payload = lambda self, image, context: fill_alpha(image, style=self.style)


class GaussianBlur_IOP(image_ops.ImageOperatorGenerator):
    def generate(self):
        self.props["width"] = bpy.props.IntProperty(name="Width", min=1, default=2)
        # self.props["intensity"] = bpy.props.FloatProperty(name="Intensity", min=0.0, default=1.0)
        self.prefix = "gaussian_blur"
        self.info = "Does a Gaussian blur"
        self.category = "Filter"
        self.payload = lambda self, image, context: gaussian_repeat(image, self.width)


class BlobMedian_IOP(image_ops.ImageOperatorGenerator):
    def generate(self):
        self.props["style"] = bpy.props.EnumProperty(
            name="Style",
            items=[
                ("start", "Erode", "", 1),
                ("center", "Neutral", "", 2),
                ("end", "Dilate", "", 3),
            ],
        )
        self.props["width"] = bpy.props.IntProperty(name="Width", min=1, default=2)
        self.prefix = "blob_median"
        self.info = "Blob median filter"
        self.category = "Filter"
        self.payload = lambda self, image, context: median_filter_blobs(
            image, self.width, picked=self.style
        )


class Bilateral_IOP(image_ops.ImageOperatorGenerator):
    def generate(self):
        # self.props["source"] = bpy.props.EnumProperty(
        #     name="Source", items=[("LUMINANCE", "Luminance", "", 1), ("SOBEL", "Sobel", "", 2)]
        # )
        self.props["sigma_a"] = bpy.props.FloatProperty(name="Sigma A", min=0.01, default=3.0)
        self.props["sigma_b"] = bpy.props.FloatProperty(
            name="Sigma B", min=0.01, max=1.0, default=0.3
        )
        self.prefix = "bilateral"
        self.info = "Bilateral"
        self.category = "Filter"
        self.payload = lambda self, image, context: bilateral_filter(
            image, self.sigma_a, self.sigma_b, ""
        )


class HiPass_IOP(image_ops.ImageOperatorGenerator):
    def generate(self):
        self.props["width"] = bpy.props.IntProperty(name="Width", min=1, default=2)
        self.props["intensity"] = bpy.props.FloatProperty(name="Intensity", min=0.0, default=1.0)
        self.prefix = "high_pass"
        self.info = "High pass"
        self.category = "Filter"
        self.payload = lambda self, image, context: hi_pass(image, self.width, self.intensity)


class HiPassBalance_IOP(image_ops.ImageOperatorGenerator):
    def generate(self):
        self.props["width"] = bpy.props.IntProperty(name="Width", min=1, default=2)
        self.props["zoom"] = bpy.props.IntProperty(name="Center slice", min=5, default=1000)
        self.prefix = "hipass_balance"
        self.info = "Remove low frequencies from the image"
        self.category = "Balance"
        self.force_numpy = True
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
            mask -= cup.min(mask)
            mask /= cup.max(mask)
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
        self.force_numpy = True
        self.payload = lambda self, image, context: hgram_equalize(image, self.intensity, 0.5)


class Gaussianize_IOP(image_ops.ImageOperatorGenerator):
    def generate(self):
        self.props["count"] = bpy.props.IntProperty(name="Count", min=10, max=100000, default=1000)
        self.prefix = "gaussianize"
        self.info = "Gaussianize histogram"
        self.category = "Advanced"
        self.force_numpy = True
        self.payload = lambda self, image, context: gaussianize(image, NG=self.count)[0]


class GimpSeamless_IOP(image_ops.ImageOperatorGenerator):
    """Image seamless generator operator"""

    # TODO: the smoothing is not complete, it goes only one way
    def generate(self):
        self.prefix = "gimp_seamless"
        self.info = "Gimp style seamless image operation"
        self.category = "Advanced"
        self.force_numpy = True
        self.payload = lambda self, image, context: gimpify(image)


class HistogramSeamless_IOP(image_ops.ImageOperatorGenerator):
    def generate(self):
        self.prefix = "histogram_seamless"
        self.info = "Seamless histogram blending"
        self.category = "Advanced"
        self.force_numpy = True

        def _pl(self, image, context):
            gimg, transforms = gaussianize(image)
            blended = gimpify(gimg)
            return degaussianize(blended, transforms)

        self.payload = _pl


class Normals_IOP(image_ops.ImageOperatorGenerator):
    def generate(self):
        # self.props["source"] = bpy.props.EnumProperty(
        #     name="Source", items=[("LUMINANCE", "Luminance", "", 1), ("SOBEL", "Sobel", "", 2)]
        # )
        # self.props["width"] = bpy.props.IntProperty(name="Width", min=0, default=2)
        # self.props["intensity"] = bpy.props.FloatProperty(name="Intensity", min=0.0, default=1.0)
        self.prefix = "height_to_normals"
        self.info = "(Very rough estimate) normal map from RGB"
        self.category = "Normals"
        self.payload = lambda self, image, context: normals_simple(
            # image, self.width, self.intensity, "Luminance"
            image,
            "Luminance",
        )


class NormalsToCurvature_IOP(image_ops.ImageOperatorGenerator):
    def generate(self):
        # self.props["width"] = bpy.props.IntProperty(name="Width", min=0, default=2)
        # self.props["intensity"] = bpy.props.FloatProperty(name="Intensity", min=0.0, default=1.0)
        self.prefix = "normals_to_curvature"
        self.info = "Curvature map from tangent normal map"
        self.category = "Normals"
        self.payload = lambda self, image, context: normals_to_curvature(image)


class CurveToHeight_IOP(image_ops.ImageOperatorGenerator):
    def generate(self):
        self.props["step"] = bpy.props.FloatProperty(name="Step", min=0.00001, default=0.1)
        self.props["iterations"] = bpy.props.IntProperty(name="Iterations", min=10, default=400)
        self.prefix = "curvature_to_height"
        self.info = "Height from curvature"
        self.category = "Normals"
        self.payload = lambda self, image, context: curvature_to_height(
            image, self.step, iterations=self.iterations
        )


class NormalsToHeight_IOP(image_ops.ImageOperatorGenerator):
    def generate(self):
        self.props["grid"] = bpy.props.IntProperty(name="Grid subd", min=1, default=4)
        self.props["iterations"] = bpy.props.IntProperty(name="Iterations", min=10, default=200)
        self.prefix = "normals_to_height"
        self.info = "Normals to height"
        self.category = "Normals"
        self.payload = lambda self, image, context: normals_to_height(
            image, self.grid, iterations=self.iterations
        )


class Delight_IOP(image_ops.ImageOperatorGenerator):
    def generate(self):
        self.props["flip"] = bpy.props.BoolProperty(name="Flip direction", default=False)
        self.props["iterations"] = bpy.props.IntProperty(name="Iterations", min=10, default=200)
        self.prefix = "delighting"
        self.info = "Delight simple"
        self.category = "Normals"
        self.payload = lambda self, image, context: delight_simple(
            image, -1 if self.flip else 1, iterations=self.iterations
        )


class InpaintTangents_IOP(image_ops.ImageOperatorGenerator):
    def generate(self):
        # self.props["flip"] = bpy.props.BoolProperty(name="Flip direction", default=False)
        # self.props["iterations"] = bpy.props.IntProperty(name="Iterations", min=10, default=200)
        self.props["threshold"] = bpy.props.FloatProperty(
            name="Threshold", min=0.1, max=0.9, default=0.5
        )
        self.prefix = "inpaint_invalid"
        self.info = "Inpaint invalid tangents"
        self.category = "Normals"
        self.payload = lambda self, image, context: inpaint_tangents(image, self.threshold)


class NormalizeTangents_IOP(image_ops.ImageOperatorGenerator):
    def generate(self):
        self.prefix = "normalize_tangents"
        self.info = "Make all tangents length 1"
        self.category = "Normals"
        self.payload = lambda self, image, context: normalize_tangents(image)


class ImageToMaterial_IOP(image_ops.ImageOperatorGenerator):
    def generate(self):
        self.prefix = "image_to_material"
        self.info = "Create magic material from image"
        self.category = "Magic"
        self.payload = lambda self, image, context: image_to_material(image)


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

additional_classes = [BTT_InstallLibraries, BTT_AddonPreferences]

register, unregister = image_ops.create(locals(), additional_classes)
