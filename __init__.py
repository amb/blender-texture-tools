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
    "version": (0, 1, 17),
    "blender": (2, 80, 0),
}

import numpy

np = numpy
import bpy
import copy
from . import image_ops
import importlib
import bmesh
import mathutils as mu

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


# def fast_blur(tpx, s):
#     d = 2 ** s
#     ystep = tpx.shape[1]
#     while d > 1:
#         tpx = (tpx * 2 + numpy.roll(tpx, -d * 4) + numpy.roll(tpx, d * 4)) / 4
#         tpx = (tpx * 2 + numpy.roll(tpx, -d * (ystep * 4)) + numpy.roll(tpx, d * (ystep * 4))) / 4
#         d = int(d / 2)
#     return tpx


# def normals_simple(self, intensity):
#     self.grayscale()
#     sshape = self.pixels.shape

#     px = self.copy().sobel_x(1.0).pixels
#     px[:, :, 2] = px[:, :, 2] * intensity
#     px[:, :, 1] = 0
#     px[:, :, 0] = 1

#     py = self.sobel_y(1.0).pixels
#     py[:, :, 2] = py[:, :, 2] * intensity
#     py[:, :, 1] = 1
#     py[:, :, 0] = 0

#     arr = numpy.cross(px[:, :, :3], py[:, :, :3])

#     # vec *= 1/len(vec)
#     m = 1.0 / numpy.sqrt(arr[:, :, 0] ** 2 + arr[:, :, 1] ** 2 + arr[:, :, 2] ** 2)
#     arr[..., 0] *= m
#     arr[..., 1] *= m
#     arr[..., 2] *= m
#     vectors = arr

#     retarr = numpy.zeros(sshape)
#     retarr[:, :, 0] = 0.5 - vectors[:, :, 0]
#     retarr[:, :, 1] = vectors[:, :, 1] + 0.5
#     retarr[:, :, 2] = vectors[:, :, 2]
#     retarr[:, :, 3] = 1.0
#     self.pixels = retarr
#     return self


def sobel(pix, s, intensity):
    retarr = numpy.zeros(pix.shape)
    retarr = sobel_x(retarr, 1.0)
    retarr += sobel_y(retarr, 1.0)
    retarr = (retarr * intensity) * 0.5 + 0.5
    retarr[..., 3] = 1.0
    return retarr


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
        self.props["width"] = bpy.props.IntProperty(name="Width", min=1, default=3)
        self.props["intensity"] = bpy.props.FloatProperty(name="Intensity", min=0.01, default=0.1)
        self.prefix = "bilateral_filter"
        self.info = "Bilateral"
        self.category = "Filter"
        self.payload = lambda self, image, context: bilateral_filter(
            image, self.width, self.intensity
        )


class HistogramQ_IOP(image_ops.ImageOperatorGenerator):
    def generate(self):
        self.props["intensity"] = bpy.props.FloatProperty(name="Intensity", min=0.01, default=0.1)
        self.prefix = "histogram_q"
        self.info = "Histogram quantization"
        self.category = "Filter"
        self.payload = lambda self, image, context: image


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


# class Normals_IOP(image_ops.ImageOperatorGenerator):
#     def generate(self):
#         self.props["intensity"] = bpy.props.FloatProperty(name="Intensity", min=0.0, default=1.0)
#         self.prefix = "normals"
#         self.info = "(Very rough estimate) normal map from RGB"
#         self.category = "Filter"
#         self.payload = (
#             lambda self, image, context: ImageOperations(image)
#             .normals_simple(image.pixels[:], self.intensity)
#             .pixels
#         )


# class LaplacianBlend_IOP(image_ops.ImageOperatorGenerator):
#     def generate(self):
#         self.prefix = "laplacian_blend"
#         self.info = "Blends two images with Laplacian pyramids"
#         self.category = "Filter"

#         def _pl(self, image, context):
#             return image

#         self.payload = _pl


# class RenderObject_IOP(image_ops.ImageOperatorGenerator):
#     def generate(self):
#         self.props["object"] = bpy.props.PointerProperty(name="Target", type=bpy.types.Object)

#         self.prefix = "render_object"
#         self.info = "Simple render of selected object"
#         self.category = "Debug"

#         def _pl(self, image, context):
#             bm = bmesh.new()
#             bm.from_mesh(self.object.data)
#             bmesh.ops.triangulate(bm, faces=bm.faces[:])

#             datatype = np.float32

#             # rays
#             rays = np.empty((image.shape[0], image.shape[1], 2, 3), dtype=datatype)
#             w, h = image.shape[0], image.shape[1]
#             for x in range(w):
#                 for y in range(h):
#                     # ray origin
#                     rays[x, y, 0, 0] = y * 2 / h - 1.0
#                     rays[x, y, 0, 1] = -5.0
#                     rays[x, y, 0, 2] = x * 2 / w - 1.0
#                     # ray direction
#                     rays[x, y, 1, 0] = 0.0
#                     rays[x, y, 1, 1] = 1.0
#                     rays[x, y, 1, 2] = 0.0

#             # mesh
#             tris = np.zeros((len(bm.faces), 3, 3), dtype=datatype)
#             for fi, f in enumerate(bm.faces):
#                 vv = f.verts
#                 # tris[fi] = [i.co for i in vv]
#                 tris[fi][0][0] = vv[0].co[0]
#                 tris[fi][0][1] = vv[0].co[1]
#                 tris[fi][0][2] = vv[0].co[2]
#                 tris[fi][1][0] = vv[1].co[0]
#                 tris[fi][1][1] = vv[1].co[1]
#                 tris[fi][1][2] = vv[1].co[2]
#                 tris[fi][2][0] = vv[2].co[0]
#                 tris[fi][2][1] = vv[2].co[1]
#                 tris[fi][2][2] = vv[2].co[2]
#                 # v1v0 = vv[1].co - vv[0].co
#                 # v2v0 = vv[2].co - vv[0].co
#                 # assert v1v0.length > 0.0
#                 # assert v2v0.length > 0.0

#             bm.faces.ensure_lookup_table()

#             # sun_direction = np.array(mu.Vector([0.5, -0.5, 0.5]).normalized(), dtype=datatype)
#             # normals = np.array(
#             #     [np.array(i.normal, dtype=datatype) for i in bm.faces], dtype=datatype
#             # )

#             print(image.shape, rays.shape, tris.shape, rays.dtype)
#             # result = np.zeros((image.shape[0], image.shape[1]), dtype=datatype)

#             def rt_nb(do_a_jit=True):
#                 import numba

#                 def intersect_ray(ro, rda, vrt):
#                     def cross(a, b):
#                         return np.array(
#                             [
#                                 a[1] * b[2] - a[2] * b[1],
#                                 a[2] * b[0] - a[0] * b[2],
#                                 a[0] * b[1] - a[1] * b[0],
#                             ]
#                         )

#                     def dot(a, b):
#                         return a[0] * b[0] + a[1] * b[1] + a[2] * b[2]

#                     def tri_intersect(ro, rd, v0, v1, v2):
#                         v1v0 = v1 - v0
#                         v2v0 = v2 - v0
#                         rov0 = ro - v0
#                         n = cross(v1v0, v2v0)
#                         q = cross(rov0, rd)
#                         rdn = dot(rd, n)
#                         if rdn == 0.0:
#                             return -1.0
#                             # return (-1.0, 0.0, 0.0)
#                         d = 1.0 / rdn
#                         u = d * (dot(-q, v2v0))
#                         v = d * (dot(q, v1v0))
#                         t = d * (dot(-n, rov0))
#                         if u < 0.0 or u > 1.0 or v < 0.0 or u + v > 1.0:
#                             t = -1.0
#                         # return (t, u, v)
#                         return t

#                     c = 1.0e10
#                     n = -1
#                     for i in range(len(vrt) // 3):
#                         iv = i * 3
#                         rcast = tri_intersect(ro, rda, vrt[iv], vrt[iv + 1], vrt[iv + 2])
#                         if rcast < c and rcast > 0.0:
#                             c = rcast
#                             n = i

#                     return n

#                 if do_a_jit:
#                     intersect_ray = numba.njit(parallel=False)(intersect_ray)

#                 result = np.empty((image.shape[0], image.shape[1]), dtype=np.float32)

#                 def rnd_res(ro, rd, verts, normals, sun_direction, res):
#                     for x in range(res.shape[0]):
#                         print(x)
#                         for y in numba.prange(res.shape[1]):
#                             r = intersect_ray(ro[x, y], rd, verts)
#                             res[x, y] = np.dot(normals[r], sun_direction) if r >= 0 else 0.0

#                 rnd_res(ro, rd, verts, normals, sun_direction, result)

#                 return result

#             # numba is aboug 20x speedup with single core CPU
#             # result = rt_nb(do_a_jit=True)

#             def rt_glcompute():
#                 # in: rays, tris
#                 # out: distance, u, v, face index

#                 from .bpy_amb import raycast
#                 import importlib

#                 importlib.reload(raycast)

#                 rc = raycast.Raycaster(tris)
#                 rw = rays.shape[0]
#                 res = rc.cast(rays.reshape((rw * rw, 2, 3)))

#                 return res.reshape((rw, rw, 4))

#             result = rt_glcompute()
#             dist = result[:, :, 0]
#             dist = np.where(dist < 50.0, (dist - 4.0) / 2.0, 1.0)

#             image[:, :, 0] = dist
#             image[:, :, 1] = result[:, :, 1]
#             image[:, :, 2] = result[:, :, 2]

#             bm.free()
#             return image

#         self.payload = _pl


register, unregister = image_ops.create(locals())
