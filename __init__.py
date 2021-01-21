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
    "version": (0, 1, 26),
    "blender": (2, 81, 0),
}

import bpy
import functools
import numpy as np
from . import pycl as cl
from ctypes import c_void_p as void_p

from . import image_ops
import importlib

importlib.reload(image_ops)


def min_ptwo(val, pt):
    "Gives the minimum divisionally aligned value for input value"
    assert val > 0
    assert pt > 0
    return ((val - 1) // pt + 1) * pt


def image2d_to_ndarray(queue, buf, shape, out=None, like=None, dtype="float32", **kw):
    "See pycl.py for buffer_to_ndarray"
    out = np.empty((shape[1], shape[0], 4), dtype)
    assert out.flags.contiguous, "Don't know how to write non-contiguous yet."
    # print(type(out), out.shape)
    ptr = void_p(out.__array_interface__["data"][0])
    evt = cl.clEnqueueReadImage(queue, buf, ptr, (0, 0, 0), (shape[0], shape[1], 1), 0, 0, **kw)
    return (out, evt)


def image2d_from_ndarray(queue, ary, buf=None, **kw):
    ary = np.ascontiguousarray(ary)
    if ary.__array_interface__["strides"]:
        raise ValueError("I don't know how to handle strided arrays yet.")
    ptr = void_p(ary.__array_interface__["data"][0])
    assert buf is not None
    evt = cl.clEnqueueWriteImage(
        queue, buf, ptr, (0, 0, 0), (ary.shape[1], ary.shape[0], 1), 0, 0, **kw
    )
    return (buf, evt)


class CLImage:
    def __init__(self, cldev, w, h):
        self.cldev = cldev
        self.width = w
        self.height = h
        self.image = cl.clCreateImage2D(cldev.ctx, w, h, imgformat=cldev.image_format)
        # flags=self.cldev.mem_flags.CL_MEM_READ_WRITE,

    def from_numpy(self, source):
        a_g, a_evt = image2d_from_ndarray(self.cldev.queue, source, buf=self.image)
        a_evt.wait()

    def to_numpy(self):
        res_v, evt = image2d_to_ndarray(self.cldev.queue, self.image, (self.width, self.height, 4))
        evt.wait()
        return res_v


class CLDev:
    "OpenCL device class specifically for image processing"

    def __init__(self):
        self.ctx = cl.clCreateContext()
        for dev in self.ctx.devices:
            print(dev.name)

        d = cl.clGetDeviceIDs()[0]
        cl.clGetDeviceInfo(d, cl.cl_device_info.CL_DEVICE_NAME)
        cl.clGetDeviceInfo(d, cl.cl_device_info.CL_DEVICE_TYPE)
        print("Device 0 available:", d.available)
        print("Max work item sizes:", d.max_work_item_sizes)

        print("Supported image formats for RGBA:")
        self.supported_rgba = [
            i.image_channel_data_type
            for i in cl.clGetSupportedImageFormats(self.ctx)
            if i.image_channel_order == cl.cl_channel_order.CL_RGBA
        ]
        print(self.supported_rgba)

        # Ensure we have RGBA float32
        assert cl.cl_channel_type.CL_FLOAT in self.supported_rgba

        self.queue = cl.clCreateCommandQueue(self.ctx)
        self.kernels = {}

        self.mem_flags = cl.cl_mem_flags
        self.image_format = cl.cl_image_format(
            cl.cl_channel_order.CL_RGBA, cl.cl_channel_type.CL_FLOAT
        )

    def build(self, name, source, argtypes=None):
        "Build CL kernel. Load from cache if exists. Returns CL kernel."

        if name in self.kernels:
            # print("Cache:", name)
            return self.kernels[name]

        print("Build:", name)
        try:
            b = cl.clCreateProgramWithSource(self.ctx, source).build()
            kernel = b[name]
        except KeyError as e:
            print(e)
            print(cl.clGetProgramBuildInfo(b, cl.cl_program_build_info.CL_PROGRAM_BUILD_STATUS, 0))
            print(cl.clGetProgramBuildInfo(b, cl.cl_program_build_info.CL_PROGRAM_BUILD_OPTIONS, 0))
            print(cl.clGetProgramBuildInfo(b, cl.cl_program_build_info.CL_PROGRAM_BUILD_LOG, 0))
            raise

        # kernel.argtypes = (cl.cl_int, cl.cl_int, cl.cl_int, cl.cl_mem, cl.cl_mem, cl.cl_mem)
        kernel.argtypes = argtypes
        self.kernels[name] = kernel
        return kernel

    def new_image(self, width, height):
        return CLImage(self, width, height)

    def new_image_from_numpy(self, arr):
        "shape[0]=height, shape[1]=width"
        i = CLImage(self, arr.shape[1], arr.shape[0])
        i.from_numpy(arr)
        return i

    def get_kernel(self, name):
        if name in self.kernels:
            return self.kernels[name]
        else:
            return None

    def run_buffer(self, kernel, params, inputs):
        "Run CL kernel on params. Multiple in, single out. Returns Numpy.float32 array."
        # mf = cl.cl_mem_flags
        # print(a_np.shape, np.max(a_np), np.min(a_np))
        assert len(inputs) > 0
        cl_inputs = []
        for ip in inputs:
            # Only f32 and matching dimensions
            assert ip.dtype == np.float32
            assert ip.shape == inputs[0].shape
            a_g, a_evt = cl.buffer_from_ndarray(self.queue, ip, blocking=False)
            a_evt.wait()
            cl_inputs.append(a_g)

        res_g = cl.clCreateBuffer(self.ctx, inputs[0].nbytes)
        f_shape = (min_ptwo(inputs[0].shape[1], 8), min_ptwo(inputs[0].shape[0], 8))
        run_evt = kernel(*params, *cl_inputs, res_g).on(self.queue, gsize=f_shape, lsize=(8, 8))
        res_v, evt = cl.buffer_to_ndarray(self.queue, res_g, wait_for=run_evt, like=inputs[0])
        evt.wait()

        return res_v

    def run(self, kernel, params, inputs):
        "Run CL kernel on params. Multiple in, single out. Returns Numpy.float32 array."
        assert len(inputs) > 0

        mf = self.mem_flags
        imgf = self.image_format

        cl_inputs = []
        w, h = inputs[0].shape[1], inputs[0].shape[0]
        for ip in inputs:
            # Only f32 and matching dimensions
            assert ip.dtype == np.float32
            assert ip.shape == inputs[0].shape
            img_b = cl.clCreateImage2D(self.ctx, w, h, imgformat=imgf, flags=mf.CL_MEM_READ_ONLY)
            a_g, a_evt = image2d_from_ndarray(self.queue, ip, buf=img_b)
            a_evt.wait()
            cl_inputs.append(a_g)

        f_shape = (min_ptwo(w, 8), min_ptwo(h, 8))
        res_g = cl.clCreateImage2D(self.ctx, w, h, imgformat=imgf, flags=mf.CL_MEM_WRITE_ONLY)

        run_evt = kernel(*params, *cl_inputs, res_g).on(self.queue, gsize=f_shape, lsize=(8, 8))
        res_v, evt = image2d_to_ndarray(self.queue, res_g, (w, h, 4), wait_for=run_evt)
        evt.wait()

        # print(np.max(res_v), np.min(res_v))
        # return np.ones(inputs[0].shape, dtype=np.float32)

        return res_v

    def run_raw(self, kernel, params, inputs, output, shape=None):
        "Run CL kernel on params. Multiple in, single out. CLImage buffers."
        assert len(inputs) > 0
        if shape is None:
            shape = (min_ptwo(output.width, 8), min_ptwo(output.height, 8))
        run_evt = kernel(*params, *[i.image for i in inputs], output.image).on(
            self.queue, offset=(0, 0), gsize=shape, lsize=(8, 8)
        )
        run_evt.wait()

    def to_buffer(self, narray):
        gc_c, gc_e = cl.buffer_from_ndarray(self.queue, narray)
        gc_e.wait()
        return gc_c


cl_builder = CLDev()


class BTT_AddonPreferences(bpy.types.AddonPreferences):
    bl_idname = __name__

    def draw(self, context):

        # if CUDA_ACTIVE is False:
        #     info_text = (
        #         "The button below should automatically install required CUDA libs.\n"
        #         "You need to run the reload scripts command in Blender to activate the\n"
        #         " functionality after the installation finishes, or restart Blender."
        #     )
        #     col = self.layout.box().column(align=True)
        #     for l in info_text.split("\n"):
        #         row = col.row()
        #         row.label(text=l)
        #     # col.separator()
        #     row = self.layout.row()
        #     row.operator(BTT_InstallLibraries.bl_idname, text="Install CUDA acceleration library")
        # else:
        row = self.layout.row()
        row.label(text="All optional libraries installed")


def grayscale_cl_array(ssp):
    src = """
    __kernel void grayscale(
        const int WIDTH,
        __global const float4 *A,
        __global const float4 *B,
        __global float4 *output)
    {
        int i = get_global_id(0);
        int j = get_global_id(1);
        int loc = i + j*WIDTH;

        float g = A[loc].x * 0.2989 + A[loc].y * 0.5870 + A[loc].z * 0.1140;

        output[loc].x = g;
        output[loc].y = g;
        output[loc].z = g;
        output[loc].w = A[loc].w;
    }
    """

    k = cl_builder.build("grayscale", src, (cl.cl_int, cl.cl_mem, cl.cl_mem, cl.cl_mem))
    res = cl_builder.run(k, (ssp.shape[0],), (ssp, ssp))
    # print(type(res), res.dtype, res.shape, np.max(res), np.min(res))
    # res = np.ones(ssp.shape, dtype=np.float32)
    return res


def grayscale(ssp):
    src = """
    __kernel void grayscale(
        __read_only image2d_t A,
        //__read_only image2d_t B,
        __write_only image2d_t output)
    {
        const int2 loc = (int2)(get_global_id(0), get_global_id(1));
        const sampler_t sampler = \
            CLK_NORMALIZED_COORDS_FALSE |
            CLK_ADDRESS_REPEAT |
            CLK_FILTER_NEAREST;
        float4 px = read_imagef(A, sampler, loc);
        //write_imagef(output, loc, (float4)(loc.x/1024.0, 0.5, loc.y/1024.0, 1.0));
        float g = px.x * 0.2989 + px.y * 0.5870 + px.z * 0.1140;
        write_imagef(output, loc, (float4)(g, g, g, px.w));
    }
    """

    k = cl_builder.build("grayscale", src, (cl.cl_image, cl.cl_image))
    return cl_builder.run(k, [], (ssp,))


@functools.lru_cache(maxsize=128)
def gauss_curve(x):
    # gaussian with 0.01831 at last
    res = np.array([np.exp(-((i * (2 / x)) ** 2)) for i in range(-x, x + 1)], dtype=np.float32)
    res /= np.sum(res)
    return res


def gaussian_repeat(pix, s):
    "Separated gaussian for image. Over borders = wraparound"
    assert pix.dtype == np.float32

    SAMPLER_DEF = """
    const sampler_t sampler = \
        CLK_NORMALIZED_COORDS_FALSE |
        CLK_ADDRESS_CLAMP_TO_EDGE |
        CLK_FILTER_NEAREST;
    const int x = get_global_id(0), y = get_global_id(1);
    const int width = get_image_width(output), height = get_image_height(output);
    """

    def _builder(name, core):
        return cl_builder.build(
            name,
            """
            __kernel void {NAME}(
                const int s,
                const __global float *gc,
                __read_only image2d_t input,
                __write_only image2d_t output)
            {{
                {SAMPLER}
                float4 color = (float4)0.0f;
                for (int i=0;i<s*2+1;i++)  {{
                    color += read_imagef(input, sampler, (int2)({CORE})) * gc[i];
                }}
                write_imagef(output, (int2)(x,y), color);
            }}
            """.format(
                SAMPLER=SAMPLER_DEF, CORE=core, NAME=name
            ),
            (cl.cl_int, cl.cl_mem, cl.cl_image, cl.cl_image),
        )

    # Horizontal gaussian blur wraparound
    kh = _builder("gaussian_h", "((x+i-s)+width)%width,y")

    # Vertical gaussian blur wraparound
    kv = _builder("gaussian_v", "x,((y+i-s)+height)%height")

    gc_c = cl_builder.to_buffer(gauss_curve(s))
    img = cl_builder.new_image_from_numpy(pix)
    out = cl_builder.new_image(img.width, img.height)
    cl_builder.run_raw(kh, [s, gc_c], (img,), out)
    cl_builder.run_raw(kv, [s, gc_c], (out,), img)
    return img.to_numpy()


def bilateral_cl(pix, radius, preserve):
    "Bilateral filter, OpenCL implementation"

    src = """
    #define POW2(a) ((a) * (a))
    kernel void bilateral(
        const float radius,
        const float preserve,
        __read_only image2d_t input,
        __write_only image2d_t output
    )
    {
        int gidx       = get_global_id(0);
        int gidy       = get_global_id(1);
        const sampler_t sampler = \
            CLK_NORMALIZED_COORDS_FALSE |
            CLK_ADDRESS_CLAMP_TO_EDGE |
            CLK_FILTER_NEAREST;

        int n_radius   = ceil(radius);
        int dst_width  = get_global_size(0);
        int src_width  = dst_width + n_radius * 2;

        int u, v, i, j;

        float4 center_pix =
            read_imagef(input, sampler, (int2)(gidx, gidy));

        float4 accumulated = 0.0f;
        float4 tempf       = 0.0f;
        float  count       = 0.0f;
        float  diff_map, gaussian_weight, weight;

        for (v = -n_radius;v <= n_radius; ++v) {
            for (u = -n_radius;u <= n_radius; ++u) {
                i = gidx + u;
                j = gidy + v;

                tempf = read_imagef(input, sampler, (int2)(i, j));
                diff_map = exp (
                    - (   POW2(center_pix.x - tempf.x)
                        + POW2(center_pix.y - tempf.y)
                        + POW2(center_pix.z - tempf.z))
                    * preserve);

                gaussian_weight =
                    exp( - 0.5f * (POW2(u) + POW2(v)) / radius);

                weight = diff_map * gaussian_weight;

                accumulated += tempf * weight;
                count += weight;
            }
        }
        write_imagef(output, (int2)(gidx,gidy), accumulated / count);
        //write_imagef(output, (int2)(gidx,gidy), (float4)(1.0, 0.0, 1.0, 1.0));
    }
    """
    blr = cl_builder.build("bilateral", src, (cl.cl_float, cl.cl_float, cl.cl_image, cl.cl_image))
    img = cl_builder.new_image_from_numpy(pix)
    out = cl_builder.new_image(img.width, img.height)
    cl_builder.run_raw(blr, [radius, preserve], (img,), out)
    return out.to_numpy()


def median_filter(pix, s, picked="center"):

    return pix


def vectors_to_nmap(vectors, nmap):
    vectors *= 0.5
    nmap[:, :, 0] = vectors[:, :, 0] + 0.5
    nmap[:, :, 1] = vectors[:, :, 1] + 0.5
    nmap[:, :, 2] = vectors[:, :, 2] + 0.5


def explicit_cross(a, b):
    x = a[..., 1] * b[..., 2] - a[..., 2] * b[..., 1]
    y = a[..., 2] * b[..., 0] - a[..., 0] * b[..., 2]
    z = a[..., 0] * b[..., 1] - a[..., 1] * b[..., 0]
    return np.dstack([x, y, z])


def nmap_to_vectors(nmap):
    vectors = np.empty((nmap.shape[0], nmap.shape[1], 3), dtype=np.float32)
    vectors[..., 0] = nmap[..., 0] - 0.5
    vectors[..., 1] = nmap[..., 1] - 0.5
    vectors[..., 2] = nmap[..., 2] - 0.5
    vectors *= 2.0
    return vectors


def neighbour_average(ig):
    return (ig[1:-1, :-2] + ig[1:-1, 2:] + ig[:-2, 1:-1] + ig[:-2, 1:-1]) * 0.25


def convolution(ssp, intens, sfil):
    # source, intensity, convolution matrix
    tpx = np.zeros(ssp.shape, dtype=float)
    ysz, xsz = sfil.shape[0], sfil.shape[1]
    ystep = int(4 * ssp.shape[1])
    for y in range(ysz):
        for x in range(xsz):
            tpx += np.roll(ssp, (x - xsz // 2) * 4 + (y - ysz // 2) * ystep) * sfil[y, x]
    return tpx


def normalize(pix, save_alpha=False):
    if save_alpha:
        A = pix[..., 3]
    t = pix - np.min(pix)
    t = t / np.max(t)
    if save_alpha:
        t[..., 3] = A
    return t


def sobel_x(pix, intensity):
    gx = np.array([[-1, 0, 1], [-2, 0, 2], [-1, 0, 1]])
    return convolution(pix, intensity, gx)


def sobel_y(pix, intensity):
    gy = np.array([[1, 2, 1], [0, 0, 0], [-1, -2, -1]])
    return convolution(pix, intensity, gy)


def sobel(pix, intensity):
    retarr = np.zeros(pix.shape)
    retarr = sobel_x(pix, 1.0)
    retarr += sobel_y(pix, 1.0)
    retarr = (retarr * intensity) * 0.5 + 0.5
    retarr[..., 3] = pix[..., 3]
    return retarr


def sharpen(pix, width, intensity):
    # return convolution(pix, intensity, np.array([[-1, -1, -1], [-1, 9, -1], [-1, -1, -1]]))
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
        # s_values, bin_idx, s_counts = np.lib.arraysetops.unique(
        s_values, bin_idx, s_counts = np.unique(
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
        s_values, bin_idx, s_counts = np.unique(
            output[..., i].ravel(), return_inverse=True, return_counts=True
        )
        t_values, t_quantiles, _ = transforms[i]

        s_quantiles = np.cumsum(s_counts).astype(np.float64)
        s_quantiles /= s_quantiles[-1]

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
    # aw = np.argwhere(pix[..., 3] > atest)
    aw = (pix[..., 3] > atest).nonzero()
    aws = (aw[0], aw[1])
    # aws = (aw[:, 0], aw[:, 1])
    for c in range(3):
        t = pix[..., c][aws]
        pix[..., c][aws] = np.sort(t).searchsorted(t)
        # pix[..., c][aws] = np.argsort(t)
    pix[..., :3] /= np.max(pix[..., :3])
    return old * (1.0 - intensity) + pix * intensity


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
    # dv = max(abs(np.min(curve)), abs(np.max(curve)))
    # curve /= dv

    # find the imagined approximate surface normal
    # arr = np.cross(px[:, :, :3], py[:, :, :3])
    arr = explicit_cross(px[:, :, :3], py[:, :, :3])
    print(arr.shape)

    # normalization: vec *= 1/len(vec)
    m = 1.0 / np.sqrt(arr[:, :, 0] ** 2 + arr[:, :, 1] ** 2 + arr[:, :, 2] ** 2)
    arr[..., 0] *= m
    arr[..., 1] *= m
    arr[..., 2] *= m
    arr[..., 0] = -arr[..., 0]

    # normals format
    retarr = np.zeros(sshape)
    vectors_to_nmap(arr, retarr)
    retarr[:, :, 3] = pix[..., 3]
    return retarr


def normals_to_curvature(pix):
    intensity = 1.0
    curve = np.zeros((pix.shape[0], pix.shape[1]), dtype=np.float32)
    vectors = nmap_to_vectors(pix)

    # y_vec = np.array([1, 0, 0], dtype=np.float32)
    # x_vec = np.array([0, 1, 0], dtype=np.float32)

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
    dv = max(abs(np.min(curve)), abs(np.max(curve)))
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
    u = np.ones_like(f) * 0.5

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
    u -= np.min(u)
    u /= np.max(u)

    return np.dstack([u, u, u, image[..., 3]])


def normals_to_height(image, grid_steps, iterations=2000, intensity=1.0):
    # A = image[..., 3]
    ih, iw = image.shape[0], image.shape[1]
    u = np.ones((ih, iw), dtype=np.float32) * 0.5

    vectors = nmap_to_vectors(image)
    # vectors[..., 0] = 0.5 - image[..., 0]
    # vectors[..., 1] = image[..., 1] - 0.5

    vectors *= intensity

    t = np.empty_like(u, dtype=np.float32)

    for k in range(grid_steps, -1, -1):
        # multigrid
        k = 2 ** k
        print("grid step:", k)

        n = np.roll(vectors[..., 0], k, axis=1)
        n -= np.roll(vectors[..., 0], -k, axis=1)
        n += np.roll(vectors[..., 1], k, axis=0)
        n -= np.roll(vectors[..., 1], -k, axis=0)
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
            # u = u * A + np.max(u) * (1 - A)

    u = -u
    u -= np.min(u)
    u /= np.max(u)

    return np.dstack([u, u, u, image[..., 3]])


def delight_simple(image, dd, iterations=500):
    A = image[..., 3]
    u = np.ones_like(image[..., 0])

    grads = np.zeros((image.shape[0], image.shape[1], 2), dtype=np.float32)
    grads[..., 0] = (np.roll(image[..., 0], 1, axis=0) - image[..., 0]) * dd
    grads[..., 1] = (image[..., 0] - np.roll(image[..., 0], 1, axis=1)) * dd
    # grads[..., 0] = (image[..., 0] - 0.5) * (dd)
    # grads[..., 1] = (image[..., 0] - 0.5) * (dd)
    for k in range(5, -1, -1):
        # multigrid
        k = 2 ** k
        print("grid step:", k)

        n = np.roll(grads[..., 0], k, axis=1)
        n -= np.roll(grads[..., 0], -k, axis=1)
        n += np.roll(grads[..., 1], k, axis=0)
        n -= np.roll(grads[..., 1], -k, axis=0)
        n *= 0.125 * image[..., 3]

        for ic in range(iterations):
            if ic % 100 == 0:
                print(ic)
            t = np.roll(u, -k, axis=0)
            t += np.roll(u, k, axis=0)
            t += np.roll(u, -k, axis=1)
            t += np.roll(u, k, axis=1)
            t *= 0.25

            # zero alpha = zero height
            u = t + n
            u = u * A + np.max(u) * (1 - A)

    u = -u
    u -= np.min(u)
    u /= np.max(u)

    # u *= image[..., 3]

    # u -= np.mean(u)
    # u /= max(abs(np.min(u)), abs(np.max(u)))
    # u *= 0.5
    # u += 0.5
    # u = 1.0 - u

    # return np.dstack([(u - image[..., 0]) * 0.5 + 0.5, u, u, image[..., 3]])
    u = (image[..., 0] - u) * 0.5 + 0.5
    return np.dstack([u, u, u, image[..., 3]])


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
    pix[..., :3] = np.abs(gaussian_repeat(pix, a) - gaussian_repeat(pixb, b))[..., :3]
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
    vectors = np.zeros((ih, iw, 3), dtype=np.float32)
    vectors[..., 0] = image[..., 0] - 0.5
    vectors[..., 1] = image[..., 1] - 0.5
    vectors[..., 2] = image[..., 2] - 0.5

    vectors = (vectors.T / np.linalg.norm(vectors, axis=2)).T * 0.5

    retarr = np.empty_like(image)
    retarr[:, :, 0] = 0.5 + vectors[:, :, 0]
    retarr[:, :, 1] = 0.5 + vectors[:, :, 1]
    retarr[:, :, 2] = 0.5 + vectors[:, :, 2]
    retarr[:, :, 3] = image[..., 3]

    return retarr


def image_to_material(image):
    # TODO: Finish this
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


class TestPattern_IOP(image_ops.ImageOperatorGenerator):
    def generate(self):
        # self.props["order_a"] = bpy.props.StringProperty(name="Order A", default="RGBA")
        # self.props["order_b"] = bpy.props.StringProperty(name="Order B", default="RBGa")
        # self.props["direction"] = bpy.props.EnumProperty(
        #     name="Direction", items=[("ATOB", "A to B", "", 1), ("BTOA", "B to A", "", 2)]
        # )
        self.prefix = "test_pattern"
        self.info = "Test pattern"
        self.category = "Basic"

        def _pl(self, image, context):
            # RED
            image[:, 90:100, 0] = 1.0

            # GREEN
            image[90:100, :, 1] = 1.0

            return image

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

            wpow = int(np.log2(w))
            hpow = int(np.log2(h))

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
        self.props["intensity"] = bpy.props.FloatProperty(name="Intensity", min=0.0, default=1.0)
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


class Median_IOP(image_ops.ImageOperatorGenerator):
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
        self.prefix = "median_filter"
        self.info = "Median filter"
        self.category = "Filter"
        self.payload = lambda self, image, context: median_filter(
            image, self.width, picked=self.style
        )


class Bilateral_IOP(image_ops.ImageOperatorGenerator):
    def generate(self):
        self.props["radius"] = bpy.props.FloatProperty(
            name="Radius", min=0.01, max=100.0, default=3.0
        )
        self.props["preserve"] = bpy.props.FloatProperty(
            name="Preserve", min=0.01, max=100.0, default=0.3
        )
        self.prefix = "bilateral"
        self.info = "Bilateral"
        self.category = "Filter"
        self.payload = lambda self, image, context: bilateral_cl(image, self.radius, self.preserve)


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


# class Delight_IOP(image_ops.ImageOperatorGenerator):
#     def generate(self):
#         self.props["flip"] = bpy.props.BoolProperty(name="Flip direction", default=False)
#         self.props["iterations"] = bpy.props.IntProperty(name="Iterations", min=10, default=200)
#         self.prefix = "delighting"
#         self.info = "Delight simple"
#         self.category = "Normals"
#         self.payload = lambda self, image, context: delight_simple(
#             image, -1 if self.flip else 1, iterations=self.iterations
#         )


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


# class ImageToMaterial_IOP(image_ops.ImageOperatorGenerator):
#     def generate(self):
#         self.prefix = "image_to_material"
#         self.info = "Create magic material from image"
#         self.category = "Magic"
#         self.payload = lambda self, image, context: image_to_material(image)


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

# additional_classes = [BTT_InstallLibraries, BTT_AddonPreferences]
additional_classes = [BTT_AddonPreferences]

register, unregister = image_ops.create(locals(), additional_classes)
