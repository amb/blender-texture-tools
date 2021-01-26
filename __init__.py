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

import bpy  # noqa
import functools
import numpy as np
from . import pycl as cl
from ctypes import c_void_p as void_p
from typing import NewType

from . import image_ops
import importlib

importlib.reload(image_ops)


def min_ptwo(val, pt):
    "Gives the minimum divisionally aligned value for input value"
    assert val > 0
    assert pt > 0
    return ((val - 1) // pt + 1) * pt


class CLImage:
    def __init__(self, cldev, w, h):
        self.cldev = cldev
        self.width = w
        self.height = h
        # Default memflags are CL_MEM_READ_WRITE
        self.image = cl.clCreateImage2D(cldev.ctx, w, h, imgformat=cldev.image_format)

    def from_numpy(self, source):
        ary = np.ascontiguousarray(source)
        if ary.__array_interface__["strides"]:
            raise ValueError("I don't know how to handle strided arrays yet.")
        ptr = void_p(ary.__array_interface__["data"][0])
        evt = cl.clEnqueueWriteImage(
            self.cldev.queue, self.image, ptr, (0, 0, 0), (ary.shape[1], ary.shape[0], 1), 0, 0
        )
        evt.wait()

    def to_numpy(self):
        "See pycl.py for buffer_to_ndarray"
        out = np.empty((self.height, self.width, 4), dtype=np.float32)
        assert out.flags.contiguous, "Don't know how to write non-contiguous yet."
        ptr = void_p(out.__array_interface__["data"][0])
        # TODO: enforce x%8==0 pixel step
        evt = cl.clEnqueueReadImage(
            self.cldev.queue,
            self.image,
            ptr,
            (0, 0, 0),
            (self.width, self.height, 1),
            self.width * 4 * 4,
            0,
        )
        evt.wait()
        return out


CLFloat = NewType("CLFloat", float)
CLInt = NewType("CLInt", int)


class CLFloatArray:
    def __init__(self, a):
        assert a.dtype == np.float32
        self.data = a


class CLIntArray:
    def __init__(self, a):
        assert a.dtype == np.int32
        self.data = a


class CLDev:
    "OpenCL device class specifically for image processing"

    def __init__(self):
        self.ctx = cl.clCreateContext()
        for dev in self.ctx.devices:
            print(dev.name)

        dvids = cl.clGetDeviceIDs()
        # print(len(dvids))
        d = dvids[0]
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
        cl_inputs = []
        # TODO: w, h is different between in and out images (do min_ptwo for both)
        w, h = inputs[0].shape[1], inputs[0].shape[0]
        for ip in inputs:
            # Only f32 and matching dimensions
            assert ip.dtype == np.float32
            assert ip.shape == inputs[0].shape
            cl_inputs.append(self.new_image_from_numpy(ip))
        out = cl_builder.new_image(w, h)
        cl_builder.run_raw(kernel, params, cl_inputs, out)
        return out.to_numpy()

    def run_raw(self, kernel, params, inputs, output, shape=None):
        "Run CL kernel on params. Multiple in, single out. CLImage buffers."
        assert len(inputs) > 0
        if shape is None:
            shape = (min_ptwo(output.width, 8), min_ptwo(output.height, 8))
        run_evt = kernel(*params, *[i.image for i in inputs], output.image).on(
            self.queue, offset=(0, 0), gsize=shape, lsize=(8, 8)
        )
        run_evt.wait()

    def run_raw_buffer(self, kernel, params, output, shape=None):
        "Run CL kernel on params. Multiple in, single out. Generic buffers."
        assert shape is not None
        f_shape = (min_ptwo(shape[1], 8), min_ptwo(shape[0], 8))
        run_evt = kernel(*params, output).on(self.queue, gsize=f_shape, lsize=(8, 8))
        run_evt.wait()

    def to_buffer(self, narray):
        gc_c, gc_e = cl.buffer_from_ndarray(self.queue, narray)
        gc_e.wait()
        return gc_c

    def _decorator_builder(self, input_func, seamless=False):
        cl_types = {
            "CLFloat": ("const float", cl.cl_float),
            "CLInt": ("const int", cl.cl_int),
            "CLFloatArray": ("const __global float*", cl.cl_mem),
            "CLIntArray": ("const __global int*", cl.cl_mem),
            "CLImage": ("__read_only image2d_t", cl.cl_image),
        }

        # parse input func
        params = {}
        for i in input_func.__annotations__.items():
            params[i[0]] = i[1]

        source = input_func(*([None] * len(params)))

        pstr = []
        iparams = []
        for p in params.items():
            pname = p[1].__name__
            assert pname in cl_types

            pstr.append(cl_types[pname][0] + " " + p[0])
            pstr.append(",\n")

            iparams.append(cl_types[pname][1])
        pstr.append("__write_only image2d_t output")
        iparams = tuple(iparams)

        sampler = """
        const sampler_t sampler = \\
            CLK_NORMALIZED_COORDS_FALSE |
            CLK_ADDRESS_CLAMP_TO_EDGE |
            CLK_FILTER_NEAREST;
        const int gx = get_global_id(0), gy = get_global_id(1);
        const int2 loc = (int2)(gx, gy);
        const int width = get_image_width(output), height = get_image_height(output);
        """

        src = f"""
        #define READP(input,loc) read_imagef(input, sampler, loc)
        __kernel void {input_func.__name__}(
            {"".join(pstr)})
        {{
            float4 out;
            {sampler}
            {source}
            write_imagef(output, loc, out);
        }}
        """

        print(src)

        # new_kernel = cl_builder.build(
        #     name,
        #     f"""
        #     __kernel void {name}(
        #         const int s,
        #         const __global float *gc,
        #         __read_only image2d_t input,
        #         __write_only image2d_t output)
        #     {{
        #         {sampler}
        #         {source}
        #         write_imagef(output, (int2)(x,y), out);
        #     }}
        #     """,
        #     (cl.cl_int, cl.cl_mem, cl.cl_image, cl.cl_image),
        # )

        return None

    def image_kernel(self, input_func):
        """ Builds a OpenCL kernel and a Python function to run it """

        # TBD:  figure out if this is even a smart idea

        # self._decorator_builder(input_func)

        return None


cl_builder = CLDev()


# @cl_builder.image_kernel
# def gtscale(vee: CLInt, fio: CLImage):
#     return """
#     float4 px = READP(input, loc);
#     float g = px.x * 0.2989 + px.y * 0.5870 + px.z * 0.1140;
#     out = ((float4)(g, g, g, px.w));
#     """


def grayscale(ssp):
    src = """
    __kernel void grayscale(
        __read_only image2d_t A,
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


def linear_to_srgb(c, clamp=True):
    "linear sRGB to sRGB"
    assert c.dtype == np.float32
    srgb = np.where(c < 0.0031308, c * 12.92, 1.055 * np.pow(c, 1.0 / 2.4) - 0.055)
    if clamp:
        srgb[srgb > 1.0] = 1.0
        srgb[srgb < 0.0] = 0.0
    return srgb


def srgb_to_linear(c):
    "sRGB to linear sRGB"
    assert c.dtype == np.float32
    return np.where(c >= 0.04045, ((c + 0.055) / 1.055) ** 2.4, c / 12.92)


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

    # TODO: out of memory on big images and radius
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
                tempf = read_imagef(input, sampler, (int2)(gidx + u, gidy + v));
                diff_map = exp (
                    - (   POW2(center_pix.x - tempf.x)
                        + POW2(center_pix.y - tempf.y)
                        + POW2(center_pix.z - tempf.z))
                    * preserve);

                gaussian_weight = exp( - 0.5f * (POW2(u) + POW2(v)) / radius);
                weight = diff_map * gaussian_weight;

                accumulated += tempf * weight;
                count += weight;
            }
        }
        write_imagef(output, (int2)(gidx,gidy), accumulated / count);
    }
    """
    blr = cl_builder.build("bilateral", src, (cl.cl_float, cl.cl_float, cl.cl_image, cl.cl_image))
    img = cl_builder.new_image_from_numpy(pix)
    out = cl_builder.new_image(img.width, img.height)
    cl_builder.run_raw(blr, [radius, preserve], (img,), out)
    return out.to_numpy()


def median_filter(pix, radius):
    src = f"""
    #define RADIUS {radius}
    #define READP(x,y) read_imagef(input, sampler, (int2)(x, y))
    kernel void wirth_median_{radius}(
        __read_only image2d_t input,
        __write_only image2d_t output)
    {{
        const int x = get_global_id(0);
        const int y = get_global_id(1);
        const sampler_t sampler = \
            CLK_NORMALIZED_COORDS_FALSE |
            CLK_ADDRESS_CLAMP_TO_EDGE |
            CLK_FILTER_NEAREST;

        float rcol[4] = {{0.0, 0.0, 0.0, 1.0}};
        float a[RADIUS][RADIUS*RADIUS];

        for (int m = 0; m < RADIUS; m++) {{
            for (int n = 0; n < RADIUS; n++) {{
                float4 ta = READP(x + n - (RADIUS / 2), y + m - (RADIUS / 2));
                a[0][n+RADIUS*m] = ta.x;
                a[1][n+RADIUS*m] = ta.y;
                a[2][n+RADIUS*m] = ta.z;
            }}
        }}

        // Wirth median
        for (int z=0; z<RADIUS; z++) {{
            int k = (RADIUS*RADIUS)/2;
            int n = (RADIUS*RADIUS);
            int i,j,l,m;

            float val;

            l=0;
            m=n-1;
            while (l < m) {{
                val = a[z][k];
                i=l;
                j=m;
                do {{
                    while (a[z][i] < val) i++;
                    while (val < a[z][j]) j--;
                    if (i<=j) {{
                        float tmp = a[z][i];
                        a[z][i] = a[z][j];
                        a[z][j] = tmp;
                        i++; j--;
                    }}
                }} while (i <= j);
                if (j < k) l=i;
                if (k < i) m=j;
            }}

            rcol[z] = a[z][k];
        }}

        write_imagef(output, (int2)(x, y), (float4)(rcol[0], rcol[1], rcol[2], 1.0f));
    }}"""

    k = cl_builder.build("wirth_median_" + repr(radius), src, (cl.cl_image, cl.cl_image))
    img = cl_builder.new_image_from_numpy(pix)
    out = cl_builder.new_image(img.width, img.height)
    cl_builder.run_raw(k, [], (img,), out)
    return out.to_numpy()


def vectors_to_nmap(vectors):
    nmap = np.empty((vectors.shape[0], vectors.shape[1], 4), dtype=np.float32)
    vectors *= 0.5
    nmap[:, :, 0] = vectors[:, :, 0] + 0.5
    nmap[:, :, 1] = vectors[:, :, 1] + 0.5
    nmap[:, :, 2] = vectors[:, :, 2] + 0.5
    nmap[..., 3] = 1.0
    return nmap


def nmap_to_vectors(nmap):
    vectors = np.empty((nmap.shape[0], nmap.shape[1], 4), dtype=np.float32)
    vectors[..., 0] = nmap[..., 0] - 0.5
    vectors[..., 1] = nmap[..., 1] - 0.5
    vectors[..., 2] = nmap[..., 2] - 0.5
    vectors *= 2.0
    vectors[..., 3] = 1.0
    return vectors


if False:
    True
    # function rgb2hue(r, g, b) {
    #   r /= 255;
    #   g /= 255;
    #   b /= 255;
    #   var max = Math.max(r, g, b);
    #   var min = Math.min(r, g, b);
    #   var c   = max - min;
    #   var hue;
    #   if (c == 0) {
    #     hue = 0;
    #   } else {
    #     switch(max) {
    #       case r:
    #         var segment = (g - b) / c;
    #         var shift   = 0 / 60;       // R° / (360° / hex sides)
    #         if (segment < 0) {          // hue > 180, full rotation
    #           shift = 360 / 60;         // R° / (360° / hex sides)
    #         }
    #         hue = segment + shift;
    #         break;
    #       case g:
    #         var segment = (b - r) / c;
    #         var shift   = 120 / 60;     // G° / (360° / hex sides)
    #         hue = segment + shift;
    #         break;
    #       case b:
    #         var segment = (r - g) / c;
    #         var shift   = 240 / 60;     // B° / (360° / hex sides)
    #         hue = segment + shift;
    #         break;
    #     }
    #   }
    #   return hue * 60; // hue is in [0,6], scale it up
    # }

    # void YUVfromRGB(double& Y, double& U, double& V, const double R, const double G, const double B)
    # {
    #   Y =  0.257 * R + 0.504 * G + 0.098 * B +  16;
    #   U = -0.148 * R - 0.291 * G + 0.439 * B + 128;
    #   V =  0.439 * R - 0.368 * G - 0.071 * B + 128;
    # }
    # void RGBfromYUV(double& R, double& G, double& B, double Y, double U, double V)
    # {
    #   Y -= 16;
    #   U -= 128;
    #   V -= 128;
    #   R = 1.164 * Y             + 1.596 * V;
    #   G = 1.164 * Y - 0.392 * U - 0.813 * V;
    #   B = 1.164 * Y + 2.017 * U;
    # }


def normalize(pix, save_alpha=False):
    # TODO: HSL or Lab lightness normalization, maintain chroma
    if save_alpha:
        A = pix[..., 3]
    t = pix - np.min(pix)
    t = t / np.max(t)
    if save_alpha:
        t[..., 3] = A
    return t


def sharpen(pix, width, intensity):
    A = pix[..., 3]
    gas = gaussian_repeat(pix, width)
    pix += (pix - gas) * intensity
    pix[..., 3] = A
    return pix


def hi_pass(pix, s):
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
    "Make histogram into gaussian, save transform"
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
    "Make a Gaussianized histogram back to the original using the transform"
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

    steepness = 1.0

    # TODO: better vector calc, not just side pixels

    src = """
    #define READP(x,y) read_imagef(input, sampler, (int2)(x, y))
    kernel void height_to_normals(
        const float steepness,
        __read_only image2d_t input,
        __write_only image2d_t output
    )
    {
        int x = get_global_id(0);
        int y = get_global_id(1);
        const sampler_t sampler = \
            CLK_NORMALIZED_COORDS_FALSE |
            CLK_ADDRESS_CLAMP_TO_EDGE |
            CLK_FILTER_NEAREST;

        float4 pix = read_imagef(input, sampler, (int2)(x, y));

        // sobel operator
        float x_comp = READP(x-1, y).x
            +READP(x-1, y+1).x
            +READP(x-1, y-1).x
            - READP(x+1, y).x
            - READP(x+1, y+1).x
            - READP(x+1, y-1).x;

        float y_comp = READP(x, y-1).x
            + READP(x+1, y-1).x
            + READP(x-1, y-1).x
            - READP(x, y+1).x
            - READP(x+1, y+1).x
            - READP(x-1, y+1).x;

        float2 grad = (float2)(x_comp, y_comp);
        float l = length(grad);
        grad /= l;

        // from pythagoras
        float height;
        height = l < 1.0f ? sqrt(1.0f - l*l) : 0.0f;

        float4 out = (float4)(x_comp*0.5 + 0.5, y_comp*0.5 + 0.5, height*0.5 + 0.5, 1.0f);
        write_imagef(output, (int2)(x,y), out);
    }
    """
    blr = cl_builder.build("height_to_normals", src, (cl.cl_float, cl.cl_image, cl.cl_image))
    img = cl_builder.new_image_from_numpy(pix)
    out = cl_builder.new_image(img.width, img.height)
    assert steepness != 0.0
    cl_builder.run_raw(blr, [steepness], (img,), out)
    return out.to_numpy()


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


def gauss_seidel_cl(w, h, h2, target, inp, outp):
    src = """
    __kernel void curvature_to_height(
        const int i_width,
        const int i_height,
        const float step,
        __global const float *input,
        __global const float *target,
        __global float *output
    )
    {
        int x = get_global_id(0);
        int y = get_global_id(1);
        int loc = x + y * i_width;

        float t = 0.0f;

        t += x > 0 ? input[loc-1] : input[loc+(i_width-1)];
        t += y > 0 ? input[loc-i_width] : input[loc+(i_height-1)*i_width];

        t += x < i_width-1 ? input[loc+1] : input[loc-(i_width-1)];
        t += y < i_height-1 ? input[loc+i_width] : input[loc-(i_height-1)*i_width];

        t *= 0.25;
        t -= step * target[loc];
        output[loc] = t;
    }
    """
    cth = cl_builder.build(
        "curvature_to_height",
        src,
        (cl.cl_int, cl.cl_int, cl.cl_float, cl.cl_mem, cl.cl_mem, cl.cl_mem),
    )
    cl_builder.run_raw_buffer(cth, [w, h, h2, inp, target], outp, shape=(h, w))


def curvature_to_height(image, h2, iterations=2000):
    target = image[..., 0]

    w, h = target.shape[1], target.shape[0]
    f = cl_builder.to_buffer(target)

    ping = cl_builder.to_buffer(np.ones_like(target) * 0.5)
    pong = cl_builder.to_buffer(np.zeros_like(target))

    for ic in range(iterations):
        gauss_seidel_cl(w, h, h2, f, ping, pong)
        gauss_seidel_cl(w, h, h2, f, pong, ping)

    res_v, evt = cl.buffer_to_ndarray(cl_builder.queue, ping, like=image[..., 0])
    evt.wait()

    u = res_v
    u = -u
    u -= np.min(u)
    u /= np.max(u)

    return np.dstack([u, u, u, image[..., 3]])


def normals_to_height(image, iterations=2000, intensity=1.0):
    vectors = nmap_to_vectors(image)
    vectors *= intensity

    target = np.roll(vectors[..., 0], 1, axis=1)
    target -= np.roll(vectors[..., 0], -1, axis=1)
    target += np.roll(vectors[..., 1], 1, axis=0)
    target -= np.roll(vectors[..., 1], -1, axis=0)
    target *= 0.125

    w, h = target.shape[1], target.shape[0]
    f = cl_builder.to_buffer(target)

    ping = cl_builder.to_buffer(np.ones_like(target) * 0.5)
    pong = cl_builder.to_buffer(np.zeros_like(target))

    for ic in range(iterations):
        gauss_seidel_cl(w, h, 1.0, f, ping, pong)
        gauss_seidel_cl(w, h, 1.0, f, pong, ping)

    res_v, evt = cl.buffer_to_ndarray(cl_builder.queue, ping, like=image[..., 0])
    evt.wait()

    u = res_v
    u -= np.min(u)
    u /= np.max(u)

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


def dog(pix, a, b, threshold):
    "Difference of Gaussians with a threshold"
    size = max(a, b)
    gpix = grayscale(pix)
    res = (gaussian_repeat(gpix, a) - gaussian_repeat(gpix, b))[..., :3]
    tt = threshold / size
    # Xdog Winnemöller et al
    pix[..., :3] = np.where(tt >= res, 1.0, 1.0 + np.tanh(40.0 * (tt - res)))
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
    vectors = nmap_to_vectors(image)[..., :3]
    vectors = (vectors.T / np.linalg.norm(vectors, axis=2).T).T
    retarr = vectors_to_nmap(vectors)
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


# class TestPattern_IOP(image_ops.ImageOperatorGenerator):
#     def generate(self):
#         # self.props["order_a"] = bpy.props.StringProperty(name="Order A", default="RGBA")
#         # self.props["order_b"] = bpy.props.StringProperty(name="Order B", default="RBGa")
#         # self.props["direction"] = bpy.props.EnumProperty(
#         #     name="Direction", items=[("ATOB", "A to B", "", 1), ("BTOA", "B to A", "", 2)]
#         # )
#         self.prefix = "test_pattern"
#         self.info = "Test pattern"
#         self.category = "Basic"

#         def _pl(self, image, context):
#             # RED
#             image[:, 90:100, 0] = 1.0

#             # GREEN
#             image[90:100, :, 1] = 1.0

#             return image

#         self.payload = _pl


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
        self.category = "Dimensions"

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
        self.category = "Dimensions"

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


class DoG_IOP(image_ops.ImageOperatorGenerator):
    def generate(self):
        self.props["width_a"] = bpy.props.IntProperty(name="Width A", min=2, default=5)
        self.props["width_b"] = bpy.props.IntProperty(name="Width B", min=2, default=4)
        self.props["threshold"] = bpy.props.FloatProperty(
            name="Threshold", min=0.0, max=1.0, default=0.01
        )
        self.props["preserve"] = bpy.props.BoolProperty(name="Preserve", default=True)
        # self.props["intensity"] = bpy.props.FloatProperty(name="Intensity", min=0.0, default=1.0)
        self.prefix = "dog"
        self.info = "DoG"
        self.category = "Advanced"

        def _pl(self, image, context):
            t = image.copy()
            d = dog(image, self.width_a, self.width_b, self.threshold)
            if self.preserve:
                return t * d
            else:
                return d

        self.payload = _pl


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
        self.props["width"] = bpy.props.IntProperty(name="Width", min=1, default=20)
        # self.props["intensity"] = bpy.props.FloatProperty(name="Intensity", min=0.0, default=1.0)
        self.prefix = "gaussian_blur"
        self.info = "Does a Gaussian blur"
        self.category = "Filter"
        self.payload = lambda self, image, context: gaussian_repeat(image, self.width)


class Median_IOP(image_ops.ImageOperatorGenerator):
    def generate(self):
        # self.props["width"] = bpy.props.IntProperty(name="Width", min=3, max=9, default=3)
        self.props["width"] = bpy.props.EnumProperty(
            name="Width",
            items=[
                ("3", "3", "", 3),
                ("5", "5", "", 5),
                ("9", "9", "", 9),
                ("15", "15 (crash your computer)", "", 15),
            ],
            default="5",
        )
        self.prefix = "median_filter"
        self.info = "Median filter"
        self.category = "Filter"
        self.payload = lambda self, image, context: median_filter(image, int(self.width))


class Bilateral_IOP(image_ops.ImageOperatorGenerator):
    def generate(self):
        self.props["radius"] = bpy.props.FloatProperty(
            name="Radius", min=0.01, max=100.0, default=10.0
        )
        self.props["preserve"] = bpy.props.FloatProperty(
            name="Preserve", min=0.01, max=100.0, default=20.0
        )
        self.prefix = "bilateral"
        self.info = "Bilateral"
        self.category = "Filter"
        self.payload = lambda self, image, context: bilateral_cl(image, self.radius, self.preserve)


class HiPass_IOP(image_ops.ImageOperatorGenerator):
    def generate(self):
        self.props["width"] = bpy.props.IntProperty(name="Width", min=1, default=20)
        # self.props["intensity"] = bpy.props.FloatProperty(name="Intensity", min=0.0, default=1.0)
        self.prefix = "high_pass"
        self.info = "High pass"
        self.category = "Filter"
        self.payload = lambda self, image, context: hi_pass(image, self.width)


class HiPassBalance_IOP(image_ops.ImageOperatorGenerator):
    def generate(self):
        self.props["width"] = bpy.props.IntProperty(name="Width", min=1, default=50)
        self.props["zoom"] = bpy.props.IntProperty(name="Center slice", min=5, default=200)
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
        # self.props["grid"] = bpy.props.IntProperty(name="Grid subd", min=1, default=4)
        self.props["iterations"] = bpy.props.IntProperty(name="Iterations", min=10, default=200)
        self.prefix = "normals_to_height"
        self.info = "Normals to height"
        self.category = "Normals"
        self.payload = lambda self, image, context: normals_to_height(
            image, iterations=self.iterations
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


# class ImageToMaterial_IOP(image_ops.ImageOperatorGenerator):
#     def generate(self):
#         self.prefix = "image_to_material"
#         self.info = "Create magic material from image"
#         self.category = "Magic"
#         self.payload = lambda self, image, context: image_to_material(image)


# additional_classes = [BTT_InstallLibraries, BTT_AddonPreferences]
additional_classes = []

register, unregister = image_ops.create(locals(), additional_classes)
