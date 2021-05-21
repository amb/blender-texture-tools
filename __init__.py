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
    "version": (0, 1, 29),
    "blender": (2, 81, 0),
}

import bpy  # noqa
import functools
import numpy as np
import random
from . import pycl as cl

rnd = random.random

from . import image_ops
import importlib

from .oklab import linear_to_srgb, srgb_to_linear

importlib.reload(image_ops)

import json
from .cl_abstraction import CLDev
from .toml_loader import load as cl_load

cl_builder = CLDev(0)
cl_nodes = cl_load(cl_builder)


def grayscale(ssp):
    out = cl_builder.new_image(ssp.shape[1], ssp.shape[0])
    cl_nodes["grayscale"].run([], [cl_builder.new_image_from_ndarray(ssp)], [out])
    return out.to_numpy()


def rgb_to_luminance(c):
    r = c[..., 0]
    g = c[..., 1]
    b = c[..., 2]
    return 0.2989 * r + 0.5870 * g + 0.1140 * b


@functools.lru_cache(maxsize=128)
def gauss_curve(x):
    # gaussian with 0.01831 at last
    res = np.array([np.exp(-((i * (2 / x)) ** 2)) for i in range(-x, x + 1)], dtype=np.float32)
    res /= np.sum(res)
    return res


def gaussian_repeat_cl(img, out, s):
    # TODO: store local pass & barrier(CLK_LOCAL_MEM_FENCE);
    cl_nodes["gaussian_h"].run([s], [img], [out])
    cl_nodes["gaussian_v"].run([s], [out], [img])
    return (img, out)


def gaussian_repeat(pix, s):
    "Separated gaussian for image. Over borders = wraparound"
    assert pix.dtype == np.float32
    img = cl_builder.new_image_from_ndarray(pix)
    out = cl_builder.new_image(img.width, img.height)
    gaussian_repeat_cl(img, out, s)
    return img.to_numpy()


def bilateral_cl(pix, radius, preserve):
    "Bilateral filter, OpenCL implementation"
    img = cl_builder.new_image_from_ndarray(pix)
    out = cl_builder.new_image(img.width, img.height)
    cl_nodes["bilateral"].run([radius, preserve], [img], [out])
    return out.to_numpy()


def image_gradient_cl(img, out):
    src = """
    #define READP(x,y) read_imagef(input, sampler, (int2)(x, y))
    kernel void image_flow(
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
        //grad = l > 0.0f ? grad/l : (float2)(0.0f, 0.0f);

        // from pythagoras
        float height;
        height = l < 1.0f ? sqrt(1.0f - l*l) : 0.0f;

        float4 out = (float4)(x_comp, y_comp, height, l);
        write_imagef(output, (int2)(x,y), out);
    }
    """
    blr = cl_builder.build("image_flow", src, (cl.cl_image, cl.cl_image))

    (out, img) = grayscale_cl(img, out)
    cl_builder.run(blr, [], (out,), img)
    return (img, out)


def directional_blur_cl(pix, radius, preserve):
    "Directional bilateral filter, OpenCL implementation"

    original = np.copy(pix)

    img = cl_builder.new_image_from_ndarray(pix)
    out = cl_builder.new_image(img.width, img.height)

    (grad, l0) = image_gradient_cl(img, out)
    (grad, l0) = gaussian_repeat_cl(grad, l0, 2)

    src = """
    #define POW2(a) ((a) * (a))
    #define F4_ABS(v) ((float4)(fabs(v.x), fabs(v.y), fabs(v.z), 1.0f))
    kernel void guided_bilateral(
        const float radius,
        const float preserve,
        __read_only image2d_t gradient,
        __read_only image2d_t input,
        __write_only image2d_t output
    )
    {
        int gidx = get_global_id(0);
        int gidy = get_global_id(1);
        float2 gvec = (float2)(gidx, gidy);
        const sampler_t sampler = \
            CLK_NORMALIZED_COORDS_FALSE |
            CLK_ADDRESS_CLAMP_TO_EDGE |
            CLK_FILTER_NEAREST;
        const sampler_t sampler_f = \
            CLK_NORMALIZED_COORDS_FALSE |
            CLK_ADDRESS_CLAMP_TO_EDGE |
            CLK_FILTER_LINEAR;

        int n_radius = ceil(radius);

        float4 center_pix = read_imagef(input, sampler, (int2)(gidx, gidy));
        float4 grad = read_imagef(gradient, sampler, (int2)(gidx, gidy));

        float4 acc_A = 0.0f;
        float4 acc_B = 0.0f;
        float4 tempf = 0.0f;
        float  count = 0.0f;
        float  diff_map, gaussian_weight, weight;

        float dx = grad.x;
        float dy = grad.y;

        // along tangent flow
        float2 v_vec = (float2)(-dy, dx);
        // against tangent flow
        float2 u_vec = (float2)(dx, dy);

        weight = 1.0f;

        for (float v = -n_radius; v <= n_radius; v=v+1.0f) {
            float2 loc = gvec + (v_vec * v) + (float2)(0.5f, 0.5f);

            tempf = read_imagef(input, sampler_f, loc);

            diff_map = exp (
                - (   POW2(center_pix.x - tempf.x)
                    + POW2(center_pix.y - tempf.y)
                    + POW2(center_pix.z - tempf.z))
                * preserve);

            gaussian_weight = exp(-0.5f * (POW2(v)) / radius);
            weight = diff_map * gaussian_weight;

            // weight = gaussian_weight;
            // weight = 1.0;

            acc_A += tempf * weight;
            count += weight;
        }

        float4 res = acc_A/fabs(count);
        res.w = 1.0f;
        write_imagef(output, (int2)(gidx,gidy), res);
        //write_imagef(output, (int2)(gidx,gidy), F4_ABS(res));
    }
    """
    blr = cl_builder.build(
        "guided_bilateral", src, (cl.cl_float, cl.cl_float, cl.cl_image, cl.cl_image, cl.cl_image)
    )
    l1 = cl_builder.new_image_from_ndarray(original)
    cl_builder.run(blr, [radius, preserve], (grad, l1), l0)
    for _ in range(8):
        cl_builder.run(blr, [radius, preserve], (grad, l0), l1)
        cl_builder.run(blr, [radius, preserve], (grad, l1), l0)

    return l0.to_numpy()


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
    img = cl_builder.new_image_from_ndarray(pix)
    out = cl_builder.new_image(img.width, img.height)
    cl_builder.run(k, [], (img,), out)
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
        const int width,
        const int height,
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
        float hg;
        hg = l < 1.0f ? sqrt(1.0f - l*l) : 0.0f;

        float4 out = (float4)(x_comp*0.5 + 0.5, y_comp*0.5 + 0.5, hg*0.5 + 0.5, 1.0f);
        write_imagef(output, (int2)(x,y), out);
    }
    """
    blr = cl_builder.build(
        "height_to_normals", src, (cl.cl_int, cl.cl_int, cl.cl_float, cl.cl_image, cl.cl_image)
    )
    img = cl_builder.new_image_from_ndarray(pix)
    out = cl_builder.new_image(img.width, img.height)
    assert steepness != 0.0
    cl_builder.run(blr, [steepness], [img.data], [out.data], shape=img.shape)
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
    # TODO: fix name
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
    assert w % 8 == 0, "Image width must be divisible by 8"
    assert h % 8 == 0, "Image width must be divisible by 8"
    # cl_builder.run_buffer(cth, [w, h, h2, inp, target], outp, shape=(h, w))
    # kernel, params, inputs, outputs
    cl_builder.run(cth, [h2], [inp, target], [outp], shape=(h, w))


def curvature_to_height(image, h2, iterations=2000):
    target = image[..., 0]
    # TODO: from grayscale, not just 1 component

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


def normals_to_height(image, iterations=2000, intensity=1.0, step=1.0):
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
        gauss_seidel_cl(w, h, step, f, ping, pong)
        gauss_seidel_cl(w, h, step, f, pong, ping)

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

            # make compatible with CL calcs
            w = w - (w % 8)
            h = h - (h % 8)

            xt = w // 2
            yt = w // 2

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


class TextureToNormals_IOP(image_ops.ImageOperatorGenerator):
    def generate(self):
        self.props["high_freq"] = bpy.props.FloatProperty(
            name="High frequency", min=0.0, max=1.0, default=0.1
        )
        self.props["mid_freq"] = bpy.props.FloatProperty(
            name="Mid frequency", min=0.0, max=1.0, default=0.2
        )
        self.props["low_freq"] = bpy.props.FloatProperty(
            name="Low frequency", min=0.0, max=1.0, default=0.7
        )

        self.prefix = "texture_to_normals"
        self.info = "Texture to Normals"
        self.category = "Advanced"

        def _pl(self, image, context):
            # imgg = gaussian_repeat(image, 4)
            g = grayscale(image)
            b = curvature_to_height(g, 0.5, iterations=100)
            c = curvature_to_height(g, 0.5, iterations=1000)

            d = normals_simple(
                g * self.high_freq + b * self.mid_freq + c * self.low_freq, "Luminance"
            )
            d = normals_to_height(d, iterations=500, step=0.5)
            d = normals_simple(d, "Luminance")
            # d = srgb_to_linear(d)
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


class DirectionalBilateral_IOP(image_ops.ImageOperatorGenerator):
    def generate(self):
        self.props["radius"] = bpy.props.FloatProperty(
            name="Radius", min=0.01, max=100.0, default=10.0
        )
        self.props["preserve"] = bpy.props.FloatProperty(
            name="Preserve", min=0.01, max=100.0, default=20.0
        )
        self.prefix = "directional_blur"
        self.info = "Directional bilateral"
        self.category = "Advanced"
        self.payload = lambda self, image, context: directional_blur_cl(
            image, self.radius, self.preserve
        )


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
        self.category = "Seamless"
        self.payload = lambda self, image, context: gimpify(image)


class KnifeSeamless_IOP(image_ops.ImageOperatorGenerator):
    def generate(self):
        self.prefix = "knife_seamless"
        self.info = "Optimal knife cut into seamless"
        self.category = "Seamless"

        self.props["step"] = bpy.props.IntProperty(name="Step", min=1, max=16, default=3)
        self.props["margin"] = bpy.props.IntProperty(name="Margin", min=4, max=256, default=40)
        self.props["smooth"] = bpy.props.IntProperty(
            name="Cut smoothing", min=0, max=64, default=16
        )
        self.props["constrain"] = bpy.props.FloatProperty(
            name="Middle constraint", min=0.0, max=15.0, default=2.0
        )
        # self.props["square"] = bpy.props.BoolProperty(name="To square", default=False)

        def diffblocks(a, b, constrain_middle):
            l = len(a)
            if constrain_middle >= 0.0 and constrain_middle <= 15.0:
                penalty = np.abs(((np.arange(l) - (l - 1) * 0.5) * 2.0 / (l - 1))) ** (
                    constrain_middle + 1.0
                )
            else:
                penalty = 0.0
            # assert np.all(penalty) >= 0.0
            # assert np.all(penalty) <= 1.0
            # TODO: adding power might be better
            # return rgb_to_luminance(np.abs(a - b)) ** 2.0 + penalty
            return rgb_to_luminance(np.abs(a - b)) + penalty

        def findmin(ar, loc, step):
            minloc = loc
            lar = len(ar)
            for x in range(-step, step + 1):
                if loc + x >= 0 and loc + x < lar and ar[loc + x] < ar[minloc]:
                    minloc = loc + x
            return minloc

        def copy_to_v(image, img_orig, sr, rv, y):
            w = image.shape[1]
            hw = w // 2
            image[y, hw - sr : hw - sr + rv, :] = img_orig[y, w - 2 * sr : w - 2 * sr + rv, :]
            r2 = sr * 2 - rv
            image[y, hw + sr - r2 : hw + sr, :] = img_orig[y, sr * 2 - r2 : sr * 2, :]

        def copy_to_h(image, img_orig, sr, rv, y):
            w = image.shape[0]
            hw = w // 2
            image[hw - sr : hw - sr + rv, y, :] = img_orig[w - 2 * sr : w - 2 * sr + rv, y, :]
            r2 = sr * 2 - rv
            image[hw + sr - r2 : hw + sr, y, :] = img_orig[sr * 2 - r2 : sr * 2, y, :]

        def _pl(self, image, context):
            h, w = image.shape[0], image.shape[1]

            v_margin = self.margin
            h_margin = self.margin
            step = self.step
            m_constraint = 16.0 - self.constrain

            # if self.square:
            #     max_space = min(h, w)
            #     h_margin += w - max_space
            #     v_margin += h - max_space

            # new_width = w
            # new_height = h

            # -- vertical cut
            if self.smooth > 0:
                smoothed = gaussian_repeat(image, self.smooth)
            else:
                smoothed = image.copy()
            img_orig = image.copy()
            hw = w // 2

            # right on left
            image[:, : hw + h_margin, :] = img_orig[:, hw - h_margin :, :]

            # left on right
            image[:, hw - h_margin :, :] = img_orig[:, : hw + h_margin, :]

            abr = diffblocks(
                smoothed[0, -(2 * h_margin) :, :], smoothed[0, : h_margin * 2, :], m_constraint
            )
            rv = np.argmin(abr)
            for y in range(h):
                abr = diffblocks(
                    smoothed[y, -(2 * h_margin) :, :], smoothed[y, : h_margin * 2, :], m_constraint
                )
                rv = findmin(abr, rv, step)
                copy_to_v(image, img_orig, h_margin, rv, y)

            # -- horizontal cut
            if self.smooth > 0:
                smoothed = gaussian_repeat(image, self.smooth)
            else:
                smoothed = image.copy()
            img_orig = image.copy()
            hw = h // 2
            image[: hw + v_margin, ...] = img_orig[hw - v_margin :, ...]
            image[hw - v_margin :, ...] = img_orig[: hw + v_margin, ...]

            abr = diffblocks(
                smoothed[-(2 * v_margin) :, 0, :], smoothed[: v_margin * 2, 0, :], m_constraint
            )
            rv = np.argmin(abr)
            for x in range(w):
                abr = diffblocks(
                    smoothed[-(2 * v_margin) :, x, :], smoothed[: v_margin * 2, x, :], m_constraint
                )
                rv = findmin(abr, rv, step)
                copy_to_h(image, img_orig, v_margin, rv, x)

            return image[v_margin:-v_margin, h_margin:-h_margin]

        self.payload = _pl


class HistogramSeamless_IOP(image_ops.ImageOperatorGenerator):
    def generate(self):
        self.prefix = "histogram_seamless"
        self.info = "Seamless histogram blending"
        self.category = "Seamless"

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


def read_material_nodes_to_json(mat):
    node_tree = mat.node_tree
    d_nodes = {}

    # TODO: calc max and min bounding box, save center value
    for n in node_tree.nodes.values():
        d_temp = {}

        d_temp["bl_idname"] = n.bl_idname
        d_temp["dimensions"] = tuple(i for i in n.dimensions)
        d_temp["location"] = tuple(i for i in n.location)

        n_inputs = []
        for i in n.inputs.values():
            if len(i.links) == 0:
                if hasattr(i, "default_value"):
                    if i.bl_idname == "NodeSocketFloatFactor":
                        n_inputs.append((i.name, "float", i.default_value.real))
                    # elif i.bl_idname == "NodeSocketVector":
                    #     n_inputs.append((i.name, 'vector', i.default_value.real))
                    else:
                        n_inputs.append((i.name, "unknown", None))
                else:
                    n_inputs.append((i.name, "no_default_value", None))
            else:
                # TODO: input links should always be length 1
                for l in i.links:
                    n_inputs.append((i.name, "node", l.from_node.name, l.from_socket.name))
        d_temp["inputs"] = n_inputs

        # n_outputs = []
        # for i in n.outputs.values():
        #     if len(i.links) == 0:
        #         continue
        #     for l in i.links:
        #         n_outputs.append(l.from_node.name)
        # d_temp["outputs"] = n_outputs

        d_nodes[n.name] = d_temp

    # return d_nodes
    return json.dumps(d_nodes, indent=4)


def overwrite_material_from_json(mat, json_in):
    d_nodes = json.loads(json_in)

    # Enable 'Use nodes':
    mat.use_nodes = True
    nodes = mat.node_tree.nodes

    # Remove existing nodes
    for node in nodes:
        nodes.remove(node)

    # Create nodes and store them into dict
    new_nodes = {}
    for nk, nv in d_nodes.items():
        node = nodes.new(nv["bl_idname"])
        node.location = nv["location"]
        new_nodes[nk] = node

    # Link nodes (use only inputs data)
    for nk, nv in d_nodes.items():
        for l in nv["inputs"]:
            # name, type, value, (from_socket.name)
            if l[2] is not None:
                if l[1] == "node":
                    mat.node_tree.links.new(
                        new_nodes[nk].inputs[l[0]], new_nodes[l[2]].outputs[l[3]]
                    )
                elif l[1] == "float":
                    new_nodes[nk].inputs[l[0]].default_value = l[2]

        # test_group.links.new(node_add.inputs[1], node_less.outputs[0])

    return d_nodes


class SaveMaterial_IOP(image_ops.ImageOperatorGenerator):
    def generate(self):
        self.prefix = "save_material"
        self.info = "Material to JSON"
        self.category = "Materials"

        def _pl(self, image, context):
            json_out = read_material_nodes_to_json(bpy.data.materials[0])

            mat_name = "test"
            mat = bpy.data.materials.get(mat_name) or bpy.data.materials.new(mat_name)

            d_nodes = overwrite_material_from_json(mat, json_out)

            import pprint
            pp = pprint.PrettyPrinter(indent=4)
            pp.pprint(d_nodes)

            return image

        self.payload = _pl


# additional_classes = [BTT_InstallLibraries, BTT_AddonPreferences]
additional_classes = []

register, unregister = image_ops.create(locals(), additional_classes)
