"""
https://bottosson.github.io/posts/oklab/

Lab
L=lightness
a=green/red
b=blue/yellow

Lab to lightness, chroma, hue (LCh)
C=sqrt(a**2+b**2)
h=atan2(b,a)

LCh to Lab
a=C cos(h)
b=C sin(h)

"""
import numpy as np


def linear_srgb_to_oklab(c):
    cr, cg, cb = c[..., 0], c[..., 1], c[..., 2]
    l = (0.4122214708 * cr + 0.5363325363 * cg + 0.0514459929 * cb) ** (1 / 3)
    m = (0.2119034982 * cr + 0.6806995451 * cg + 0.1073969566 * cb) ** (1 / 3)
    s = (0.0883024619 * cr + 0.2817188376 * cg + 0.6299787005 * cb) ** (1 / 3)

    return np.dstack(
        [
            0.2104542553 * l + 0.7936177850 * m - 0.0040720468 * s,
            1.9779984951 * l - 2.4285922050 * m + 0.4505937099 * s,
            0.0259040371 * l + 0.7827717662 * m - 0.8086757660 * s,
            c[..., 3],
        ]
    )


def oklab_to_linear_srgb(c):
    cL, ca, cb = c[..., 0], c[..., 1], c[..., 2]
    l = (cL + 0.3963377774 * ca + 0.2158037573 * cb) ** 3.0
    m = (cL - 0.1055613458 * ca - 0.0638541728 * cb) ** 3.0
    s = (cL - 0.0894841775 * ca - 1.2914855480 * cb) ** 3.0

    return np.dstack(
        [
            +4.0767416621 * l - 3.3077115913 * m + 0.2309699292 * s,
            -1.2684380046 * l + 2.6097574011 * m - 0.3413193965 * s,
            -0.0041960863 * l - 0.7034186147 * m + 1.7076147010 * s,
            c[..., 3],
        ]
    )


def linear_to_srgb(c, clamp=True):
    "linear sRGB to sRGB"
    assert c.dtype == np.float32
    srgb = np.where(c < 0.0031308, c * 12.92, 1.055 * (c ** (1.0 / 2.4)) - 0.055)
    if clamp:
        srgb[srgb > 1.0] = 1.0
        srgb[srgb < 0.0] = 0.0
    return srgb


def srgb_to_linear(c):
    "sRGB to linear sRGB"
    assert c.dtype == np.float32
    return np.where(c >= 0.04045, ((c + 0.055) / 1.055) ** 2.4, c / 12.92)


def srgb_to_LCh(c):
    t = linear_srgb_to_oklab(srgb_to_linear(c))
    C = np.sqrt(t[..., 1] ** 2 + t[..., 2] ** 2)
    h = np.arctan2(t[..., 2], t[..., 1])
    L = t[..., 0]
    return np.dstack([L, C, h, t[..., 3]])


def LCh_to_srgb(c):
    L, C, h = c[..., 0], c[..., 1], c[..., 2]
    a = C * np.cos(h)
    b = C * np.sin(h)
    return linear_to_srgb(oklab_to_linear_srgb(np.dstack([L, a, b, c[..., 3]])))


def srgb_to_Lab(c):
    return linear_srgb_to_oklab(srgb_to_linear(c))


def Lab_to_srgb(c):
    return linear_to_srgb(oklab_to_linear_srgb(c))
