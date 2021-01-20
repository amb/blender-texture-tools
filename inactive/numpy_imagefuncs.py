import numpy as np


def gauss_curve(x):
    # gaussian with 0.01831 at last
    res = np.array([np.exp(-((i * (2 / x)) ** 2)) for i in range(-x, x + 1)], dtype=np.float32)
    res /= np.sum(res)
    return res


def vectors_to_nmap(vectors, nmap):
    vectors *= 0.5
    nmap[:, :, 0] = vectors[:, :, 0] + 0.5
    nmap[:, :, 1] = vectors[:, :, 1] + 0.5
    nmap[:, :, 2] = vectors[:, :, 2] + 0.5


def nmap_to_vectors(nmap):
    vectors = np.empty((nmap.shape[0], nmap.shape[1], 3), dtype=np.float32)
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
    return np.dstack([x, y, z])


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


def gaussian_repeat(pix, s):
    res = np.zeros(pix.shape, dtype=np.float32)
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


def bilateral(img_in, sigma_s, sigma_v, eps=1e-8):
    # gaussian
    gsi = lambda r2, sigma: np.exp(-0.5 * r2 / sigma ** 2)
    win_width = int(np.ceil(3 * sigma_s))
    wgt_sum = np.ones(img_in.shape) * eps
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
    r = np.zeros((ph, s * 2, 4), dtype=np.float32)
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

        pix[:, x, :] = np.sort(r, axis=1)[:, pick, :]

    temp = pix.copy()
    r = np.zeros((s * 2, pw, 4), dtype=np.float32)
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

        pix[y, :, :] = np.sort(r, axis=0)[pick, :, :]

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

