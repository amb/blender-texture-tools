from . import pycl as cl
import numpy as np
from ctypes import c_void_p as void_p


def min_ptwo(val, pt):
    "Gives the minimum divisionally aligned value for input value"
    assert val > 0
    assert pt > 0
    return ((val - 1) // pt + 1) * pt


class CLType:
    def __init__(self, cldev, w, h, simple=False):
        self.cldev = cldev
        self.simple = simple
        self.original_width = w
        self.original_height = h
        self.width = min_ptwo(w, 8)
        self.height = min_ptwo(h, 8)
        self.shape = (self.width, self.height)


class CLImage(CLType):
    def __init__(self, cldev, w, h):
        super().__init__(cldev, w, h, simple=False)
        assert self.width % 8 == 0, "Image width must be divisible by 8"
        assert self.height % 8 == 0, "Image height must be divisible by 8"
        # Default memflags are CL_MEM_READ_WRITE
        self.data = cl.clCreateImage2D(
            cldev.ctx, self.width, self.height, imgformat=cldev.image_format
        )

    def from_numpy(self, source):
        h, w = source.shape[0], source.shape[1]
        if h % 8 != 0 or w % 8 != 0:
            padding = np.zeros(
                (min_ptwo(h, 8), min_ptwo(w, 8), source.shape[2]), dtype=source.dtype
            )
            padding[:h, :w] = source[:, :]
            source = padding
        ary = np.ascontiguousarray(source)
        if ary.__array_interface__["strides"]:
            raise ValueError("I don't know how to handle strided arrays yet.")
        ptr = void_p(ary.__array_interface__["data"][0])
        # print(self.data._height, self.data._width, ary.shape)
        evt = cl.clEnqueueWriteImage(
            self.cldev.queue, self.data, ptr, (0, 0, 0), (ary.shape[1], ary.shape[0], 1), 0, 0
        )
        evt.wait()

    def to_numpy(self):
        "See pycl.py for buffer_to_ndarray"
        out = np.empty((self.height, self.width, 4), dtype=np.float32)
        assert out.flags.contiguous, "Don't know how to write non-contiguous yet."
        evt = cl.clEnqueueReadImage(
            self.cldev.queue,
            self.data,
            void_p(out.__array_interface__["data"][0]),
            (0, 0, 0),
            (self.width, self.height, 1),
            self.width * 4 * 4,
            0,
        )
        evt.wait()
        return out[: self.original_height, : self.original_width]


class CLFloat2D(CLType):
    def __init__(self, cldev, a):
        super().__init__(cldev, a.shape[1], a.shape[0], simple=False)
        assert a.dtype == np.float32
        res, evt = cl.buffer_from_ndarray(self.cldev.queue, a)
        evt.wait()
        self.data = res

    def to_numpy(self):
        res, evt = cl.buffer_to_ndarray(
            self.cldev.queue, self.data, dtype=np.float32, shape=self.shape
        )
        evt.wait()
        return res


class CLDev:
    """ OpenCL device class for 2D-3D image array processing """

    def __init__(self, dev_id):
        self.ctx = cl.clCreateContext()
        for dev in self.ctx.devices:
            print(dev.name)

        dvids = cl.clGetDeviceIDs()
        # print(len(dvids))
        d = dvids[dev_id]
        cl.clGetDeviceInfo(d, cl.cl_device_info.CL_DEVICE_NAME)
        cl.clGetDeviceInfo(d, cl.cl_device_info.CL_DEVICE_TYPE)
        print(f"Device {dev_id} available:", d.available)
        print("Max work item sizes:", d.max_work_item_sizes)

        print("Supported image formats for RGBA:")
        self.supported_rgba = [
            i.image_channel_data_type
            for i in cl.clGetSupportedImageFormats(self.ctx)
            if i.image_channel_order == cl.cl_channel_order.CL_RGBA
        ]
        print(self.supported_rgba)

        # Ensure we have RGBA float32
        assert (
            cl.cl_channel_type.CL_FLOAT in self.supported_rgba
        ), "Your device doesn't support CL_FLOAT for RGBA"

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
            prg = cl.clCreateProgramWithSource(self.ctx, source)
            b = prg.build()
            kernel = b[name]
        except KeyError as e:
            print(e)
            prg_info = cl.cl_program_build_info
            print(cl.clGetProgramBuildInfo(prg, prg_info.CL_PROGRAM_BUILD_STATUS, 0))
            print(cl.clGetProgramBuildInfo(prg, prg_info.CL_PROGRAM_BUILD_OPTIONS, 0))
            print(cl.clGetProgramBuildInfo(prg, prg_info.CL_PROGRAM_BUILD_LOG, 0))
            raise

        # kernel.argtypes = (cl.cl_int, cl.cl_int, cl.cl_int, cl.cl_mem, cl.cl_mem, cl.cl_mem)
        kernel.argtypes = argtypes
        self.kernels[name] = kernel
        return kernel

    def new_image(self, width, height):
        return CLImage(self, width, height)

    def new_image_from_ndarray(self, arr):
        "shape[0]=height, shape[1]=width"
        i = CLImage(self, arr.shape[1], arr.shape[0])
        i.from_numpy(arr)
        return i

    def to_buffer(self, narray):
        gc_c, gc_e = cl.buffer_from_ndarray(self.queue, narray)
        gc_e.wait()
        return gc_c

    def run(self, kernel, params, inputs, outputs, shape=None):
        "Run CL kernel on params. Multiple in, single out"
        assert len(outputs) > 0, "No outputs given"
        assert shape is not None, "Invalid kernel shape"
        assert type(shape[1]) == int, "Invalid kernel shape"
        assert type(shape[0]) == int, "Invalid kernel shape"
        assert shape in [(a.width, a.height) for a in outputs], "Kernel shape not in outputs"
        assert shape[1] % 8 == 0, "Input image height must be divisible by 8"
        assert shape[0] % 8 == 0, "Input image width must be divisible by 8"
        # width, height, params, inputs, outputs
        run_evt = kernel(shape[0], shape[1], *params, *inputs, *outputs).on(
            self.queue, offset=(0, 0), gsize=shape, lsize=(8, 8)
        )
        run_evt.wait()
