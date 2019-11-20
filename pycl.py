#!/usr/bin/env python
"""

Brief usage example:

>>> from array import array
>>> source = '''
... kernel void mxplusb(float m, global float *x, float b, global float *out) {
...     int i = get_global_id(0);
...     out[i] = m*x[i]+b;
... }
... '''
>>> ctx = clCreateContext()
>>> queue = clCreateCommandQueue(ctx)
>>> program = clCreateProgramWithSource(ctx, source).build()
>>> kernel = program['mxplusb']
>>> kernel.argtypes = (cl_float, cl_mem, cl_float, cl_mem)
>>> x = array('f', range(100))
>>> x_buf, in_evt = buffer_from_pyarray(queue, x, blocking=False)
>>> y_buf = x_buf.empty_like_this()
>>> run_evt = kernel(2, x_buf, 5, y_buf).on(queue, len(x), wait_for=in_evt)
>>> y, evt = buffer_to_pyarray(queue, y_buf, wait_for=run_evt, like=x)
>>> evt.wait()
>>> y[0:10]
array('f', [5.0, 7.0, 9.0, 11.0, 13.0, 15.0, 17.0, 19.0, 21.0, 23.0])

For Numpy users, see :func:`buffer_from_ndarray` and
:func:`buffer_to_ndarray`.

Additionally, if run as a script, will print out a summary
of your platforms and devices.

Most of the C typedefs are available as subclasses of
Python ctypes datatypes. The spelling might be slightly
different.

The various enumeration and bitfield types have attributes
representing their defined constants (e.g.
:const:`~cl_device_type.CL_DEVICE_TYPE_GPU`). These
constants are also available at the module level, in case
you can't remember what type
:const:`~cl_command_execution_status.CL_QUEUED` is supposed
to be. They are all somewhat magical in that they'll
make a reasonable effort to pretty-print themselves:

    >>> cl_device_type.CL_DEVICE_TYPE_GPU | cl_device_type.CL_DEVICE_TYPE_CPU
    CL_DEVICE_TYPE_CPU | CL_DEVICE_TYPE_GPU
    >>> cl_mem_info(0x1100)
    CL_MEM_TYPE

The types representing various object-like datastructures
often have attributes so that you can view their infos
without needing to call the appropriate ``clGetThingInfo``
function. They may have other methods and behaviors.

One last note about the datatypes: despite any appearance
of magic and high-level function, these are just ctypes
objects. It is entirely possible for you to assign things
to the :attr:`value` attribute of the enum/bitfield
constants or of object-like items. Overwriting constants
and clobbering pointers is generally a bad idea, though,
so you should probably avoid it. (I tried vetoing
assignment to .value, but PyPy didn't like that.
So you're on your own.)

Wrapped OpenCL functions have their usual naming convention
(``clDoSomething``). These are't the naked C function
pointers - you will find that the argument lists,
return types, and exception raising are more in line with
Python. Check the docstrings. That said, you can refer to
the function pointer itself with the wrapped function's
:attr:`call` attribute, which is how the functions
themselves do it. The function pointer itself has argument
type, return type, and error checking added in the usual
ctypes manner.

The list of wrapped functions is *very* incomplete. Feel
free to contribute if you need a function that hasn't been
wrapped yet.

There are currently no plans to provide wrappers for OpenCL
extensions (like OpenGL interop). Maybe later.

"""
# Copyright (c) 2011 Ken Watford
#
# Permission is hereby granted, free of charge, to any person
# obtaining a copy of this software and associated documentation
# files (the "Software"), to deal in the Software without
# restriction, including without limitation the rights to use,
# copy, modify, merge, publish, distribute, sublicense, and/or
# sell copies of the Software, and to permit persons to whom the
# Software is furnished to do so, subject to the following conditions:
#
# The above copyright notice and this permission notice shall be
# included in all copies or substantial portions of the Software.
#
# THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND,
# EXPRESS OR IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES
# OF MERCHANTABILITY, FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT.
# IN NO EVENT SHALL THE AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR
# ANY CLAIM, DAMAGES OR OTHER LIABILITY, WHETHER IN AN ACTION OF
# CONTRACT, TORT OR OTHERWISE, ARISING FROM, OUT OF OR IN CONNECTION
# WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE SOFTWARE.
#
# tl;dr - MIT license.

__version__ = "0.1.1"

import ctypes
import _ctypes
from ctypes import (
    c_size_t as size_t,
    c_void_p as void_p,
    c_char_p as char_p,
    POINTER as P,
    byref,
    sizeof,
    pointer,
    cast,
    create_string_buffer,
)
import os
import sys
from warnings import warn
from array import array

try:
    import numpy as np
except ImportError:
    np = None
    pass


class cl_sampler(void_p):
    pass


class cl_char(ctypes.c_int8):
    pass


class cl_uchar(ctypes.c_uint8):
    pass


class cl_short(ctypes.c_int16):
    pass


class cl_ushort(ctypes.c_uint16):
    pass


class cl_int(ctypes.c_int32):
    pass


class cl_uint(ctypes.c_uint32):
    pass


class cl_long(ctypes.c_int64):
    pass


class cl_ulong(ctypes.c_uint64):
    pass


class cl_half(ctypes.c_uint16):
    pass


class cl_float(ctypes.c_float):
    pass


class cl_double(ctypes.c_double):
    pass


class cl_bool(cl_uint):
    pass


class cl_uenum(cl_uint):
    # Base class for the various unsigned int
    # constants defined in OpenCL.
    def __eq__(self, other):
        if not isinstance(other, self.__class__):
            return False
        else:
            return self.value == other.value

    def __ne__(self, other):
        return not (self == other)

    def __hash__(self):
        return self.value.__hash__()

    def __repr__(self):
        by_value = self.__class__._by_value
        names = []
        if self in by_value:
            return by_value[self]
        elif self.value:
            return "UNKNOWN(0%x)" % self.value
        else:
            return "NONE"


class cl_enum(cl_int):
    # Base class for various signed int enums.
    def __eq__(self, other):
        if not isinstance(other, self.__class__):
            return False
        else:
            return self.value == other.value

    def __ne__(self, other):
        return not (self == other)

    def __hash__(self):
        return self.value.__hash__()

    def __repr__(self):
        by_value = self.__class__._by_value
        names = []
        if self in by_value:
            return by_value[self]
        elif self.value:
            return "UNKNOWN(0x%x)" % self.value
        else:
            return "NONE"


class cl_bitfield(cl_ulong):
    # Base class for bitfield values found in OpenCL.
    # Bitwise operations for combining flags are supported.
    def __or__(self, other):
        assert isinstance(other, self.__class__)
        return self.__class__(self.value | other.value)

    def __and__(self, other):
        assert isinstance(other, self.__class__)
        return self.__class__(self.value & other.value)

    def __xor__(self, other):
        assert isinstance(other, self.__class__)
        return self.__class__(self.value ^ other.value)

    def __not__(self):
        return self.__class__(~self.value)

    def __contains__(self, other):
        assert isinstance(other, self.__class__)
        return (self.value & other.value) == other.value

    def __hash__(self):
        return self.value.__hash__()

    def __eq__(self, other):
        if not isinstance(other, self.__class__):
            return False
        else:
            return self.value == other.value

    def __ne__(self, other):
        return not (self == other)

    def __repr__(self):
        by_value = self.__class__._by_value
        names = []
        if self in by_value:
            return by_value[self]
        for val in by_value:
            if val in self:
                names.append(by_value[val])
        if names:
            return " | ".join(names)
        elif self.value:
            return "UNKNOWN(0x%x)" % self.value
        else:
            return "NONE"


class cl_device_type(cl_bitfield):
    """
    Bitfield used by :func:`clCreateContextFromType` to
    create a context from one or more matching device types.

    See also :attr:`cl_device.type` and :func:`clGetDeviceInfo`

    """

    CL_DEVICE_TYPE_DEFAULT = 1 << 0
    CL_DEVICE_TYPE_CPU = 1 << 1
    CL_DEVICE_TYPE_GPU = 1 << 2
    CL_DEVICE_TYPE_ACCELERATOR = 1 << 3
    CL_DEVICE_TYPE_ALL = 0xFFFFFFFF


class cl_errnum(cl_enum):
    """
    A status code returned by most OpenCL functions.
    Exceptions exist for each error code and will be
    raised in the event that the code is flagged by
    any wrapper function. The exception names are formed
    by removing the 'CL', title-casing the words, removing
    the underscores, and appending 'Error' to the end.
    Some of these are a little redundant, like
    :exc:`BuildProgramFailureError`.

    And no, there is no :exc:`SuccessError`.
    """

    CL_SUCCESS = 0
    CL_DEVICE_NOT_FOUND = -1
    CL_DEVICE_NOT_AVAILABLE = -2
    CL_COMPILER_NOT_AVAILABLE = -3
    CL_MEM_OBJECT_ALLOCATION_FAILURE = -4
    CL_OUT_OF_RESOURCES = -5
    CL_OUT_OF_HOST_MEMORY = -6
    CL_PROFILING_INFO_NOT_AVAILABLE = -7
    CL_MEM_COPY_OVERLAP = -8
    CL_IMAGE_FORMAT_MISMATCH = -9
    CL_IMAGE_FORMAT_NOT_SUPPORTED = -10
    CL_BUILD_PROGRAM_FAILURE = -11
    CL_MAP_FAILURE = -12
    CL_MISALIGNED_SUB_BUFFER_OFFSET = -13
    CL_EXEC_STATUS_ERROR_FOR_EVENTS_IN_WAIT_LIST = -14
    CL_INVALID_VALUE = -30
    CL_INVALID_DEVICE_TYPE = -31
    CL_INVALID_PLATFORM = -32
    CL_INVALID_DEVICE = -33
    CL_INVALID_CONTEXT = -34
    CL_INVALID_QUEUE_PROPERTIES = -35
    CL_INVALID_COMMAND_QUEUE = -36
    CL_INVALID_HOST_PTR = -37
    CL_INVALID_MEM_OBJECT = -38
    CL_INVALID_IMAGE_FORMAT_DESCRIPTOR = -39
    CL_INVALID_IMAGE_SIZE = -40
    CL_INVALID_SAMPLER = -41
    CL_INVALID_BINARY = -42
    CL_INVALID_BUILD_OPTIONS = -43
    CL_INVALID_PROGRAM = -44
    CL_INVALID_PROGRAM_EXECUTABLE = -45
    CL_INVALID_KERNEL_NAME = -46
    CL_INVALID_KERNEL_DEFINITION = -47
    CL_INVALID_KERNEL = -48
    CL_INVALID_ARG_INDEX = -49
    CL_INVALID_ARG_VALUE = -50
    CL_INVALID_ARG_SIZE = -51
    CL_INVALID_KERNEL_ARGS = -52
    CL_INVALID_WORK_DIMENSION = -53
    CL_INVALID_WORK_GROUP_SIZE = -54
    CL_INVALID_WORK_ITEM_SIZE = -55
    CL_INVALID_GLOBAL_OFFSET = -56
    CL_INVALID_EVENT_WAIT_LIST = -57
    CL_INVALID_EVENT = -58
    CL_INVALID_OPERATION = -59
    CL_INVALID_GL_OBJECT = -60
    CL_INVALID_BUFFER_SIZE = -61
    CL_INVALID_MIP_LEVEL = -62
    CL_INVALID_GLOBAL_WORK_SIZE = -63
    CL_INVALID_PROPERTY = -64

    CL_INVALID_GL_SHAREGROUP_REFERENCE_KHR = -1000


class cl_platform_info(cl_uenum):
    """
    The set of possible parameter names used
    with the :func:`clGetPlatformInfo` function.
    """

    CL_PLATFORM_PROFILE = 0x0900
    CL_PLATFORM_VERSION = 0x0901
    CL_PLATFORM_NAME = 0x0902
    CL_PLATFORM_VENDOR = 0x0903
    CL_PLATFORM_EXTENSIONS = 0x0904


class cl_device_info(cl_uenum):
    """
    The set of possible parameter names used
    with the :func:`clGetDeviceInfo` function.
    """

    CL_DEVICE_TYPE = 0x1000
    CL_DEVICE_VENDOR_ID = 0x1001
    CL_DEVICE_MAX_COMPUTE_UNITS = 0x1002
    CL_DEVICE_MAX_WORK_ITEM_DIMENSIONS = 0x1003
    CL_DEVICE_MAX_WORK_GROUP_SIZE = 0x1004
    CL_DEVICE_MAX_WORK_ITEM_SIZES = 0x1005
    CL_DEVICE_PREFERRED_VECTOR_WIDTH_CHAR = 0x1006
    CL_DEVICE_PREFERRED_VECTOR_WIDTH_SHORT = 0x1007
    CL_DEVICE_PREFERRED_VECTOR_WIDTH_INT = 0x1008
    CL_DEVICE_PREFERRED_VECTOR_WIDTH_LONG = 0x1009
    CL_DEVICE_PREFERRED_VECTOR_WIDTH_FLOAT = 0x100A
    CL_DEVICE_PREFERRED_VECTOR_WIDTH_DOUBLE = 0x100B
    CL_DEVICE_MAX_CLOCK_FREQUENCY = 0x100C
    CL_DEVICE_ADDRESS_BITS = 0x100D
    CL_DEVICE_MAX_READ_IMAGE_ARGS = 0x100E
    CL_DEVICE_MAX_WRITE_IMAGE_ARGS = 0x100F
    CL_DEVICE_MAX_MEM_ALLOC_SIZE = 0x1010
    CL_DEVICE_IMAGE2D_MAX_WIDTH = 0x1011
    CL_DEVICE_IMAGE2D_MAX_HEIGHT = 0x1012
    CL_DEVICE_IMAGE3D_MAX_WIDTH = 0x1013
    CL_DEVICE_IMAGE3D_MAX_HEIGHT = 0x1014
    CL_DEVICE_IMAGE3D_MAX_DEPTH = 0x1015
    CL_DEVICE_IMAGE_SUPPORT = 0x1016
    CL_DEVICE_MAX_PARAMETER_SIZE = 0x1017
    CL_DEVICE_MAX_SAMPLERS = 0x1018
    CL_DEVICE_MEM_BASE_ADDR_ALIGN = 0x1019
    CL_DEVICE_MIN_DATA_TYPE_ALIGN_SIZE = 0x101A
    CL_DEVICE_SINGLE_FP_CONFIG = 0x101B
    CL_DEVICE_GLOBAL_MEM_CACHE_TYPE = 0x101C
    CL_DEVICE_GLOBAL_MEM_CACHELINE_SIZE = 0x101D
    CL_DEVICE_GLOBAL_MEM_CACHE_SIZE = 0x101E
    CL_DEVICE_GLOBAL_MEM_SIZE = 0x101F
    CL_DEVICE_MAX_CONSTANT_BUFFER_SIZE = 0x1020
    CL_DEVICE_MAX_CONSTANT_ARGS = 0x1021
    CL_DEVICE_LOCAL_MEM_TYPE = 0x1022
    CL_DEVICE_LOCAL_MEM_SIZE = 0x1023
    CL_DEVICE_ERROR_CORRECTION_SUPPORT = 0x1024
    CL_DEVICE_PROFILING_TIMER_RESOLUTION = 0x1025
    CL_DEVICE_ENDIAN_LITTLE = 0x1026
    CL_DEVICE_AVAILABLE = 0x1027
    CL_DEVICE_COMPILER_AVAILABLE = 0x1028
    CL_DEVICE_EXECUTION_CAPABILITIES = 0x1029
    CL_DEVICE_QUEUE_PROPERTIES = 0x102A
    CL_DEVICE_NAME = 0x102B
    CL_DEVICE_VENDOR = 0x102C
    CL_DRIVER_VERSION = 0x102D
    CL_DEVICE_PROFILE = 0x102E
    CL_DEVICE_VERSION = 0x102F
    CL_DEVICE_EXTENSIONS = 0x1030
    CL_DEVICE_PLATFORM = 0x1031
    CL_DEVICE_DOUBLE_FP_CONFIG = 0x1032
    CL_DEVICE_HALF_FP_CONFIG = 0x1033
    CL_DEVICE_PREFERRED_VECTOR_WIDTH_HALF = 0x1034
    CL_DEVICE_HOST_UNIFIED_MEMORY = 0x1035
    CL_DEVICE_NATIVE_VECTOR_WIDTH_CHAR = 0x1036
    CL_DEVICE_NATIVE_VECTOR_WIDTH_SHORT = 0x1037
    CL_DEVICE_NATIVE_VECTOR_WIDTH_INT = 0x1038
    CL_DEVICE_NATIVE_VECTOR_WIDTH_LONG = 0x1039
    CL_DEVICE_NATIVE_VECTOR_WIDTH_FLOAT = 0x103A
    CL_DEVICE_NATIVE_VECTOR_WIDTH_DOUBLE = 0x103B
    CL_DEVICE_NATIVE_VECTOR_WIDTH_HALF = 0x103C
    CL_DEVICE_OPENCL_C_VERSION = 0x103D


class cl_device_fp_config(cl_bitfield):
    """
    One of the possible return types from :func:`clGetDeviceInfo`.
    Bitfield identifying the floating point capabilities of the device.
    """

    CL_FP_DENORM = 1 << 0
    CL_FP_INF_NAN = 1 << 1
    CL_FP_ROUND_TO_NEAREST = 1 << 2
    CL_FP_ROUND_TO_ZERO = 1 << 3
    CL_FP_ROUND_TO_INF = 1 << 4
    CL_FP_FMA = 1 << 5
    CL_FP_SOFT_FLOAT = 1 << 6


class cl_device_mem_cache_type(cl_uenum):
    """
    One of the possible return types from :func:`clGetDeviceInfo`.
    Describes the nature of the device's cache, if any.
    """

    CL_NONE = 0x0
    CL_READ_ONLY_CACHE = 0x1
    CL_READ_WRITE_CACHE = 0x2


class cl_device_local_mem_type(cl_uenum):
    """
    One of the possible return types from :func:`clGetDeviceInfo`.
    Describes where 'local' memory lives in the device.
    Presumably, :const:`~cl_device_local_mem_type.CL_GLOBAL` means
    the device's local memory lives in the same address space as its
    global memory.
    """

    CL_LOCAL = 0x1
    CL_GLOBAL = 0x2


class cl_device_exec_capabilities(cl_bitfield):
    """
    One of the possible return types from :func:`clGetDeviceInfo`.
    Bitfield identifying what kind of kernels can be executed.
    All devices can execute OpenCL C kernels, but some have their
    own native kernel types as well.
    """

    CL_EXEC_KERNEL = 1 << 0
    CL_EXEC_NATIVE_KERNEL = 1 << 1


class cl_command_queue_properties(cl_bitfield):
    """
    Bitfield representing the properties of a command queue.
    """

    CL_QUEUE_OUT_OF_ORDER_EXEC_MODE_ENABLE = 1 << 0
    CL_QUEUE_PROFILING_ENABLE = 1 << 1


class cl_context_properties(void_p):
    """
    If you find yourself looking at an array of these and
    need to make any sense of them... good luck! It's a list
    of key-value pairs, null-terminated. The keys are unsigned ints
    representing enum constants.
    :const:`~cl_context_info.CL_CONTEXT_PLATFORM` (0x1084)
    is the most common one you'll see. I believe the rest are
    parts of extensions, such as the OpenGL interop extension.

    The meaning of the odd elements depends entirely on the
    enum that came just before it. In the case of
    :const:`~cl_context_info.CL_CONTEXT_PLATFORM`,
    the value represents a pointer to a cl_platform object.
    """

    pass


class cl_context_info(cl_uenum):
    """
    Parameter names understood by :func:`clGetContextInfo`.

    Note that :const:`cl_context_inf.CL_CONTEXT_PLATFORM` does not technically
    belong here, and the C-level code won't accept it. The wrapped
    version of :func:`clGetContextInfo` will, however, recognize it
    and extract the appropriate value from the context's
    properties list.
    """

    CL_CONTEXT_REFERENCE_COUNT = 0x1080
    CL_CONTEXT_DEVICES = 0x1081
    CL_CONTEXT_PROPERTIES = 0x1082
    CL_CONTEXT_NUM_DEVICES = 0x1083
    CL_CONTEXT_PLATFORM = 0x1084

    # FIXME: right place for these?
    CL_GL_CONTEXT_KHR = 0x2008
    CL_EGL_DISPLAY_KHR = 0x2009
    CL_GLX_DISPLAY_KHR = 0x200A
    CL_WGL_HDC_KHR = 0x200B
    CL_CGL_SHAREGROUP_KHR = 0x200C


class cl_command_queue_info(cl_uenum):
    """
    Parameter names understood by :func:`clGetCommandQueueInfo`
    """

    CL_QUEUE_CONTEXT = 0x1090
    CL_QUEUE_DEVICE = 0x1091
    CL_QUEUE_REFERENCE_COUNT = 0x1092
    CL_QUEUE_PROPERTIES = 0x1093


class cl_channel_order(cl_uenum):
    """
    Indicates the meanings of vector fields in an image.
    """

    CL_R = 0x10B0
    CL_A = 0x10B1
    CL_RG = 0x10B2
    CL_RA = 0x10B3
    CL_RGB = 0x10B4
    CL_RGBA = 0x10B5
    CL_BGRA = 0x10B6
    CL_ARGB = 0x10B7
    CL_INTENSITY = 0x10B8
    CL_LUMINANCE = 0x10B9
    CL_Rx = 0x10BA
    CL_RGx = 0x10BB
    CL_RGBx = 0x10BC


class cl_channel_type(cl_uenum):
    """
    Indicates the type and size of image channels.
    """

    CL_SNORM_INT8 = 0x10D0
    CL_SNORM_INT16 = 0x10D1
    CL_UNORM_INT8 = 0x10D2
    CL_UNORM_INT16 = 0x10D3
    CL_UNORM_SHORT_565 = 0x10D4
    CL_UNORM_SHORT_555 = 0x10D5
    CL_UNORM_INT_101010 = 0x10D6
    CL_SIGNED_INT8 = 0x10D7
    CL_SIGNED_INT16 = 0x10D8
    CL_SIGNED_INT32 = 0x10D9
    CL_UNSIGNED_INT8 = 0x10DA
    CL_UNSIGNED_INT16 = 0x10DB
    CL_UNSIGNED_INT32 = 0x10DC
    CL_HALF_FLOAT = 0x10DD
    CL_FLOAT = 0x10DE


class cl_mem_flags(cl_bitfield):
    """
    Bitfield used when constructing a memory object.
    Indicates both the read/write status of the memory as
    well as how the memory interacts with whatever host
    pointer was provided. See the OpenCL docs_ for
    :func:`clCreateBuffer` for more information.

    .. _docs: http://www.khronos.org/registry/cl/sdk/1.1/docs/man/xhtml/clCreateBuffer.html
    """

    CL_MEM_READ_WRITE = 1 << 0
    CL_MEM_WRITE_ONLY = 1 << 1
    CL_MEM_READ_ONLY = 1 << 2
    CL_MEM_USE_HOST_PTR = 1 << 3
    CL_MEM_ALLOC_HOST_PTR = 1 << 4
    CL_MEM_COPY_HOST_PTR = 1 << 5


class cl_mem_object_type(cl_uenum):
    """
    Possible return type for :func:`clGetMemObjectInfo`.
    Indicates the type of the memory object.
    """

    CL_MEM_OBJECT_BUFFER = 0x10F0
    CL_MEM_OBJECT_IMAGE2D = 0x10F1
    CL_MEM_OBJECT_IMAGE3D = 0x10F2


class cl_mem_info(cl_uenum):
    """
    Parameter names accepted by :func:`clGetMemObjectInfo`
    """

    CL_MEM_TYPE = 0x1100
    CL_MEM_FLAGS = 0x1101
    CL_MEM_SIZE = 0x1102
    CL_MEM_HOST_PTR = 0x1103
    CL_MEM_MAP_COUNT = 0x1104
    CL_MEM_REFERENCE_COUNT = 0x1105
    CL_MEM_CONTEXT = 0x1106
    CL_MEM_ASSOCIATED_MEMOBJECT = 0x1107
    CL_MEM_OFFSET = 0x1108


class cl_image_info(cl_uenum):
    """
    Parameter names accepted by :func:`clGetImageInfo`
    """

    CL_IMAGE_FORMAT = 0x1110
    CL_IMAGE_ELEMENT_SIZE = 0x1111
    CL_IMAGE_ROW_PITCH = 0x1112
    CL_IMAGE_SLICE_PITCH = 0x1113
    CL_IMAGE_WIDTH = 0x1114
    CL_IMAGE_HEIGHT = 0x1115
    CL_IMAGE_DEPTH = 0x1116


class cl_buffer_create_type(cl_uenum):
    """
    Parameter type for :func:`clCreateSubBuffer` that indicates
    how the subbuffer will be described.

    The only supported value is
    :const:`~cl_buffer_create_type.CL_BUFFER_CREATE_TYPE_REGION`,
    which indicates the subbuffer will be a contiguous region as
    defined by a :class:`cl_buffer_region` struct.
    """

    CL_BUFFER_CREATE_TYPE_REGION = 0x1220


class cl_addressing_mode(cl_uenum):
    """
    Addressing mode for sampler objects.
    Returned by :func:`clGetSamplerInfo`.
    """

    CL_ADDRESS_NONE = 0x1130
    CL_ADDRESS_CLAMP_TO_EDGE = 0x1131
    CL_ADDRESS_CLAMP = 0x1132
    CL_ADDRESS_REPEAT = 0x1133
    CL_ADDRESS_MIRRORED_REPEAT = 0x1134


class cl_filter_mode(cl_uenum):
    """
    Filter mode for sampler objects.
    Returned by :func:`clGetSamplerInfo`.
    """

    CL_FILTER_NEAREST = 0x1140
    CL_FILTER_LINEAR = 0x1141


class cl_sampler_info(cl_uenum):
    """
    Parameter names for :func:`clGetSamplerInfo`.
    """

    CL_SAMPLER_REFERENCE_COUNT = 0x1150
    CL_SAMPLER_CONTEXT = 0x1151
    CL_SAMPLER_NORMALIZED_COORDS = 0x1152
    CL_SAMPLER_ADDRESSING_MODE = 0x1153
    CL_SAMPLER_FILTER_MODE = 0x1154


class cl_map_flags(cl_bitfield):
    """
    Read/write flags used for applying memory mappings
    to memory objects. See :func:`clEnqueueMapBuffer`
    and :func:`clEnqueueMapImage`.
    """

    CL_MAP_READ = 1 << 0
    CL_MAP_WRITE = 1 << 1


class cl_program_info(cl_uenum):
    """
    Parameter names for :func:`clGetProgramInfo`
    """

    CL_PROGRAM_REFERENCE_COUNT = 0x1160
    CL_PROGRAM_CONTEXT = 0x1161
    CL_PROGRAM_NUM_DEVICES = 0x1162
    CL_PROGRAM_DEVICES = 0x1163
    CL_PROGRAM_SOURCE = 0x1164
    CL_PROGRAM_BINARY_SIZES = 0x1165
    CL_PROGRAM_BINARIES = 0x1166


class cl_program_build_info(cl_uenum):
    """
    Parameter names for :func:`clGetProgramBuildInfo`
    """

    CL_PROGRAM_BUILD_STATUS = 0x1181
    CL_PROGRAM_BUILD_OPTIONS = 0x1182
    CL_PROGRAM_BUILD_LOG = 0x1183


class cl_build_status(cl_enum):
    """
    Returned by :func:`clGetProgramBuildInfo`.
    Indicates build status for the program on the
    specified device.
    """

    CL_BUILD_SUCCESS = 0
    CL_BUILD_NONE = -1
    CL_BUILD_ERROR = -2
    CL_BUILD_IN_PROGRESS = -3


class cl_kernel_info(cl_uenum):
    """
    Parameter names for :func:`clGetKernelInfo`
    """

    CL_KERNEL_FUNCTION_NAME = 0x1190
    CL_KERNEL_NUM_ARGS = 0x1191
    CL_KERNEL_REFERENCE_COUNT = 0x1192
    CL_KERNEL_CONTEXT = 0x1193
    CL_KERNEL_PROGRAM = 0x1194


class cl_kernel_work_group_info(cl_uenum):
    """
    Parameter names for :func:`clGetKernelWorkGroupInfo`
    """

    CL_KERNEL_WORK_GROUP_SIZE = 0x11B0
    CL_KERNEL_COMPILE_WORK_GROUP_SIZE = 0x11B1
    CL_KERNEL_LOCAL_MEM_SIZE = 0x11B2
    CL_KERNEL_PREFERRED_WORK_GROUP_SIZE_MULTIPLE = 0x11B3
    CL_KERNEL_PRIVATE_MEM_SIZE = 0x11B4


class cl_event_info(cl_uenum):
    """
    Parameter names for :func:`clGetEventInfo`
    """

    CL_EVENT_COMMAND_QUEUE = 0x11D0
    CL_EVENT_COMMAND_TYPE = 0x11D1
    CL_EVENT_REFERENCE_COUNT = 0x11D2
    CL_EVENT_COMMAND_EXECUTION_STATUS = 0x11D3
    CL_EVENT_CONTEXT = 0x11D4


class cl_command_type(cl_uenum):
    """
    Command types recorded on events and returned by
    :func:`clGetEventInfo`.
    """

    CL_COMMAND_NDRANGE_KERNEL = 0x11F0
    CL_COMMAND_TASK = 0x11F1
    CL_COMMAND_NATIVE_KERNEL = 0x11F2
    CL_COMMAND_READ_BUFFER = 0x11F3
    CL_COMMAND_WRITE_BUFFER = 0x11F4
    CL_COMMAND_COPY_BUFFER = 0x11F5
    CL_COMMAND_READ_IMAGE = 0x11F6
    CL_COMMAND_WRITE_IMAGE = 0x11F7
    CL_COMMAND_COPY_IMAGE = 0x11F8
    CL_COMMAND_COPY_IMAGE_TO_BUFFER = 0x11F9
    CL_COMMAND_COPY_BUFFER_TO_IMAGE = 0x11FA
    CL_COMMAND_MAP_BUFFER = 0x11FB
    CL_COMMAND_MAP_IMAGE = 0x11FC
    CL_COMMAND_UNMAP_MEM_OBJECT = 0x11FD
    CL_COMMAND_MARKER = 0x11FE
    CL_COMMAND_ACQUIRE_GL_OBJECTS = 0x11FF
    CL_COMMAND_RELEASE_GL_OBJECTS = 0x1200
    CL_COMMAND_READ_BUFFER_RECT = 0x1201
    CL_COMMAND_WRITE_BUFFER_RECT = 0x1202
    CL_COMMAND_COPY_BUFFER_RECT = 0x1203
    CL_COMMAND_USER = 0x1204


class cl_command_execution_status(cl_uenum):
    """
    Status of the command associated with an event,
    returned by :func:`clGetEventInfo`.
    """

    CL_COMPLETE = 0x0
    CL_RUNNING = 0x1
    CL_SUBMITTED = 0x2
    CL_QUEUED = 0x3


class cl_profiling_info(cl_uenum):
    """
    Parameter names for :func:`clGetEventProfilingInfo`.
    Indicates the point in time of the event's life that
    should be queried.
    """

    CL_PROFILING_COMMAND_QUEUED = 0x1280
    CL_PROFILING_COMMAND_SUBMIT = 0x1281
    CL_PROFILING_COMMAND_START = 0x1282
    CL_PROFILING_COMMAND_END = 0x1283


class cl_image_format(ctypes.Structure):
    """
    Represents image formats. See :func:`clCreateImage2D`.

    .. attribute:: image_channel_order

        A :class:`cl_channel_order` value

    .. attribute:: image_channel_data_type

        A :class:`cl_channel_type` value
    """

    _fields_ = [
        ("image_channel_order", cl_channel_order),
        ("image_channel_data_type", cl_channel_type),
    ]

    def __repr__(self):
        return "%s(%s, %s)" % (
            self.__class__.__name__,
            self.image_channel_order,
            self.image_channel_data_type,
        )


class cl_buffer_region(ctypes.Structure):
    """
    A buffer region has two fields: :attr:`origin` and :attr:`size`.
    Both are of type :c:type:`size_t`.

    See :func:`clCreateSubBuffer` for usage.
    """

    _fields_ = [("origin", size_t), ("size", size_t)]

    def __repr__(self):
        return "%s(%s, %s)" % (self.__class__.__name__, int(self.origin), int(self.size))


# Take care of some last-minute meta stuff.
# I would use metaclasses to handle this, but Python 3 expects different
# metaclass syntax, and I didn't want to have to run it through 2to3.
# I would use class decorators to handle this, but Python 2.5 doesn't
# understand them. And it's easier to iterate through like this than to
# write in the "manual class decorator" line after each class.

# For enums and bitfields, do magic. Each type gets a registry of the
# names and values of their defined elements, to support pretty printing.
# Further, each of the class variables (which are defined using ints) is
# upgraded to be a member of the class in question.
# Additionally, each of the constants is copied into the module scope.
for cls in cl_enum.__subclasses__() + cl_uenum.__subclasses__() + cl_bitfield.__subclasses__():
    if cls.__name__ not in globals():
        # Don't apply this to types that ctypes makes automatically,
        # like the _be classes. Doing so will overwrite the declared
        # constants at global scope, which is really weird.
        continue
    cls._by_name = dict()
    cls._by_value = dict()
    if not cls.__doc__:
        cls.__doc__ = ""
    for name, value in cls.__dict__.items():
        if isinstance(value, int):
            obj = cls(value)
            setattr(cls, name, obj)
            cls._by_name[name] = obj
            cls._by_value[obj] = name
            globals()[name] = obj
            cls.__doc__ += (
                """
            .. attribute:: %s
            """
                % name
            )
    cls.NONE = cls(0)
# cleanup
del cls
del name
del value
del obj

# Generate exception tree
class OpenCLError(Exception):
    """
    The base class from which all of the (generated)
    OpenCL errors are descended. These exceptions
    correspond to the :class:`cl_errnum` status codes.
    """

    pass


cl_errnum._errors = dict()
for name, val in cl_errnum._by_name.items():
    if name == "CL_SUCCESS":
        continue  # Sorry, no SuccessError
    errname = "".join(y.title() for y in name.split("_")[1:]) + "Error"
    errtype = type(errname, (OpenCLError,), {"value": val})
    globals()[errname] = errtype
    cl_errnum._errors[val] = errtype
del name
del val
del errname
del errtype

# Locate and load the shared library.

_dll_filename = os.getenv("PYCL_OPENCL")
if not _dll_filename:
    try:
        from ctypes.util import find_library as _find_library

        _dll_filename = _find_library("OpenCL")
    except ImportError:
        pass
if _dll_filename:
    try:
        _dll = ctypes.cdll.LoadLibrary(_dll_filename)
    except:
        raise RuntimeError("Could not load OpenCL dll: %s" % _dll_filename)
else:
    if os.environ.get("READTHEDOCS", None) == "True":
        # Don't care if we can load the DLL on RTD.
        _dll = None
    else:
        raise RuntimeError(
            "Could not locate OpenCL dll. Please set the PYCL_OPENCL environment variable to its full path."
        )


def _result_errcheck(result, func, args):
    """
    For use in the errcheck attribute of a ctypes function wrapper.

    Most OpenCL functions return a cl_errnum. This checks it for
    an error code and raises an appropriate exception if it finds one.

    This is the default error checker when using _wrapdll
    """
    if result != cl_errnum.CL_SUCCESS:
        raise cl_errnum._errors[result]
    return result


def _lastarg_errcheck(result, func, args):
    """
    For use in the errcheck attribute of a ctypes function wrapper.

    Most OpenCL functions that don't return their error code expect
    you to provide a pointer for it as the last argument. To use this,
    the last argument of the call should be something like `byref(cl_errnum())`
    """
    lastarg = args[-1]
    if hasattr(lastarg, "_obj"):
        status = lastarg._obj
    else:
        # In PyPy, the byref object is an actual pointer.
        status = lastarg[0]
    if status != cl_errnum.CL_SUCCESS:
        raise cl_errnum._errors[status]
    return result


def _wrapdll(*argtypes, **kw):
    """
    Decorator used to simplify wrapping OpenCL functions a bit.

    The positional arguments represent the ctypes argument types the
    C-level function expects, and will be used to do argument type checking.

    If a `res` keyword argument is given, it represents the C-level
    function's expected return type. The default is `cl_errnum`.

    If an `err` keyword argument is given, it represents an error checker
    that should be run after low-level calls. The `_result_errcheck` and
    `_lastarg_errcheck` functions should be sufficient for most OpenCL
    functions. `_result_errcheck` is the default value.

    The decorated function should have the same name as the underlying
    OpenCL function, since the function name is used to do the lookup. The
    C-level function pointer will be stored in the decorated function's
    `call` attribute, and should be used by the decorated function to
    perform the actual call(s). The wrapped function is otherwise untouched.

    If no C-level function by this name is found in the OpenCL library
    (perhaps it's version 1.0?) the decorator will discard the original
    function. The replacement simply raises NotImplementedError if called.

    .. todo::
        Reconsider this last bit. Maybe let the wrapper compensate for the
        lack of function pointer.
    """

    def dowrap(f):
        try:
            wrapped_func = getattr(_dll, f.__name__)
        except:

            def badfunc(*args, **kw):
                raise NotImplementedError(
                    "Function %s not present " "in this version of OpenCL" % f.__name__
                )

            wrapped_func = badfunc
        wrapped_func.argtypes = argtypes
        res = kw.pop("res", cl_errnum)
        wrapped_func.restype = res
        err = kw.pop("err", _result_errcheck)
        wrapped_func.errcheck = err
        f.call = wrapped_func
        return f

    return dowrap


#################
# Event Objects #
#################


class cl_event(void_p):
    """
    An OpenCL Event object. Returned by functions that add commands
    to a :class:`cl_command_queue`, and often accepted (singly or in
    lists) by the ``wait_for`` argument of these functions to impose
    ordering.

    Use :meth:`wait` to wait for a particular event to complete, or
    :func:`clWaitForEvents` to wait for several of them at once.

    These objects participate in OpenCL's reference counting scheme.
    """

    @property
    def queue(self):
        """The queue this event was emitted from."""
        try:
            return self._queue
        except AttributeError:
            return clGetEventInfo(self, cl_event_info.CL_EVENT_COMMAND_QUEUE)

    @property
    def context(self):
        """The context this event exists within."""
        try:
            return self._context
        except AttributeError:
            return clGetEventInfo(self, cl_event_info.CL_EVENT_CONTEXT)

    @property
    def type(self):
        """
        The type of command this event is linked to.
        See :class:`cl_command_type`.
        """
        try:
            return self._type
        except AttributeError:
            return clGetEventInfo(self, cl_event_info.CL_EVENT_COMMAND_TYPE)

    @property
    def status(self):
        """
        Execution status of the command the event is linked to.
        See :class:`cl_command_exec_status`.
        """
        return clGetEventInfo(self, cl_event_info.CL_EVENT_COMMAND_EXECUTION_STATUS)

    @property
    def reference_count(self):
        """Reference count for OpenCL garbage collection."""
        return clGetEventInfo(self, cl_event_info.CL_EVENT_REFERENCE_COUNT)

    def wait(self):
        """Blocks until this event completes."""
        clWaitForEvents(self)

    def __repr__(self):
        try:
            return "<cl_event %s (%s) >" % (self.type, self.status)
        except:
            return "<cl_event 0x%x>" % (self.value or 0)

    def __del__(self):
        try:
            if self:
                clReleaseEvent(self)
        except:
            pass


def _make_event_array(events):
    if not events:
        return (0, None)
    if isinstance(events, cl_event):
        events = [events]
    valid_events = [e for e in events if e]
    numevents = len(valid_events)
    event_array = (cl_event * numevents)()
    for i, e in enumerate(valid_events):
        event_array[i] = e
    return (numevents, event_array)


@_wrapdll(cl_uint, P(cl_event))
def clWaitForEvents(*events):
    """
    Accepts several events and blocks until they all complete.
    """
    if not events:
        return
    nevents, event_array = _make_event_array(events)
    if nevents:
        clWaitForEvents.call(nevents, event_array)


@_wrapdll(cl_event)
def clRetainEvent(event):
    clRetainEvent.call(event)


@_wrapdll(cl_event)
def clReleaseEvent(event):
    clReleaseEvent.call(event)


@_wrapdll(cl_event, cl_event_info, size_t, void_p, P(size_t))
def clGetEventInfo(event, param_name):
    """
    :param param_name: An instance of :class:`cl_event_info`.

    Event information can be more easily obtained by querying
    the properties of the event object, which in turn will
    call this function.
    """
    if param_name == cl_event_info.CL_EVENT_COMMAND_QUEUE:
        try:
            return event._queue
        except AttributeError:
            param_value = cl_command_queue()
            clGetEventInfo.call(event, param_name, sizeof(param_value), byref(param_value), None)
            clRetainCommandQueue(param_value)
            event._queue = param_value
            return param_value
    elif param_name == cl_event_info.CL_EVENT_CONTEXT:
        try:
            return event._context
        except AttributeError:
            param_value = cl_context()
            clGetEventInfo.call(event, param_name, sizeof(param_value), byref(param_value), None)
            clRetainContext(param_value)
            event._context = param_value
            return param_value
    elif param_name == cl_event_info.CL_EVENT_COMMAND_TYPE:
        try:
            return event._type
        except AttributeError:
            param_value = cl_command_type()
            clGetEventInfo.call(event, param_name, sizeof(param_value), byref(param_value), None)
            event._type = param_value
            return param_value
    elif param_name == cl_event_info.CL_EVENT_COMMAND_EXECUTION_STATUS:
        param_value = cl_command_execution_status()
        clGetEventInfo.call(event, param_name, sizeof(param_value), byref(param_value), None)
        return param_value
    elif param_name == cl_event_info.CL_EVENT_REFERENCE_COUNT:
        param_value = cl_uint()
        clGetEventInfo.call(event, param_name, sizeof(param_value), byref(param_value), None)
        return int(param_value.value)
    else:
        raise ValueError("Unknown parameter type: %s" % param_name)


####################
# Platform Objects #
####################


class cl_platform(void_p):
    """
    Represents an OpenCL Platform.
    Should not be directly instantiated by users of PyCL.
    Use :func:`clGetPlatformIDs` or the :attr:`platform` attribute of
    some OpenCL objects to procure a cl_platform instance.
    """

    def __repr__(self):
        try:
            return "<cl_platform '%s'>" % self.name
        except:
            return "<cl_platform 0x%x>" % (self.value or 0)

    @property
    def name(self):
        """
        Name of the platform. (str)
        """
        return clGetPlatformInfo(self, cl_platform_info.CL_PLATFORM_NAME)

    @property
    def vendor(self):
        """
        Vendor that distributes the platform. (str)
        """
        return clGetPlatformInfo(self, cl_platform_info.CL_PLATFORM_VENDOR)

    @property
    def version(self):
        """
        Platform version. Likely starts with 'OpenCL 1.1'. (str)
        """
        return clGetPlatformInfo(self, cl_platform_info.CL_PLATFORM_VERSION)

    @property
    def extensions(self):
        """
        Platform extensions supported. (list of str)
        Note that devices have their own set of extensions which
        should be inspected separately.
        """
        return clGetPlatformInfo(self, cl_platform_info.CL_PLATFORM_EXTENSIONS).split()

    @property
    def profile(self):
        """
        One of 'FULL_PROFILE' or 'EMBEDDED_PROFILE'.
        """
        return clGetPlatformInfo(self, cl_platform_info.CL_PLATFORM_PROFILE)

    @property
    def devices(self):
        """
        All devices available on this platform. (list of cl_device)
        """
        return clGetDeviceIDs(self)


@_wrapdll(cl_uint, P(cl_platform), P(cl_uint))
def clGetPlatformIDs():
    """
    Returns a list of :class:`cl_platform` objects available on your system.
    It should probably not be possible for this list to be empty if
    you are able to call this function.

    >>> clGetPlatformIDs()  # doctest: +ELLIPSIS
    (<cl_platform '...'>...)
    """
    num_platforms = cl_uint()
    clGetPlatformIDs.call(0, None, byref(num_platforms))
    n = num_platforms.value
    if n > 0:
        platform_array = (cl_platform * n)()
        clGetPlatformIDs.call(n, platform_array, None)
        return tuple(x for x in platform_array)
    else:
        return ()


@_wrapdll(cl_platform, cl_platform_info, size_t, void_p, P(size_t))
def clGetPlatformInfo(platform, param_name):
    """
    :param param_name: One of :class:`cl_platform_info`.

    :class:`cl_platform` objects have attributes that will call this for
    you, so you should probably use those instead of calling this directly.

    >>> plat = clGetPlatformIDs()[0]
    >>> clGetPlatformInfo(plat, cl_platform_info.CL_PLATFORM_VERSION) # doctest: +ELLIPSIS
    'OpenCL ...'
    >>> plat.version # doctest: +ELLIPSIS
    'OpenCL ...'

    Note that :const:`~cl_platform_info.CL_PLATFORM_EXTENSIONS` returns a
    string while the :attr:`extensions` attribute returns a list:

    >>> clGetPlatformInfo(plat, cl_platform_info.CL_PLATFORM_EXTENSIONS)  # doctest: +ELLIPSIS
    '...'
    >>> plat.extensions                              # doctest: +ELLIPSIS
    [...]
    """
    sz = size_t()
    clGetPlatformInfo.call(platform, param_name, 0, None, byref(sz))
    # All parameter types currently return strings.
    param_value = create_string_buffer(sz.value)
    clGetPlatformInfo.call(platform, param_name, sz.value, param_value, None)
    if sys.version_info[0] > 2:
        return str(param_value.value, "utf-8")
    else:
        return param_value.value


##################
# Device Objects #
##################


class cl_device(void_p):
    """
    Represents an OpenCL Device belonging to some platform.
    Should not be directly instantiated by users of PyCL.
    Use :func:`clGetDeviceIDs` or the :attr:`devices` attribute of
    some OpenCL objects to procure a cl_device instances.
    """

    def __repr__(self):
        try:
            return "<cl_device '%s'>" % (self.name)
        except:
            return "<cl_device 0x%x>" % (self.value or 0)

    # Devices have so many freaking properties that I'm not going
    # to bother listing them all here. There's a for loop after the
    # various type definitions that adds them all. The ones here
    # take precedence.
    @property
    def driver_version(self):
        # Defined here because it doesn't start with "CL_DEVICE_",
        # so the for-loop can't handle it.
        return clGetDeviceInfo(self, cl_device_info.CL_DRIVER_VERSION)

    @property
    def extensions(self):
        # Split extension list into an actual list.
        return clGetDeviceInfo(self, cl_device_info.CL_DEVICE_EXTENSIONS).split()


# Laziness on my part. There are a *lot* of cl_device_info constants
# representing possible inputs to clGetDeviceInfo. There should be
# convenience properties for each of these, but I don't want to type
# out all those property definitions. So we generate them.
for name, val in cl_device_info._by_name.items():
    if name.startswith("CL_DEVICE_"):
        propname = name[10:].lower()
        if not hasattr(cl_device, propname):
            setattr(
                cl_device,
                propname,
                property(
                    lambda self, val=val: clGetDeviceInfo(self, val),
                    doc="Same as calling :func:`clGetDeviceInfo` "
                    " with :const:`~cl_device_info.%s`" % name,
                ),
            )
# cleanup
del name
del val
del propname


@_wrapdll(cl_platform, cl_device_type, cl_uint, P(cl_device), P(cl_uint))
def clGetDeviceIDs(platform=None, device_type=cl_device_type.CL_DEVICE_TYPE_ALL):
    """
    :param platform: The :class:`cl_platform` whose devices you are
      interested in. If none is provided, the first platform on the
      system is used.
    :param device_type: A :class:`cl_device_type` bitfield indicating which
      devices should be listed. By default, all are listed.

    >>> clGetDeviceIDs() # doctest: +ELLIPSIS
    (<cl_device '...'>...)
    """
    num_devices = cl_uint()
    if platform is None:
        platform = clGetPlatformIDs()[0]
    clGetDeviceIDs.call(platform, device_type, 0, None, byref(num_devices))
    n = num_devices.value
    if n > 0:
        device_array = (cl_device * n)()
        clGetDeviceIDs.call(platform, device_type, num_devices, device_array, None)
        return tuple(x for x in device_array)
    else:
        return ()


# clGetDeviceInfo has a lot of different possible return types.
# Anything not handled identified in one of these sets or in
# a special case in the wrapper function is assumed to return a cl_uint.

_device_info_sizes = frozenset(
    (
        cl_device_info.CL_DEVICE_MAX_WORK_GROUP_SIZE,
        cl_device_info.CL_DEVICE_IMAGE2D_MAX_WIDTH,
        cl_device_info.CL_DEVICE_IMAGE2D_MAX_HEIGHT,
        cl_device_info.CL_DEVICE_IMAGE3D_MAX_WIDTH,
        cl_device_info.CL_DEVICE_IMAGE3D_MAX_DEPTH,
        cl_device_info.CL_DEVICE_MAX_PARAMETER_SIZE,
        cl_device_info.CL_DEVICE_PROFILING_TIMER_RESOLUTION,
    )
)

_device_info_counts = frozenset(
    (cl_device_info.CL_DEVICE_MAX_COMPUTE_UNITS, cl_device_info.CL_DEVICE_MAX_WORK_ITEM_SIZES)
)

_device_info_ulongs = frozenset(
    (
        cl_device_info.CL_DEVICE_MAX_MEM_ALLOC_SIZE,
        cl_device_info.CL_DEVICE_GLOBAL_MEM_CACHE_SIZE,
        cl_device_info.CL_DEVICE_GLOBAL_MEM_SIZE,
        cl_device_info.CL_DEVICE_MAX_CONSTANT_BUFFER_SIZE,
        cl_device_info.CL_DEVICE_LOCAL_MEM_SIZE,
    )
)

_device_info_bools = frozenset(
    (
        cl_device_info.CL_DEVICE_IMAGE_SUPPORT,
        cl_device_info.CL_DEVICE_HOST_UNIFIED_MEMORY,
        cl_device_info.CL_DEVICE_ENDIAN_LITTLE,
        cl_device_info.CL_DEVICE_AVAILABLE,
        cl_device_info.CL_DEVICE_COMPILER_AVAILABLE,
    )
)

_device_info_strings = frozenset(
    (
        cl_device_info.CL_DEVICE_NAME,
        cl_device_info.CL_DEVICE_VENDOR,
        cl_device_info.CL_DRIVER_VERSION,
        cl_device_info.CL_DEVICE_PROFILE,
        cl_device_info.CL_DEVICE_VERSION,
        cl_device_info.CL_DEVICE_EXTENSIONS,
    )
)


@_wrapdll(cl_device, cl_device_info, size_t, void_p, P(size_t))
def clGetDeviceInfo(device, param_name):
    """
    :param device: A :class:`cl_device`.
    :param param_name: The :class:`cl_device_info` item to be queried.

    :class:`cl_device` objects have attributes that will call this for
    you, so you should probably use those instead of calling this directly.

    >>> d = clGetDeviceIDs()[0]
    >>> clGetDeviceInfo(d, cl_device_info.CL_DEVICE_NAME)  # doctest: +ELLIPSIS
    '...'
    >>> clGetDeviceInfo(d, cl_device_info.CL_DEVICE_TYPE)  # doctest: +ELLIPSIS
    CL_DEVICE_TYPE_...
    >>> d.available
    True
    >>> d.max_work_item_sizes               # doctest: +ELLIPSIS
    (...)

    Note that :const:`~cl_device_info.CL_DEVICE_EXTENSIONS` returns a
    string while the :attr:`extensions` attribute returns a list:

    >>> clGetDeviceInfo(d, cl_device_info.CL_DEVICE_EXTENSIONS)  # doctest: +ELLIPSIS
    '...'
    >>> d.extensions                              # doctest: +ELLIPSIS
    [...]

    """
    if param_name == cl_device_info.CL_DEVICE_TYPE:
        param_value = cl_device_type()
        clGetDeviceInfo.call(device, param_name, sizeof(param_value), byref(param_value), None)
        return param_value
    elif param_name in _device_info_sizes:
        param_value = size_t()
        clGetDeviceInfo.call(device, param_name, sizeof(param_value), byref(param_value), None)
        return int(param_value.value)
    elif param_name in _device_info_ulongs:
        param_value = cl_ulong()
        clGetDeviceInfo.call(device, param_name, sizeof(param_value), byref(param_value), None)
        return int(param_value.value)
    elif param_name in _device_info_bools:
        param_value = cl_bool()
        clGetDeviceInfo.call(device, param_name, sizeof(param_value), byref(param_value), None)
        return bool(param_value.value)
    elif param_name in _device_info_strings:
        sz = size_t()
        clGetDeviceInfo.call(device, param_name, 0, None, byref(sz))
        param_value = create_string_buffer(sz.value)
        clGetDeviceInfo.call(device, param_name, sz, param_value, None)
        if sys.version_info[0] > 2:
            return str(param_value.value, "utf-8")
        else:
            return param_value.value
    elif param_name == cl_device_info.CL_DEVICE_SINGLE_FP_CONFIG:
        param_value = cl_device_fp_config()
        clGetDeviceInfo.call(device, param_name, sizeof(param_value), byref(param_value), None)
        return param_value
    elif param_name == cl_device_info.CL_DEVICE_GLOBAL_MEM_CACHE_TYPE:
        param_value = cl_device_mem_cache_type()
        clGetDeviceInfo.call(device, param_name, sizeof(param_value), byref(param_value), None)
        return param_value
    elif param_name == cl_device_info.CL_DEVICE_LOCAL_MEM_TYPE:
        param_value = cl_device_local_mem_type()
        clGetDeviceInfo.call(device, param_name, sizeof(param_value), byref(param_value), None)
        return param_value
    elif param_name == cl_device_info.CL_DEVICE_MAX_WORK_ITEM_SIZES:
        sz = size_t()
        clGetDeviceInfo.call(device, param_name, 0, None, byref(sz))
        nd = sz.value // sizeof(size_t)
        param_value = (size_t * nd)()
        clGetDeviceInfo.call(device, param_name, sz, byref(param_value), None)
        return tuple(int(x) for x in param_value)
    elif param_name == cl_device_info.CL_DEVICE_EXECUTION_CAPABILITIES:
        param_value = cl_device_exec_capabilities()
        clGetDeviceInfo.call(device, param_name, sizeof(param_value), byref(param_value), None)
        return param_value
    elif param_name == cl_device_info.CL_DEVICE_QUEUE_PROPERTIES:
        param_value = cl_command_queue_properties()
        clGetDeviceInfo.call(device, param_name, sizeof(param_value), byref(param_value), None)
        return param_value
    elif param_name == cl_device_info.CL_DEVICE_PLATFORM:
        param_value = cl_platform()
        clGetDeviceInfo.call(device, param_name, sizeof(param_value), byref(param_value), None)
        return param_value
    else:
        param_value = cl_uint()
        clGetDeviceInfo.call(device, param_name, sizeof(param_value), byref(param_value), None)
        return int(param_value.value)


###################
# Context Objects #
###################


class cl_context(void_p):
    """
    Represents an OpenCL Context instance.

    Use :func:`clCreateContext` or :func:`clCreateContextFromType` to
    create a new context.

    Participates in OpenCL's reference counting scheme.
    """

    @property
    def platform(self):
        """
        Retrieve the platform this context was made using. (cl_platform)
        """
        try:
            return self._platform
        except AttributeError:
            return clGetContextInfo(self, cl_context_info.CL_CONTEXT_PLATFORM)

    @property
    def reference_count(self):
        """
        Reference count for OpenCL's internal garbage collector. (int)
        Using :func:`clReleaseContext` via pycl is an excellent way to
        generate segmentation faults.
        """
        return clGetContextInfo(self, cl_context_info.CL_CONTEXT_REFERENCE_COUNT)

    @property
    def num_devices(self):
        """
        Number of devices present in this particular context. (int)
        """
        return clGetContextInfo(self, cl_context_info.CL_CONTEXT_NUM_DEVICES)

    @property
    def devices(self):
        """
        List of devices present in this particular context.
        (list of :class:`cl_device`)
        """
        try:
            return self._context
        except AttributeError:
            return clGetContextInfo(self, cl_context_info.CL_CONTEXT_DEVICES)

    @property
    def properties(self):
        """
        Low-level ctypes array that is probably not user-interpretable.
        """
        return clGetContextInfo(self, cl_context_info.CL_CONTEXT_PROPERTIES)

    def __repr__(self):
        try:
            plat = self.platform.name
        except:
            plat = "Unknown"
        nd = self.num_devices or 0
        address = self.value or 0
        return "<cl_context %s:%d 0x%x>" % (plat, nd, address)

    def __del__(self):
        try:
            if self and self.reference_count > 0:
                clReleaseContext(self)
        except:
            pass


@_wrapdll(
    P(cl_context_properties),
    cl_uint,
    P(cl_device),
    void_p,
    void_p,
    P(cl_errnum),
    res=cl_context,
    err=_lastarg_errcheck,
)
def clCreateContext(devices=None, platform=None, other_props=None):
    """
    Create a context with the given devices and platform.

    :param devices: A list of devices. If None, the first device from
      the given platform is used.
    :param platform: If no platform or devices are provided, the first
      platform found will be used. If a device list is provided but no
      platform, the platform will be recovered from the devices.

    If you just need a context and don't care what you get, calling with
    no arguments should hopefully get you something usable.

    >>> clCreateContext()  # doctest: +ELLIPSIS
    <cl_context ...>
    >>> one_device = clGetDeviceIDs()[0]
    >>> clCreateContext(devices = [one_device]) # doctest: +ELLIPSIS
    <cl_context ...>
    """
    properties = dict()
    if platform is None:
        if devices:
            platform = devices[0].platform
        else:
            platform = clGetPlatformIDs()[0]
    properties[cl_context_info.CL_CONTEXT_PLATFORM] = platform
    if other_props:
        properties.update(other_props)
    if devices is None:
        devices = platform.devices[:1]
    props = (cl_context_properties * (2 * len(properties) + 1))()
    for i, p in enumerate(properties):
        props[2 * i] = p.value
        if isinstance(properties[p], _ctypes._Pointer):
            props[2 * i + 1] = ctypes.addressof(properties[p].contents)
        elif isinstance(properties[p], _ctypes._SimpleCData):
            props[2 * i + 1] = properties[p].value
        else:
            props[2 * i + 1] = properties[p]
    props[2 * len(properties)] = 0
    if devices:
        device_array = (cl_device * len(devices))()
    else:
        device_array = None
    for i, d in enumerate(devices):
        device_array[i] = d
    ctx = clCreateContext.call(props, len(devices), device_array, None, None, byref(cl_errnum()))
    if clGetContextInfo(ctx, cl_context_info.CL_CONTEXT_REFERENCE_COUNT) < 1:
        raise ValueError("Unusable context")
    return ctx


@_wrapdll(
    P(cl_context_properties),
    cl_device_type,
    void_p,
    void_p,
    P(cl_errnum),
    res=cl_context,
    err=_lastarg_errcheck,
)
def clCreateContextFromType(
    device_type=cl_device_type.CL_DEVICE_TYPE_DEFAULT, platform=None, other_props=None
):
    """
    Like :func:`clCreateContext`, but works by device type instead
    of expecting you to list the desired devices. This can, for instance,
    be used to create a context with GPU devices without the user having
    to pick a platform and inspect its device list.

    :param device_type: A :class:`cl_device_type` field indicating which
      types of devices should be included.
    :param platform: A :class:`cl_platform`. If no platform is provided,
      each platform will be tried in turn until a context with the specified
      device type can created.

    If you just need a context and don't care what you get, calling with
    no arguments should hopefully get you something usable.

    >>> clCreateContextFromType(CL_DEVICE_TYPE_CPU | CL_DEVICE_TYPE_GPU) # doctest: +ELLIPSIS
    <cl_context ...>
    """
    properties = dict()
    if platform is None:
        all_plats = clGetPlatformIDs()
        for plat in all_plats:
            try:
                ctx = clCreateContextFromType(device_type, plat, other_props)
            except ValueError:
                continue
            return ctx
        else:
            raise ValueError("Could not create suitable context")
    properties[cl_context_info.CL_CONTEXT_PLATFORM] = platform
    if other_props:
        properties.update(other_props)
    props = (cl_context_properties * (2 * len(properties) + 1))()
    for i, p in enumerate(properties):
        props[2 * i] = p.value
        try:
            props[2 * i + 1] = properties[p]
        except TypeError:
            props[2 * i + 1] = properties[p].value
    props[2 * len(properties)] = 0
    ctx = clCreateContextFromType.call(props, device_type, None, None, byref(cl_errnum()))
    if clGetContextInfo(ctx, cl_context_info.CL_CONTEXT_REFERENCE_COUNT) < 1:
        raise ValueError("Unusable context")
    return ctx


@_wrapdll(cl_context, cl_context_info, size_t, void_p, P(size_t))
def clGetContextInfo(context, param_name):
    """
    Retrieve context info.

    :param context: :class:`cl_context`.
    :param param_name: One of the :class:`cl_context_info` values.

    :class:`cl_context` objects have attributes that will call this for
    you, so you should probably use those instead of calling this directly.

    >>> ctx = clCreateContext()
    >>> clGetContextInfo(ctx, cl_context_info.CL_CONTEXT_DEVICES) # doctest: +ELLIPSIS
    (<cl_device ...>...)
    >>> ctx.platform # doctest: +ELLIPSIS
    <cl_platform ...>
    >>> ctx.reference_count
    1
    >>> ctx.properties # doctest: +ELLIPSIS
    <...cl_context_properties_Array...>
    """
    if param_name == cl_context_info.CL_CONTEXT_DEVICES:
        try:
            return context._devices
        except AttributeError:
            sz = size_t()
            clGetContextInfo.call(context, param_name, 0, None, byref(sz))
            num_dev = sz.value // sizeof(cl_device)
            dev_array = (cl_device * num_dev)()
            clGetContextInfo.call(context, param_name, sz, dev_array, None)
            context._devices = tuple(x for x in dev_array)
            return context._devices
    elif param_name == cl_context_info.CL_CONTEXT_PROPERTIES:
        sz = size_t()
        clGetContextInfo.call(context, param_name, 0, None, byref(sz))
        num_props = sz.value // sizeof(cl_context_properties)
        props = (cl_context_properties * num_props)()
        clGetContextInfo.call(context, param_name, sz, props, None)
        # TODO
        # It's not entirely clear how we should present the result object
        # to the user, since other than cl_context_info.CL_CONTEXT_PLATFORM the possible
        # values are all extension-dependent. For now, just return it.
        return props
    elif param_name == cl_context_info.CL_CONTEXT_PLATFORM:
        # Not actually a valid input, but it should probably be
        # available in the properties list.
        try:
            return context._platform
        except AttributeError:
            props = clGetContextInfo(context, cl_context_info.CL_CONTEXT_PROPERTIES)
            for i in range(0, len(props) - 1, 2):
                if props[i].value == cl_context_info.CL_CONTEXT_PLATFORM.value:
                    context._platform = cl_platform(props[i + 1].value)
                    break
            else:
                context._platform = None
            return context._platform
    elif param_name == cl_context_info.CL_CONTEXT_NUM_DEVICES:
        # Sidestep bug in NVIDIA OpenCL driver by calculating
        # this in another fashion.
        sz = size_t()
        clGetContextInfo.call(context, cl_context_info.CL_CONTEXT_DEVICES, 0, None, byref(sz))
        return sz.value // sizeof(cl_device)
    else:
        param_value = cl_uint()
        clGetContextInfo.call(context, param_name, sizeof(param_value), byref(param_value), None)
        return int(param_value.value)


@_wrapdll(cl_context)
def clRetainContext(context):
    # Not for end-user use
    clRetainContext.call(context)


@_wrapdll(cl_context)
def clReleaseContext(context):
    # Not for end-user use
    clReleaseContext.call(context)


##################
# Command Queues #
##################


class cl_command_queue(void_p):
    """
    Represents an OpenCL Command Queue instance.
    Should not be directly instantiated by users of PyCL.
    Use :func:`clCreateCommandQueue` to create a new queue.
    """

    @property
    def context(self):
        """
        The context associated with the command queue. (:class:`cl_context`)
        """
        return clGetCommandQueueInfo(self, cl_command_queue_info.CL_QUEUE_CONTEXT)

    @property
    def device(self):
        """
        The device associated with the command queue. (:class:`cl_device`)
        """
        return clGetCommandQueueInfo(self, cl_command_queue_info.CL_QUEUE_DEVICE)

    @property
    def properties(self):
        """
        Command queue property bitfield. (:class:`cl_command_queue_properties`)
        """
        return clGetCommandQueueInfo(self, cl_command_queue_info.CL_QUEUE_PROPERTIES)

    @property
    def reference_count(self):
        """
        Reference count for OpenCL's garbage collector. (int)
        """
        return clGetCommandQueueInfo(self, cl_command_queue_info.CL_QUEUE_REFERENCE_COUNT)

    def __repr__(self):
        try:
            dev = self.device
            return "<cl_command_queue '%s'>" % dev.name
        except:
            return "<cl_command_queue 0x%x>" % (self.value or 0)

    def __del__(self):
        try:
            if self and self.reference_count > 0:
                clReleaseCommandQueue(self)
        except:
            pass


@_wrapdll(
    cl_context,
    cl_device,
    cl_command_queue_properties,
    P(cl_errnum),
    res=cl_command_queue,
    err=_lastarg_errcheck,
)
def clCreateCommandQueue(context=None, device=None, properties=None):
    """
    :param context: :class:`cl_context`. If not provided, one will be
      generated for you by calling :func:`clCreateContext` with no arguments.
      (it can later be retrieved via the :attr:`context` attribute)
    :param device: The :class:`cl_device` that will be fed by this queue.
      If no device is provided, the first device in the context will be used.
    :param properties: A :class:`cl_command_queue_properties` bitfield.
    """
    if context is None:
        context = clCreateContext()
    if device is None:
        device = context.devices[0]
    if properties is None:
        properties = cl_command_queue_properties.NONE
    queue = clCreateCommandQueue.call(context, device, properties, byref(cl_errnum()))
    queue._context = context
    return queue


@_wrapdll(cl_command_queue, cl_command_queue_info, size_t, void_p, P(size_t))
def clGetCommandQueueInfo(queue, param_name):
    """
    :param queue: :class:`cl_command_queue`.
    :param param_name: One of the :class:`cl_command_queue_info` values.

    >>> q = clCreateCommandQueue()
    >>> q.context # doctest: +ELLIPSIS
    <cl_context ...>
    >>> q.device  # doctest: +ELLIPSIS
    <cl_device ...>
    >>> q.properties # doctest: +ELLIPSIS
    NONE
    >>> q.reference_count
    1
    """
    if param_name == cl_command_queue_info.CL_QUEUE_CONTEXT:
        param_value = cl_context()
        clGetCommandQueueInfo.call(queue, param_name, sizeof(param_value), byref(param_value), None)
        # Calling this doesn't increase the reference count,
        # so we need to do that.
        clRetainContext(param_value)
        return param_value
    elif param_name == cl_command_queue_info.CL_QUEUE_DEVICE:
        param_value = cl_device()
        clGetCommandQueueInfo.call(queue, param_name, sizeof(param_value), byref(param_value), None)
        return param_value
    elif param_name == cl_command_queue_info.CL_QUEUE_PROPERTIES:
        param_value = cl_command_queue_properties()
        clGetCommandQueueInfo.call(queue, param_name, sizeof(param_value), byref(param_value), None)
        return param_value
    elif param_name == cl_command_queue_info.CL_QUEUE_REFERENCE_COUNT:
        param_value = cl_uint()
        clGetCommandQueueInfo.call(queue, param_name, sizeof(param_value), byref(param_value), None)
        return int(param_value.value)
    else:
        raise ValueError("Unrecognized parameter %s" % param_name)


@_wrapdll(cl_command_queue)
def clRetainCommandQueue(queue):
    # Not for end-user use
    clRetainCommandQueue.call(queue)


@_wrapdll(cl_command_queue)
def clReleaseCommandQueue(queue):
    # Not for end-user use
    clReleaseCommandQueue.call(queue)


##################
# Memory Objects #
##################


class cl_mem(void_p):
    """
    Represents an OpenCL memory object, typically a buffer or image.

    See the individual types (:class:`cl_buffer` and :class:`cl_image`)
    for details. PyCL should probably never give you a direct instance
    of this class - treat it as abstract.

    Memory objects are reference counted.
    """

    @property
    def size(self):
        """Memory size, in bytes."""
        try:
            return self._size
        except AttributeError:
            return clGetMemObjectInfo(self, cl_mem_info.CL_MEM_SIZE)

    @property
    def reference_count(self):
        """Reference count for OpenCL garbage collector."""
        return clGetMemObjectInfo(self, cl_mem_info.CL_MEM_REFERENCE_COUNT)

    @property
    def map_count(self):
        """Number of memory maps currently active for this object."""
        return clGetMemObjectInfo(self, cl_mem_info.CL_MEM_MAP_COUNT)

    @property
    def hostptr(self):
        """Pointer to host address associated with this memory
        object at the time of creation. The meaning varies depending
        on the flags. (type is :c:type:`void*`)"""
        try:
            return self._hostptr
        except AttributeError:
            return clGetMemObjectInfo(self, cl_mem_info.CL_MEM_HOST_PTR)

    @property
    def flags(self):
        """The :class:`cl_mem_flags` the object was created with."""
        return clGetMemObjectInfo(self, cl_mem_info.CL_MEM_FLAGS)

    @property
    def type(self):
        """The :class:`cl_mem_type` of the object."""
        try:
            return self._type
        except AttributeError:
            return clGetMemObjectInfo(self, cl_mem_info.CL_MEM_TYPE)

    @property
    def context(self):
        """The :class:`cl_context` the memory belongs to."""
        try:
            return self._context
        except AttributeError:
            return clGetMemObjectInfo(self, cl_mem_info.CL_MEM_CONTEXT)

    def __del__(self):
        try:
            if self:
                clReleaseMemObject(self)
        except:
            pass


class cl_buffer(cl_mem):
    """
    A subclass of :class:`cl_mem` representing memory buffers.
    Create these with :func:`clCreateBuffer` or :func:`clCreateSubBuffer`
    """

    def empty_like_this(self):
        """Creates an empty read/write buffer of the same size
        in the same context and returns it."""
        return clCreateBuffer(self.context, self.size)

    @property
    def base(self):
        """Base memory object (for sub-buffers)"""
        try:
            return self._base
        except AttributeError:
            return clGetMemObjectInfo(self, cl_mem_info.CL_MEM_ASSOCIATED_MEMOBJECT)

    @property
    def offset(self):
        """Offset, in bytes, from origin (for sub-buffers)"""
        try:
            return self._offset
        except AttributeError:
            return clGetMemObjectInfo(self, cl_mem_info.CL_MEM_OFFSET)


class cl_image(cl_mem):
    """
    A subclass of :class:`cl_mem` representing 2D or 3D images.
    Create these with :func:`clCreateImage2D` or :func:`clCreateImage3D`.
    """

    @property
    def format(self):
        try:
            return self._format
        except AttributeError:
            return clGetImageInfo(self, cl_image_info.CL_IMAGE_FORMAT)

    @property
    def element_size(self):
        try:
            return self._element_size
        except AttributeError:
            self._element_size = clGetImageInfo(self, cl_image_info.CL_IMAGE_ELEMENT_SIZE)
            return self._element_size

    @property
    def row_pitch(self):
        try:
            return self._row_pitch
        except AttributeError:
            self._row_pitch = clGetImageInfo(self, cl_image_info.CL_IMAGE_ROW_PITCH)
            return self._row_pitch

    @property
    def slice_pitch(self):
        try:
            return self._slice_pitch
        except AttributeError:
            self._slice_pitch = clGetImageInfo(self, cl_image_info.CL_IMAGE_SLICE_PITCH)
            return self._slice_pitch

    @property
    def width(self):
        try:
            return self._width
        except AttributeError:
            self._width = clGetImageInfo(self, cl_image_info.CL_IMAGE_WIDTH)
            return self._width

    @property
    def height(self):
        try:
            return self._height
        except AttributeError:
            self._height = clGetImageInfo(self, cl_image_info.CL_IMAGE_HEIGHT)
            return self._height

    @property
    def depth(self):
        try:
            return self._depth
        except AttributeError:
            self._depth = clGetImageInfo(self, cl_image_info.CL_IMAGE_DEPTH)
            return self._depth

    def empty_like_this(self):
        if self.type == cl_mem_object_type.CL_MEM_OBJECT_IMAGE2D:
            return clCreateImage2D(self.context, self.width, self.height, self.format)
        elif self.type == cl_mem_object_type.CL_MEM_OBJECT_IMAGE3D:
            return clCreateImage3D(self.context, self.width, self.height, self.depth, self.format)
        else:
            raise TypeError("Unknown memory type")


@_wrapdll(
    cl_context, cl_mem_flags, size_t, void_p, P(cl_errnum), res=cl_buffer, err=_lastarg_errcheck
)
def clCreateBuffer(context, size, flags=cl_mem_flags.CL_MEM_READ_WRITE, host_ptr=None):
    """
    :param context: :class:`cl_context` that will own this memory.
    :param size: Desired size (in bytes) of the memory.
    :param flags: :class:`cl_mem_flags` to control the memory.
    :param host_ptr: :c:type:`void*` to associated with this memory.
      The meaning of the association depends on the flags. (An integer
      representation of a pointer is fine).

    See also :func:`buffer_from_ndarray`, :func:`buffer_from_pyarray`
    """
    mem = clCreateBuffer.call(context, flags, size, host_ptr, byref(cl_errnum()))
    mem._size = size
    mem._context = context
    mem._base = host_ptr
    mem._flags = flags
    mem._type = cl_mem_object_type.CL_MEM_OBJECT_BUFFER
    return mem


@_wrapdll(cl_mem)
def clRetainMemObject(mem):
    clRetainMemObject.call(mem)


@_wrapdll(cl_mem)
def clReleaseMemObject(mem):
    clReleaseMemObject.call(mem)


@_wrapdll(
    cl_command_queue, cl_buffer, cl_bool, size_t, size_t, void_p, cl_uint, P(cl_event), P(cl_event)
)
def clEnqueueReadBuffer(queue, mem, pointer, size=None, blocking=True, offset=0, wait_for=None):
    """
    Read from a :class:`cl_mem` buffer into host memory.

    :param queue: :class:`cl_command_queue` to queue it on.
    :param mem: :class:`cl_mem` to read from. Must be a buffer.
    :param pointer: :c:type:`void*` pointer, the address to start
      writing into. (An integer representation of the pointer is fine).
    :param size: Number of bytes to read. If not specified, the entire
      buffer is read out, which might be hazardous if the place you're
      writing it to isn't big enough.
    :param blocking: Wait for the transfer to complete. Default is True.
      If False, you can use the returned event to check its status.
    :param offset: Offset in the buffer at which to start reading. Default is 0.
    :param wait_for: :class:`cl_event` (or a list of them) that must complete
      before the memory transfer will commence.
    :returns: :class:`cl_event`

    See also :func:`buffer_to_ndarray` and :func:`buffer_to_pyarray`.

    >>> ctx = clCreateContext()
    >>> queue = clCreateCommandQueue(ctx)
    >>> array1 = (cl_int * 8)()  # 32 bytes
    >>> for i in range(8): array1[i] = i
    >>> m = clCreateBuffer(ctx, 32)
    >>> clEnqueueWriteBuffer(queue, m, array1, 32) # doctest: +ELLIPSIS
    <cl_event ...>
    >>> array2 = (cl_int * 8)()
    >>> clEnqueueReadBuffer(queue, m, array2, 32) # doctest: +ELLIPSIS
    <cl_event ...>
    >>> [x.value for x in array2]
    [0, 1, 2, 3, 4, 5, 6, 7]
    """
    if size is None:
        size = clGetMemObjectInfo(mem, cl_mem_info.CL_MEM_SIZE)
    nevents, wait_array = _make_event_array(wait_for)
    out_event = cl_event()
    clEnqueueReadBuffer.call(
        queue, mem, blocking, offset, size, pointer, nevents, wait_array, byref(out_event)
    )
    return out_event


@_wrapdll(
    cl_command_queue, cl_buffer, cl_bool, size_t, size_t, void_p, cl_uint, P(cl_event), P(cl_event)
)
def clEnqueueWriteBuffer(queue, mem, pointer, size=None, blocking=True, offset=0, wait_for=None):
    """
    Write to a :class:`cl_buffer` buffer from a location in host memory.

    See :func:`clEnqueueReadBuffer` for the meanings of the parameters.
    """
    if size is None:
        size = clGetMemObjectInfo(mem, cl_mem_info.CL_MEM_SIZE)
    nevents, wait_array = _make_event_array(wait_for)
    out_event = cl_event()
    clEnqueueWriteBuffer.call(
        queue, mem, blocking, offset, size, pointer, nevents, wait_array, byref(out_event)
    )
    return out_event


@_wrapdll(
    cl_command_queue,
    cl_buffer,
    cl_buffer,
    size_t,
    size_t,
    void_p,
    cl_uint,
    P(cl_event),
    P(cl_event),
)
def clEnqueueCopyBuffer(
    queue, src_buffer, dst_buffer, src_offset=0, dst_offset=0, size=None, wait_for=None
):
    if size is None:
        size = clGetMemObjectInfo(dst_buffer, cl_mem_info.CL_MEM_SIZE)
    nevents, wait_array = _make_event_array(wait_for)
    out_event = cl_event()
    clEnqueueCopyBuffer.call(
        queue,
        src_buffer,
        dst_buffer,
        src_offset,
        dst_offset,
        size,
        nevents,
        wait_array,
        byref(out_event),
    )
    return out_event


@_wrapdll(
    cl_command_queue, cl_buffer, void_p, size_t, size_t, void_p, cl_uint, P(cl_event), P(cl_event)
)
def clEnqueueFillBuffer(queue, mem, pattern, pattern_size=0, offset=0, size=0, wait_for=None):
    """
    Enqueues a command to fill a buffer object with a pattern of a given pattern size.

    TODO: Automatically determine pattern_size, perhaps in a wrapper function?
    """
    if size is None:
        size = clGetMemObjectInfo(mem, cl_mem_info.CL_MEM_SIZE)
    nevents, wait_array = _make_event_array(wait_for)
    out_event = cl_event()
    clEnqueueWriteBuffer.call(
        queue, mem, pattern, pattern_size, offset, size, nevents, wait_array, byref(out_event)
    )
    return out_event


@_wrapdll(cl_mem, cl_mem_info, size_t, void_p, P(size_t))
def clGetMemObjectInfo(mem, param_name):
    """
    :param mem: :class:`cl_mem`
    :param param_name: One of the :class:`cl_mem_info` values.

    Memory objects have properties that will retrieve these
    values for you, so you should probably use those.
    """
    if param_name == cl_mem_info.CL_MEM_TYPE:
        try:
            return mem._type
        except AttributeError:
            param_value = cl_mem_object_type()
            clGetMemObjectInfo.call(mem, param_name, sizeof(param_value), byref(param_value), None)
            mem._type = param_value
            return param_value
    elif param_name == cl_mem_info.CL_MEM_FLAGS:
        param_value = cl_mem_flags()
        clGetMemObjectInfo.call(mem, param_name, sizeof(param_value), byref(param_value), None)
        return param_value
    elif param_name == cl_mem_info.CL_MEM_SIZE:
        try:
            return mem._size
        except AttributeError:
            param_value = size_t()
            clGetMemObjectInfo.call(mem, param_name, sizeof(param_value), byref(param_value), None)
            mem._size = int(param_value.value)
            return mem._size
    elif param_name == cl_mem_info.CL_MEM_OFFSET:
        param_value = size_t()
        clGetMemObjectInfo.call(mem, param_name, sizeof(param_value), byref(param_value), None)
        mem._offset = int(param_value.value)
        return mem._offset
    elif param_name in (cl_mem_info.CL_MEM_MAP_COUNT, cl_mem_info.CL_MEM_REFERENCE_COUNT):
        param_value = cl_uint()
        clGetMemObjectInfo.call(mem, param_name, sizeof(param_value), byref(param_value), None)
        return int(param_value.value)
    elif param_name == cl_mem_info.CL_MEM_ASSOCIATED_MEMOBJECT:
        try:
            return mem._base
        except AttributeError:
            pass
        param_value = cl_buffer()
        clGetMemObjectInfo.call(mem, param_name, sizeof(param_value), byref(param_value), None)
        if param_value:
            mem._base = param_value
        else:
            mem._base = None
        return mem._base
    elif param_name == cl_mem_info.CL_MEM_CONTEXT:
        try:
            return mem._context
        except AttributeError:
            pass
        param_value = cl_context()
        clGetMemObjectInfo.call(mem, param_name, sizeof(param_value), byref(param_value), None)
        clRetainContext(param_value)
        mem._context = param_value
        return param_value
    elif param_name == cl_mem_info.CL_MEM_HOST_PTR:
        try:
            return mem._hostptr
        except AttributeError:
            pass
        param_value = void_p()
        clGetMemObjectInfo.call(mem, param_name, sizeof(param_value), byref(param_value), None)
        if param_value:
            mem._hostptr = param_value
        else:
            mem._hostptr = None
        return mem._hostptr
    else:  # e.g., CL_MEM_D3D10_RESOURCE_KHR
        param_value = void_p()
        clGetMemObjectInfo.call(mem, param_name, sizeof(param_value), byref(param_value), None)
        return param_value or None


#################
# Image Objects #
#################


@_wrapdll(
    cl_context,
    cl_mem_flags,
    P(cl_image_format),
    size_t,
    size_t,
    size_t,
    void_p,
    P(cl_errnum),
    res=cl_image,
    err=_lastarg_errcheck,
)
def clCreateImage2D(
    context,
    width,
    height,
    imgformat=None,  # If we can guess it
    flags=cl_mem_flags.CL_MEM_READ_WRITE,
    host_ptr=None,
    row_pitch=0,
):
    if isinstance(imgformat, (tuple, list)):
        imgformat = cl_image_format(*imgformat)
    assert imgformat is not None, "don't know how to guess this yet"
    img = clCreateImage2D.call(
        context, flags, byref(imgformat), width, height, row_pitch, host_ptr, byref(cl_errnum())
    )
    img._context = context
    img._width = width
    img._height = height
    img._base = host_ptr
    img._flags = flags
    img._format = imgformat
    return img


@_wrapdll(
    cl_context,
    cl_mem_flags,
    P(cl_image_format),
    size_t,
    size_t,
    size_t,
    size_t,
    size_t,
    void_p,
    P(cl_errnum),
    res=cl_image,
    err=_lastarg_errcheck,
)
def clCreateImage3D(
    context,
    width,
    height,
    depth,
    imgformat=None,  # If we can guess it
    flags=cl_mem_flags.CL_MEM_READ_WRITE,
    host_ptr=None,
    row_pitch=0,
    slice_pitch=0,
):
    if isinstance(imgformat, (tuple, list)):
        imgformat = cl_image_format(*imgformat)
    assert imgformat is not None, "don't know how to guess this yet"
    img = clCreateImage3D.call(
        context,
        flags,
        byref(imgformat),
        width,
        height,
        depth,
        row_pitch,
        slice_pitch,
        host_ptr,
        byref(cl_errnum()),
    )
    img._context = context
    img._width = width
    img._height = height
    img._depth = depth
    img._base = host_ptr
    img._flags = flags
    img._format = imgformat
    return img


@_wrapdll(cl_context, cl_mem_flags, cl_mem_object_type, cl_uint, P(cl_image_format), P(cl_uint))
def clGetSupportedImageFormats(
    context=None,
    type=cl_mem_object_type.CL_MEM_OBJECT_IMAGE2D,
    flags=cl_mem_flags.CL_MEM_READ_WRITE,
):
    if context is None:
        context = clCreateContext()
    num = cl_uint()
    clGetSupportedImageFormats.call(context, flags, type, 0, None, byref(num))
    formats = (cl_image_format * num.value)()
    clGetSupportedImageFormats.call(context, flags, type, num, formats, None)
    return [f for f in formats]


@_wrapdll(cl_image, cl_image_info, size_t, void_p, P(size_t))
def clGetImageInfo(img, param_name):
    if param_name == cl_image_info.CL_IMAGE_FORMAT:
        try:
            return img._format
        except AttributeError:
            param_value = cl_image_format()
            clGetImageInfo.call(img, param_name, sizeof(param_value), byref(param_value), None)
            img._format = param_value
            return param_value
    elif param_name in (
        cl_image_info.CL_IMAGE_ELEMENT_SIZE,
        cl_image_info.CL_IMAGE_ROW_PITCH,
        cl_image_info.CL_IMAGE_SLICE_PITCH,
        cl_image_info.CL_IMAGE_WIDTH,
        cl_image_info.CL_IMAGE_HEIGHT,
        cl_image_info.CL_IMAGE_DEPTH,
    ):
        param_value = size_t()
        clGetImageInfo.call(img, param_name, sizeof(param_value), byref(param_value), None)
        return int(param_value.value)
    else:  # cl_image_info.CL_IMAGE_D3D10_SUBRESOURCE_KHR
        param_value = void_p()
        clGetImageInfo.call(img, param_name, sizeof(param_value), byref(param_value), None)
        return param_value


###################
# Program Objects #
###################


class cl_program(void_p):
    """
    Represents an OpenCL program, a container for kernels.

    Use :func:`clCreateProgramWithSource` or
    :func:`clCreateProgramWithBinary` to make a program.

    Remember to call :meth:`build` to compile source programs.

    You can retrieve a kernel like so:
    >>> my_kernel = my_program['my_kernel'] # doctest: +SKIP

    Programs participate in reference counting.
    """

    def build(self, *args, **kw):
        """
        Calls :func:`clBuildProgram` on the program, passing
        along any arguments you provide. The program itself will
        be returned, so you can use this idiom:

        >>> source = 'kernel void foo(float bar) {}'
        >>> ctx = clCreateContext()
        >>> prog = clCreateProgramWithSource(ctx, source).build()
        """
        clBuildProgram(self, *args, **kw)
        return self

    def __getitem__(self, name):
        if not hasattr(self, "_kernels"):
            self._kernels = dict()
        if name in self._kernels:
            return self._kernels[name]
        else:
            return clCreateKernel(self, name)

    @property
    def context(self):
        """Returns the context the program exists within."""
        try:
            return self._context
        except AttributeError:
            return clGetProgramInfo(self, cl_program_info.CL_PROGRAM_CONTEXT)

    @property
    def reference_count(self):
        """Reference count for OpenCL garbage collector."""
        return clGetProgramInfo(self, cl_program_info.CL_PROGRAM_REFERENCE_COUNT)

    @property
    def num_devices(self):
        """Number of devices the program exists on."""
        return clGetProgramInfo(self, cl_program_info.CL_PROGRAM_NUM_DEVICES)

    @property
    def devices(self):
        """Devices on which the program exists."""
        return clGetProgramInfo(self, cl_program_info.CL_PROGRAM_DEVICES)

    @property
    def source(self):
        """Program's source code, if available."""
        return clGetProgramInfo(self, cl_program_info.CL_PROGRAM_SOURCE)

    @property
    def binary_sizes(self):
        """Sizes, in bytes, of the binaries for each of the
        devices the program is compiled for."""
        return clGetProgramInfo(self, cl_program_info.CL_PROGRAM_BINARY_SIZES)

    @property
    def binaries(self):
        """Acquires the binaries for each device."""
        return clGetProgramInfo(self, cl_program_info.CL_PROGRAM_BINARIES)

    def build_status(self, device=None):
        """
        Retrieves the :class:`cl_program_build_status` for one of more devices.
        See also :func:`clGetProgramBuildInfo`
        """
        return clGetProgramBuildInfo(self, cl_program_build_info.CL_PROGRAM_BUILD_STATUS, device)

    def build_options(self, device=None):
        """
        Retrieves the build options, as a string, for one of more devices.
        See also :func:`clGetProgramBuildInfo`.
        """
        return clGetProgramBuildInfo(self, cl_program_info.CL_PROGRAM_BUILD_OPTIONS, device)

    def build_log(self, device=None):
        """
        Returns the build log, as a string, for one or more devices.
        Mostly useful for checking compiler errors.
        See also :func:`clGetProgramBuildInfo`.
        """
        return clGetProgramBuildInfo(self, cl_program_build_info.CL_PROGRAM_BUILD_LOG, device)

    def __del__(self):
        try:
            if self:
                clReleaseProgram(self)
        except:
            pass


@_wrapdll(cl_program, cl_program_info, size_t, void_p, P(size_t))
def clGetProgramInfo(program, param_name):
    """
    :param program: :class:`cl_program`
    :param param_name: One of the :class:`cl_program_info` values.
    """
    if param_name == cl_program_info.CL_PROGRAM_CONTEXT:
        try:
            return program._context
        except AttributeError:
            param_value = cl_context()
            clGetProgramInfo.call(program, param_name, sizeof(param_value), param_value, None)
            clRetainContext(param_value)
            program._context = param_value
            return param_value
    elif param_name in (
        cl_program_info.CL_PROGRAM_REFERENCE_COUNT,
        cl_program_info.CL_PROGRAM_NUM_DEVICES,
    ):
        param_value = cl_uint()
        clGetProgramInfo.call(program, param_name, sizeof(param_value), byref(param_value), None)
        return int(param_value.value)
    elif param_name == cl_program_info.CL_PROGRAM_DEVICES:
        sz = size_t()
        clGetProgramInfo.call(program, param_name, 0, None, byref(sz))
        nd = sz.value // sizeof(cl_device)
        param_value = (cl_device * nd)()
        clGetProgramInfo.call(program, param_name, sz, param_value, None)
        return [x for x in param_value]
    elif param_name == cl_program_info.CL_PROGRAM_SOURCE:
        sz = size_t()
        clGetProgramInfo.call(program, param_name, 0, None, byref(sz))
        param_value = create_string_buffer(sz.value)
        clGetProgramInfo.call(program, param_name, sz, param_value, None)
        return param_value.value
    elif param_name == cl_program_info.CL_PROGRAM_BINARY_SIZES:
        sz = size_t()
        clGetProgramInfo.call(program, param_name, 0, None, byref(sz))
        nd = sz.value // sizeof(size_t)
        param_value = (size_t * nd)()
        clGetProgramInfo.call(program, param_name, sz, param_value, None)
        return [int(x) for x in param_value]
    elif param_name == cl_program_info.CL_PROGRAM_BINARIES:
        sz = size_t()
        clGetProgramInfo.call(program, param_name, 0, None, byref(sz))
        nd = sz.value // sizeof(char_p)
        param_value = (char_p * nd)()
        binary_sizes = clGetProgramInfo(program, cl_program_info.CL_PROGRAM_BINARY_SIZES)
        binaries = [None] * nd
        for i, bsize in enumerate(binary_sizes):
            binaries[i] = (ctypes.c_char * bsize)()
            param_value[i] = cast(binaries[i], char_p)
        clGetProgramInfo.call(program, param_name, sz, param_value, None)
        return [x.value for x in binaries]
    else:
        raise ValueError("Unknown program info %s" % param_name)


@_wrapdll(cl_program, cl_device, cl_program_build_info, size_t, void_p, P(size_t))
def clGetProgramBuildInfo(program, param_name, device=None):
    """
    :param program: The :class:`cl_program` to check.
    :param param_name: One of the :class:`cl_program_build_info` values.
    :param device: A :class:`cl_device` instance, or list of them.

    If a list of devices is provided, info will be returned
    for each of them in a list.

    If no device is specified, all devices associated with
    the program will be used.

    The :meth:`~cl_program.build_status`, :meth:`~cl_program.build_options`,
    and :meth:`~cl_program.build_log` methods of program objects are
    equivalent to using this, so they may be preferable.
    """
    if device is None:
        device = program.devices
    if not isinstance(device, cl_device):
        return [
            clGetProgramBuildInfo(program, param_name, each_device)
            for each_device in program.devices
        ]
    if param_name == cl_program_build_info.CL_PROGRAM_BUILD_STATUS:
        param_value = cl_build_status()
        clGetProgramBuildInfo.call(
            program, device, param_name, sizeof(param_value), byref(param_value), None
        )
        return param_value
    elif param_name in (
        cl_program_build_info.CL_PROGRAM_BUILD_OPTIONS,
        cl_program_build_info.CL_PROGRAM_BUILD_LOG,
    ):
        sz = size_t()
        clGetProgramBuildInfo.call(program, device, param_name, 0, None, byref(sz))
        param_value = create_string_buffer(sz.value)
        clGetProgramBuildInfo.call(program, device, param_name, sz, param_value, None)
        if sys.version_info[0] > 2:
            return str(param_value.value, "utf-8")
        else:
            return param_value.value
    else:
        raise ValueError("Unknown program build info %s" % param_name)


@_wrapdll(
    cl_context, cl_uint, P(char_p), P(size_t), P(cl_errnum), res=cl_program, err=_lastarg_errcheck
)
def clCreateProgramWithSource(context, source):
    """
    :param context: Context in which the program will exist
    :param source: Source code, as a string.

    Remember to call :meth:`~cl_program.build` on the program.
    """
    if sys.version_info[0] > 2 and isinstance(source, str):
        source = source.encode()
    c_source = char_p(source)
    p = pointer(c_source)
    # import pdb; pdb.set_trace()
    prg = clCreateProgramWithSource.call(context, 1, p, None, byref(cl_errnum()))
    prg._context = context
    return prg


@_wrapdll(cl_program, cl_uint, P(cl_device), P(char_p), void_p, void_p)
def clBuildProgram(program, options=None, devices=None):
    """
    Compiles a source program to run on one or more devices.

    :param program: The :class:`cl_program` to build.
    :param options: (optional) string with compiler options. See
      your OpenCL spec and platform provider's docs for possible values.
    :param devices: A list of devices to compile the program for. If not
      provided, it will be built for all devices in the context.

    If the build fails, it will raise a :exc:`ProgramBuildFailureError`
    with details.
    """
    if options is not None:
        options = char_p(options)
    if devices is not None:
        num_devices = len(devices)
        dev_array = (cl_device * num_devices)()
        for i, dev in enumerate(devices):
            dev_array[i] = dev
    else:
        devices = program.devices
        num_devices = 0
        dev_array = None
    try:
        clBuildProgram.call(program, num_devices, dev_array, options, None, None)
    except BuildProgramFailureError:  # this exception is dynamically created
        # Re-raise with appropriate message
        for dev in devices:
            if program.build_status(dev) == cl_build_status.CL_BUILD_ERROR:
                log = program.build_log(dev)
                raise BuildProgramFailureError(log)


@_wrapdll(cl_program)
def clRetainProgram(program):
    clRetainProgram.call(program)


@_wrapdll(cl_program)
def clReleaseProgram(program):
    clReleaseProgram.call(program)


##################
# Kernel Objects #
##################


class cl_kernel(void_p):
    """
    Represents an OpenCL kernel found in a :class:`cl_program`.

    After compiling a program, the kernels will be accessible as
    items whose keys are the kernel names.

    Kernels are reference counted.
    """

    def __del__(self):
        try:
            if self:
                clReleaseKernel(self)
        except:
            pass

    def __repr__(self):
        try:
            return "<cl_kernel %s %s>" % (self.name, self.argtypes)
        except:
            return "<cl_kernel 0x%x>" % (self.value or 0)

    @property
    def name(self):
        """Name of the kernel function."""
        try:
            return self._name
        except AttributeError:
            return clGetKernelInfo(self, cl_kernel_info.CL_KERNEL_FUNCTION_NAME)

    @property
    def program(self):
        """The :class:`cl_program` this kernel lives in."""
        try:
            return self._program
        except AttributeError:
            return clGetKernelInfo(self, cl_kernel_info.CL_KERNEL_PROGRAM)

    @property
    def context(self):
        """The :class:`cl_context` this kernel lives in."""
        try:
            return self._context
        except AttributeError:
            return clGetKernelInfo(self, cl_kernel_info.CL_KERNEL_CONTEXT)

    @property
    def num_args(self):
        """Number of arguments required to call this kernel."""
        try:
            return self._num_args
        except AttributeError:
            return clGetKernelInfo(self, cl_kernel_info.CL_KERNEL_NUM_ARGS)

    @property
    def reference_count(self):
        """Reference count for OpenCL garbage collector."""
        return clGetKernelInfo(self, cl_kernel_info.CL_KERNEL_REFERENCE_COUNT)

    def __call__(self, *args, **kw):
        """
        Equivalent to calling :meth:`setarg` for each of the
        arguments provided. No size parameter can be provided
        in this calling fashion, so be sure the datatypes are known
        or easily guessed by :meth:`setarg`.

        If the function takes a local memory argument, pass in
        an instance of :class:`localmem` to indicate the desired size.
        For example, to request that the third argument allocate 1KB of
        local memory: ``mykernel(foo, bar, localmem(1024))``

        The return value is the kernel itself, so that you can chain
        it with further methods like :meth:`on`.
        """
        for i, arg in enumerate(args):
            self.setarg(i, args[i])
        return self

    def setarg(self, index, value=None, size=None):
        """
        Sets one of the kernel's arguments.

        :param index: 0-based argument number to set.
        :param value: Value to set it to. Can be a :class:`cl_mem`,
          a Python int or float, or a :class:`localmem` object to
          indicate local memory allocation.
        :param size: The size of the parameter, in bytes. PyCL will
          attempt to guess if you don't tell it here or by setting
          :attr:`argtypes`. Guessing is bad.

        This does some extra work to try to ensure that the
        data is in a form suitable for the lower-level :func:`clSetKernelArg`
        call. The OpenCL API doesn't give us much help in determining
        what type an argument should be, so if possible you should set
        the elements of the kernel's :attr:`argtypes` field to a list of
        types. The types should be either :class:`cl_mem`, :class:`localmem`,
        a scalar type such as :class:`cl_int`, or a ctypes structure type.
        """
        if value is None and size is None:
            # Er, maybe the argument is a global pointer, and
            # the user wants it set to NULL?
            size = sizeof(cl_mem)
            dtype = cl_mem
        elif isinstance(value, localmem):
            # Local memory arguments must have a null pointer
            # and a size argument to indicate how many bytes
            # should be allocated on the device. As a convenience,
            # the user can pass in a localmem object, which serves
            # as a marker and holds the desired size. This isn't
            # necessary for this function, but is when using the
            # kernel's __call__ method.
            size = value.size
            value = None
            dtype = localmem
        elif isinstance(value, cl_mem):
            # Global memory object. Ask for its size if not specified.
            if size is None:
                size = sizeof(cl_mem)
            dtype = cl_mem
        else:
            # Otherwise, consult our records to see
            # what the appropriate c datatype should be.
            dtype = self.argtypes[index]
            if dtype is None:
                # We'll try to guess it further down
                pass
            elif dtype is localmem:
                # If the user placed localmem in the argtype,
                # they can just call the kernel with an integer
                # to indicate the desired size.
                if size is None:
                    size = value
                value = None
            elif dtype is not cl_mem:
                # Coerce the argument if necessary.
                if not isinstance(value, dtype):
                    value = dtype(value)
                size = sizeof(dtype)

        # Guess scalar datatypes.
        # OpenCL doesn't give us any means by which to query the type
        # or size of a kernel argument. If the user gives us a ctypes
        # value, we can assume they know what's up and use its type
        # and size. Otherwise, if we were given an int or a float, we
        # can infer the base type, but not the size... so we try each
        # of the sizes that type could possibly be. clSetKernelArg will
        # raise CL_INVALID_ARG_SIZE if we get it wrong, so we can choose
        # based on that. Since it obviously knows the right size, it would
        # be nice if it would just tell us, but at least this works.
        if dtype is None:
            warn(
                "Type not specified for %s argument %d. Guessing..." % (self.name, index),
                stacklevel=3,
            )
            if isinstance(value, ctypes._SimpleCData.__bases__[0]):
                candidate_types = (value.__class__,)
            if isinstance(value, float):
                candidate_types = (cl_float, cl_double)
            elif isinstance(value, str) and len(value) == 1:
                candidate_types = cl_char
            elif isinstance(value, int):
                candidate_types = (cl_int, cl_long, cl_short, cl_char)
            for t in candidate_types:
                try:
                    scalar_value = t(value)
                    clSetKernelArg.call(self, index, sizeof(t), byref(scalar_value))
                    # Hey, that worked. Record success.
                    dtype = t
                    value = scalar_value
                    size = sizeof(dtype)
                    self.argtypes[index] = dtype
                    break
                except InvalidArgSizeError:
                    # Nope, not this one.
                    continue
            else:
                raise ValueError(
                    "Could not guess kernel datatype for arg %d. "
                    "Please set it in kernel.argtypes[%d]." % (index, index)
                )

        if value is not None:
            vref = byref(value)
        else:
            vref = None
        clSetKernelArg.call(self, index, size, vref)

    def _get_argtypes(self):
        """
        Represents the data types of the kernel function arguments.
        There is no way to ask OpenCL for this information, so short of
        actually parsing the C code the only way to fill this in is to infer
        it from the way the user tries to call the kernel.

        Since this is error prone, we encourage you to fill in the list yourself.
        """
        if not hasattr(self, "_argtypes"):
            self._argtypes = [None] * self.num_args
        return self._argtypes

    def _set_argtypes(self, value):
        if len(value) != self.num_args:
            raise ValueError("Expected %d arguments." % self.num_args)
        self._argtypes = value

    argtypes = property(_get_argtypes, _set_argtypes, doc=_get_argtypes.__doc__)

    def on(self, queue, *args, **kw):
        """
        Enqueue the kernel (hopefully after setting its arguments)
        upon a command queue. This is essetially a shortcut for
        :func:`clEnqueueNDRangeKernel`.
        """
        return clEnqueueNDRangeKernel(queue, self, *args, **kw)

    def work_group_size(self, device=None):
        """
        The maximum size of workgroups for this kernel on the
        specified device.
        """
        return clGetKernelWorkGroupInfo(
            self, cl_kernel_work_group_info.CL_KERNEL_WORK_GROUP_SIZE, device
        )

    def compile_work_group_size(self, device=None):
        """
        The work group size specified by the kernel source, if any.
        Otherwise, will return (0,0,0).
        """
        return clGetKernelWorkGroupInfo(
            self, cl_kernel_work_group_info.CL_KERNEL_COMPILE_WORK_GROUP_SIZE, device
        )

    def local_mem_size(self, device=None):
        """
        The amount of local memory that would be used by this kernel
        on the given device with its current argument set.
        """
        return clGetKernelWorkGroupInfo(
            self, cl_kernel_work_group_info.CL_KERNEL_LOCAL_MEM_SIZE, device
        )

    def preferred_work_group_size_multiple(self, device=None):
        """
        Suggests a workgroup size multiplier for each dimension.
        That is, if a multiple is 8, then workgroup sizes should preferably
        be multiples of 8.
        """
        return clGetKernelWorkGroupInfo(
            self, cl_kernel_work_group_info.CL_KERNEL_PREFERRED_WORK_GROUP_SIZE_MULTIPLE, device
        )

    def private_mem_size(self, device=None):
        """
        Amount of private memory needed to execute each workitem on the device.
        """
        return clGetKernelWorkGroupInfo(
            self, cl_kernel_work_group_info.CL_KERNEL_PRIVATE_MEM_SIZE, device
        )


@_wrapdll(cl_program, char_p, P(cl_errnum), res=cl_kernel, err=_lastarg_errcheck)
def clCreateKernel(program, kernel_name):
    """
    :param program: :class:`cl_program`
    :param kernel_name: String naming a kernel function in the program.

    Using the the ``program[kernel_name]`` syntax is preferable.
    """
    if sys.version_info[0] > 2 and isinstance(kernel_name, str):
        kernel_name = kernel_name.encode()
    kernel = clCreateKernel.call(program, char_p(kernel_name), byref(cl_errnum()))
    kernel._program = program
    kernel._context = program.context
    if not hasattr(program, "_kernels"):
        program._kernels = dict()
    program._kernels[kernel_name] = kernel
    return kernel


@_wrapdll(cl_kernel, cl_kernel_info, size_t, void_p, P(size_t))
def clGetKernelInfo(kernel, param_name):
    """
    :param kernel: :class:`cl_kernel`
    :param param_name: One of the :class:`cl_kernel_info` values.

    Kernel objects have properties that call this function, so it
    is probably preferable to use those instead.
    """
    if param_name == cl_kernel_info.CL_KERNEL_FUNCTION_NAME:
        sz = size_t()
        clGetKernelInfo.call(kernel, param_name, 0, None, byref(sz))
        param_value = create_string_buffer(sz.value)
        clGetKernelInfo.call(kernel, param_name, sz, param_value, None)
        return param_value.value
    elif param_name == cl_kernel_info.CL_KERNEL_CONTEXT:
        param_value = cl_context()
        clGetKernelInfo.call(kernel, param_name, sizeof(param_value), byref(param_value), None)
        clRetainContext(param_value)
        return param_value
    elif param_name == cl_kernel_info.CL_KERNEL_PROGRAM:
        try:
            return kernel._program
        except AttributeError:
            param_value = cl_program()
            clGetKernelInfo.call(kernel, param_name, sizeof(param_value), byref(param_value), None)
            clRetainProgram(param_value)
            kernel._program = program
            return param_value
    elif param_name == cl_kernel_info.CL_KERNEL_CONTEXT:
        try:
            return kernel._context
        except AttributeError:
            param_value = cl_context()
            clGetKernelInfo.call(kernel, param_name, sizeof(param_value), byref(param_value), None)
            clRetainContext(param_value)
            kernel._context = context
            return param_value
    elif param_name == cl_kernel_info.CL_KERNEL_NUM_ARGS:
        try:
            return kernel._num_args
        except AttributeError:
            param_value = cl_uint()
            clGetKernelInfo.call(kernel, param_name, sizeof(param_value), byref(param_value), None)
            kernel._num_args = int(param_value.value)
            return kernel._num_args
    elif param_name == cl_kernel_info.CL_KERNEL_REFERENCE_COUNT:
        param_value = cl_uint()
        clGetKernelInfo.call(kernel, param_name, sizeof(param_value), byref(param_value), None)
        return int(param_value.value)
    else:
        raise ValueError("Unknown kernel info type %s" % param_name)


@_wrapdll(cl_kernel, cl_device, cl_kernel_work_group_info, size_t, void_p, P(size_t))
def clGetKernelWorkGroupInfo(kernel, param_name, device=None):
    """
    :param kernel: :class:`cl_kernel`
    :param param_name: One of the :class:`cl_kernel_work_group_info` values.
    :param device: :class:`cl_device`. If no device is specified, the first
      device in the kernel's context is queried.

    Retrieves information about the kernel specific to a particular device
    that it might be run on. This information is also available through
    specific methods of kernel objects, which may be preferable to calling this.
    """
    if device is None:
        device = kernel.context.devices[0]
    if param_name in (
        cl_kernel_work_group_info.CL_KERNEL_WORK_GROUP_SIZE,
        cl_kernel_work_group_info.CL_KERNEL_PREFERRED_WORK_GROUP_SIZE_MULTIPLE,
    ):
        param_value = size_t()
        clGetKernelWorkGroupInfo.call(
            kernel, device, param_name, sizeof(param_value), byref(param_value), None
        )
        return int(param_value.value)
    elif param_name in (
        cl_kernel_work_group_info.CL_KERNEL_LOCAL_MEM_SIZE,
        cl_kernel_work_group_info.CL_KERNEL_PRIVATE_MEM_SIZE,
    ):
        param_value = cl_ulong()
        clGetKernelWorkGroupInfo.call(
            kernel, device, param_name, sizeof(param_value), byref(param_value), None
        )
        return int(param_value.value)
    elif param_name == cl_kernel_work_group_info.CL_KERNEL_COMPILE_WORK_GROUP_SIZE:
        param_value = (size_t * 3)()
        clGetKernelWorkGroupInfo.call(
            kernel, device, param_name, sizeof(param_value), byref(param_value), None
        )
        return [int(x) for x in param_value]
    else:
        raise ValueError("Unknown param name %s" % param_name)


@_wrapdll(cl_kernel)
def clRetainKernel(kernel):
    clRetainKernel.call(kernel)


@_wrapdll(cl_kernel)
def clReleaseKernel(kernel):
    clReleaseKernel.call(kernel)


class localmem(object):
    """
    When a kernel defines an argument to be in local memory,
    no value can be passed in to that argument. Instead, the
    size of the local memory is specified. While you could do
    this directly with :func:`clSetKernelArg`, localmem allows
    you to set this using the kernel call syntax. So if you had
    a kernel whose third argument was a local memory pointer,
    you could set the arguments like so:

    >>> mykernel(x, y, localmem(1024)) # doctest: +SKIP

    localmem is also accepted in :attr:`~cl_kernel.argtypes`, in
    which case the kernel can be called using just the desired size:

    >>> mykernel.argtypes = (cl_mem, cl_mem, localmem) # doctest: +SKIP
    >>> mykernel(x, y, 1024) # doctest: +SKIP
    """

    def __init__(self, size):
        self.size = size


@_wrapdll(cl_kernel, cl_uint, size_t, void_p)
def clSetKernelArg(kernel, index, value=None, size=None):
    """
    :param kernel: :class:`cl_kernel`
    :param index: 0-based argument index to set.
    :param value: Should be None or a pointer to a ctypes
      scalar or a :class:`cl_mem` object. Does not accept :class:`localmem`.
    :param size: Size in bytes of the referenced value. That is,
      if the argument is a 32-bit integer, this should be 4. If
      the argument is a :class:`cl_mem`, it should be ``sizeof(cl_mem)``.

    Unlike most of the wrappers in PyCL, this one doesn't do
    much to help you out. Use :meth:`cl_kernel.setarg` if you want
    some help setting individual arguments. Calling the kernel
    object itself with the desired argument sequence is more preferable
    still. Set :attr:`cl_kernel.argtypes` if it can't guess the types properly.
    """
    clSetKernelArg.call(kernel, index, size, value)


@_wrapdll(
    cl_command_queue,
    cl_kernel,
    cl_uint,
    P(size_t),
    P(size_t),
    P(size_t),
    cl_uint,
    P(cl_event),
    P(cl_event),
)
def clEnqueueNDRangeKernel(queue, kernel, gsize=(1,), lsize=None, offset=None, wait_for=None):
    """
    Enqueue a kernel for execution. The kernel's arguments should
    be set already. For a more idiomatic calling syntax, set the
    kernel arguments by calling it and use its :meth:`~cl_kernel.on`
    method to queue it.

    :param queue: :class:`cl_command_queue` to enqueue it upon.
    :param kernel: The :class:`cl_kernel` object you want to run.
    :param gsize: Global work size. A 1-, 2-, or 3-tuple of integers
      indicating the dimensions of the work to be done.
      A scalar is fine too. Default is a single work item.
    :param lsize: Local work size. Should have the same dimension as
      ``gsize``. If None (the default), OpenCL will pick a size for you.
    :param offset: Global work item offset. By default, the global id of
      work items start at 0 in each dimension. Provide a tuple of the same
      dimension as ``gsize`` to offset the ids.
    :param wait_for: A :class:`cl_event` or list of them that should complete
      prior to this kernel's execution.
    :returns: :class:`cl_event` which will identify when the kernel has completed.

    Note that the OpenCL :func:`clEnqueueTask` function is equivalent to calling
    this function with the default gsize, lsize, and offset values, so we haven't
    bothered to wrap it.
    """
    if isinstance(gsize, int):
        nd = 1
        gsize = (gsize,)
    else:
        nd = len(gsize)
    gsize_array = (size_t * nd)()
    for i, s in enumerate(gsize):
        gsize_array[i] = s
    if lsize is None:
        lsize_array = None
    else:
        if isinstance(lsize, int):
            lsize = (lsize,)
        lsize_array = (size_t * nd)()
        for i, s in enumerate(lsize):
            lsize_array[i] = s
    if offset is None:
        offset_array = None
    else:
        if isinstance(offset, int):
            offset = (offset,)
        offset_array = (size_t * nd)()
        for i, s in enumerate(offset):
            offset_array[i] = s
    nevents, wait_array = _make_event_array(wait_for)
    out_event = cl_event()
    clEnqueueNDRangeKernel.call(
        queue,
        kernel,
        nd,
        offset_array,
        gsize_array,
        lsize_array,
        nevents,
        wait_array,
        byref(out_event),
    )
    return out_event


@_wrapdll(cl_command_queue)
def clFinish(queue):
    clFinish.call(queue)


try:
    from OpenGL import GL

    HAVE_OPENGL = True
except ImportError:
    HAVE_OPENGL = False

if HAVE_OPENGL:

    @_wrapdll(
        cl_context, cl_mem_flags, GL.GLuint, P(cl_errnum), res=cl_buffer, err=_lastarg_errcheck
    )
    def clCreateFromGLBuffer(context, bufobj, flags=cl_mem_flags.CL_MEM_READ_WRITE):
        return clCreateFromGLBuffer.call(context, flags, bufobj, byref(cl_errnum()))

    @_wrapdll(cl_command_queue, cl_uint, P(cl_mem), cl_uint, P(cl_event), P(cl_event))
    def clEnqueueAcquireGLObjects(queue, mem_objs, wait_for=None):
        nevents, wait_array = _make_event_array(wait_for)
        out_event = cl_event()
        mem_obj_array = (cl_mem * len(mem_objs))(*mem_objs)
        clEnqueueAcquireGLObjects.call(
            queue, len(mem_objs), mem_obj_array, nevents, wait_array, byref(out_event)
        )
        return out_event

    @_wrapdll(cl_command_queue, cl_uint, P(cl_mem), cl_uint, P(cl_event), P(cl_event))
    def clEnqueueReleaseGLObjects(queue, mem_objs, wait_for=None):
        nevents, wait_array = _make_event_array(wait_for)
        out_event = cl_event()
        mem_obj_array = (cl_mem * len(mem_objs))(*mem_objs)
        clEnqueueReleaseGLObjects.call(
            queue, len(mem_objs), mem_obj_array, nevents, wait_array, byref(out_event)
        )
        return out_event


def buffer_from_ndarray(queue, ary, buf=None, **kw):
    """
    Creates (or simply writes to) an OpenCL buffer using the contents
    of a Numpy array.

    :param queue: :class:`cl_command_queue` to enqueue the write to.
    :param ary: :class:`numpy.ndarray` object, or other object implementing
      the array interface. We haven't wrapped the rectangular read/write
      functions yet, so if the array isn't contiguous, a copy will be made.
      Note that the entirety of the provided array will be written, so be sure
      to slice it down to just the part you want to write.
    :param buf: :class:`cl_buffer` object. If not provided, one the size
      of the array will be created. In any event, it should hopefully be large
      enough to hold the provided array.
    :returns: ``(buf, evt)``, where ``evt`` is the :class:`cl_event` returned
      by the write operation.

    Any additional provided keyword arguments are passed along to
    :func:`clEnqueueWriteBuffer`.
    """
    if not np:
        raise Exception("numpy not available")

    ary = np.ascontiguousarray(ary)
    if buf is None:
        buf = clCreateBuffer(queue.context, ary.nbytes)
    if ary.__array_interface__["strides"]:
        raise ValueError("I don't know how to handle strided arrays yet.")
    ptr = void_p(ary.__array_interface__["data"][0])
    evt = clEnqueueWriteBuffer(queue, buf, ptr, ary.nbytes, **kw)
    return (buf, evt)


def buffer_to_ndarray(queue, buf, out=None, like=None, dtype="uint8", shape=None, **kw):
    """
    Reads from an OpenCL buffer into an ndarray.

    :param queue: The queue to put the read operation on.
    :param buf: The :class:`cl_buffer` to read from
    :param out: The :class:`numpy.ndarray` to read into. If not
      provided, one will be created based on the following arguments.
      Unlike :func:`buffer_from_array`, this must currently be an actual
      contiguous :class:`~numpy.ndarray` object.
    :param like: Only relevant if no out array is provided. The new array
      will have the same shape and dtype as this value.
    :param dtype: Only relevant if no out array or ``like`` parameter are provided.
      A :class:`numpy.dtype` or anything that can pass for one. Defaults to ``'uint8'``.
    :param shape: Only relevant if no out array or ``like`` parameter are provided.
      Integer or tuple determining the array's shape. If no shape is given, the
      array will be 1d and will have a number of elements based on the buffer's
      size and the itemsize of the dtype.
    :returns: ``(ary, evt)``, where ``evt`` is the :class:`cl_event` returned by the
      read operation.

    Any further keyword arguments are passed directly to
    :func:`clEnqueueReadBuffer`.
    """
    if out is None:
        if like is not None:
            if not np:
                raise Exception("numpy not available")

            out = np.empty_like(like)
        else:
            if not np:
                raise Exception("numpy not available")

            dtype = np.dtype(dtype)
            if shape is None:
                shape = buf.size // dtype.itemsize
            out = np.empty(shape, dtype)
    assert out.flags.contiguous, "Don't know how to write non-contiguous yet."
    ptr = void_p(out.__array_interface__["data"][0])
    evt = clEnqueueReadBuffer(queue, buf, ptr, out.nbytes, **kw)
    return (out, evt)


def buffer_from_pyarray(queue, ary, buf=None, **kw):
    """
    Essentially the same as :func:`buffer_from_ndarray`, except that
    it accepts arrays from the :mod:`array` module in Python's standard library.
    """
    (ptr, length) = ary.buffer_info()
    nbytes = length * ary.itemsize
    if buf is None:
        buf = clCreateBuffer(queue.context, nbytes)
    evt = clEnqueueWriteBuffer(queue, buf, ptr, nbytes, **kw)
    return (buf, evt)


def buffer_to_pyarray(queue, buf, out=None, like=None, typecode="B", length=None, **kw):
    """
    Essentially the same as :func:`buffer_to_ndarray`, except that it
    produces arrays from the :mod:`array` module in Python's standard library.
    The ``dtype`` and ``shape`` parameters are replaced:

    :param typecode: A character indicating the array typecode. See the
      documentation_ for the mappings to C data types. The default is 'B',
      for unsigned bytes.
    :param length: The number of elements that should be in the array. If
      not provided, it will be determined based on the buffer size and the
      size of the selected typecode.

    .. _documentation: http://docs.python.org/library/array.html
    """
    if out is None:
        if like is not None:
            out = array(like.typecode, like)
        else:
            if length is None:
                length = buf.size // array(typecode).itemsize
            out = array(typecode, [0]) * length
    (ptr, length) = out.buffer_info()
    nbytes = length * out.itemsize
    evt = clEnqueueReadBuffer(queue, buf, ptr, nbytes)
    return (out, evt)


### End OpenCL wrappers. ###
def _pycl_make_all():
    g = globals()
    __all__ = [
        name
        for name in g
        if not (name.startswith("_"))
        and (hasattr(g[name], "__module__") and g[name].__module__ == __name__)
    ]
    g["__all__"] = __all__


_pycl_make_all()


def pycl_main():
    import sys

    if "--doctest" in sys.argv:
        import doctest

        doctest.testmod()
    else:
        print("Using %s" % _dll_filename)
        platforms = clGetPlatformIDs()
        for p in platforms:
            print("Platform: %s" % p.name)
            print("        Vendor: %s" % p.vendor)
            print("       Version: %s" % p.version)
            print("       Profile: %s" % p.profile)
            print("    Extensions: %s" % ", ".join(p.extensions))
            for d in clGetDeviceIDs(p):
                print("=" * 80)
                print("    %s: %s" % (str(d.type)[15:], d.name))
                print("        Vendor: %s" % d.vendor)
                print("       Version: %s" % d.version)
                print("       Profile: %s" % d.profile)
                print("        Driver: %s" % d.driver_version)
                for ul in _device_info_ulongs:
                    print("       {}: {}".format(ul, clGetDeviceInfo(d, ul)))
                for ul in _device_info_sizes:
                    print("       {}: {}".format(ul, clGetDeviceInfo(d, ul)))
                for ul in _device_info_bools:
                    print("       {}: {}".format(ul, clGetDeviceInfo(d, ul)))
                for ul in _device_info_counts:
                    print("       {}: {}".format(ul, clGetDeviceInfo(d, ul)))
                # print("    Extensions: %s" % ", ".join(d.extensions))


def pycl_test2():
    from array import array

    source = """
    kernel void mxplusb(float m, global float *x, float b, global float *out) {
        int i = get_global_id(0);
        out[i] = m*x[i]+b;
    }
    """
    ctx = clCreateContext()
    queue = clCreateCommandQueue(ctx)
    program = clCreateProgramWithSource(ctx, source).build()
    kernel = program["mxplusb"]
    kernel.argtypes = (cl_float, cl_mem, cl_float, cl_mem)
    x = array("f", range(10))
    x_buf, in_evt = buffer_from_pyarray(queue, x, blocking=False)
    y_buf = x_buf.empty_like_this()
    print(type(kernel), dir(kernel))
    run_evt = kernel(2, x_buf, 5, y_buf).on(queue, len(x), wait_for=in_evt)
    y, evt = buffer_to_pyarray(queue, y_buf, wait_for=run_evt, like=x)
    evt.wait()
    print(y)


def pycl_test3():
    import numpy as np

    a_np = np.random.rand(50000).astype(np.float32)
    b_np = np.random.rand(50000).astype(np.float32)

    ctx = clCreateContext()
    queue = clCreateCommandQueue(ctx)

    a_g, a_inevt = buffer_from_ndarray(queue, a_np)
    b_g, b_inevt = buffer_from_ndarray(queue, b_np)

    prg = clCreateProgramWithSource(
        ctx,
        """
    __kernel void sum(
        __global const float *a_g, __global const float *b_g, __global float *res_g)
    {
    int gid = get_global_id(0);
    res_g[gid] = a_g[gid] + b_g[gid];
    }
    """,
    ).build()

    kernel = prg["sum"]
    kernel.argtypes = (cl_mem, cl_mem, cl_mem)

    res_g = a_g.empty_like_this()
    run_evt = kernel(a_g, b_g, res_g, wait_for=[a_inevt, b_inevt]).on(queue, len(a_np))

    res_np = np.empty_like(a_np)
    buffer_to_ndarray(queue, res_g, res_np, wait_for=run_evt)

    # Check on CPU with Numpy:
    print(res_np)
    print(res_np - (a_np + b_np))
    print(np.linalg.norm(res_np - (a_np + b_np)))
    assert np.allclose(res_np, a_np + b_np)


if __name__ == "__main__":
    pycl_test3()
