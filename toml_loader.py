from . import pycl as cl
import glob

# import json
from . import toml
import os

# TODO: this is pretty unmaintainable. Fix


class NodeCLKernel:
    cl_types = {
        "float": ("float", cl.cl_float),
        "int": ("int", cl.cl_int),
        "float2d": ("__global float*", cl.cl_mem),
        "float1d": ("__global float*", cl.cl_mem),
        # "int2d": ("__global int*", cl.cl_mem),
        "image": ("image2d_t", cl.cl_image),
    }

    kernels = {}

    # TODO: class_from_json() class method

    def _builder(self):
        # Build function definition values
        # Save function signature
        # Find array dimensions from either output or input
        pstr = []
        signature = [cl.cl_int, cl.cl_int]

        # Add parameters
        for k, v in self.cl_def["params"].items():
            signature.append(self.cl_types[v][1])
            pstr.append(f"const {v} {k}")
            pstr.append(",\n")

        # Add input and output arrays
        array_or_image = [("const", ""), ("__read_only", "__write_only")]
        shape_sources = set(["float2d", "image"])
        self.shape_source = ("", None)
        for i, (k, v) in enumerate(self.cl_def["inputs"].items()):
            assert v == "image" or v == "float2d"
            signature.append(self.cl_types[v][1])
            if v in shape_sources:
                self.shape_source = (0, i)
            a = array_or_image[(v == "image") * 1]
            pstr.append(f"{a[0]} {self.cl_types[v][0]} {k}")
            pstr.append(",\n")

        for i, (k, v) in enumerate(self.cl_def["outputs"].items()):
            assert v == "image" or v == "float2d"
            signature.append(self.cl_types[v][1])
            if v in shape_sources:
                self.shape_source = (1, i)
            a = array_or_image[(v == "image") * 1]
            pstr.append(f"{a[1]} {self.cl_types[v][0]} {k}")
            pstr.append(",\n")

        assert self.shape_source[1] is not None

        # Remove last ", "
        pstr = pstr[:-1]

        # Build kernel definition
        src = f"""
        #define POW2(a) ((a) * (a))
        #define READP(input,loc) read_imagef(input, sampler, loc)
        #define READF(input,loc) input[loc.x + loc.y * width]
        #define WRITEP(output,loc,value) write_imagef(output, loc, value)
        #define WRITEF(output,loc,value) output[loc.x + loc.y * width] = value
        __kernel void {self.cl_def["kernel"]["name"]}(
            const int width,
            const int height,
            {"".join(pstr)})
        {{
            const sampler_t sampler = \\
                CLK_NORMALIZED_COORDS_FALSE |
                CLK_ADDRESS_CLAMP_TO_EDGE |
                CLK_FILTER_NEAREST;
            const int gx = get_global_id(0), gy = get_global_id(1);
            const int2 loc = (int2)(gx, gy);
            {self.cl_def["kernel"]["source"]}
        }}
        """
        # print(src)
        return self.cl_builder.build(self.cl_def["kernel"]["name"], src, tuple(signature))

    def __init__(self, definition=None, builder=None, seamless=True, lazy=False):
        self.signature = None
        self.kernel = None
        self._operational = False

        assert definition is not None
        assert builder is not None
        self.cl_builder = builder
        self.cl_def = definition

        # Check all definitions for new kernel exist
        scdict = definition
        problem = False
        problem |= "kernel" not in scdict
        problem |= "source" not in scdict["kernel"]
        problem |= "name" not in scdict["kernel"]
        problem |= "params" not in scdict
        problem |= "inputs" not in scdict
        problem |= "outputs" not in scdict
        if problem:
            raise ValueError(
                "NodeCLKernel definition missing some required parameters. "
                "[kernel_name, params, inputs, outputs, source]"
            )

        # Check no name collisions
        if self.cl_def["kernel"]["name"] in NodeCLKernel.kernels:
            raise ValueError("NodeCLKernel kernel name already exists. ")

        # Option to init class now and build kernel later (lazily)
        if not lazy:
            self._lazy_init()

    def _lazy_init(self):
        self.kernel = self._builder()
        NodeCLKernel.kernels[self.cl_def["kernel"]["name"]] = self.kernel
        self._operational = True

    def run(self, params, inputs, outputs):
        if not self._operational:
            self._lazy_init()
        choice, loc = self.shape_source
        s = (inputs, outputs)[choice][loc].shape
        self.cl_builder.run(
            self.kernel, params, [i.data for i in inputs], [o.data for o in outputs], shape=s
        )


def load(cl_builder):
    # Turn every TOML file in cl_nodes folder into OpenCL functions
    cl_nodes = {}
    directory = os.path.dirname(os.path.realpath(__file__))
    glob_filter = os.path.join(directory, "cl_nodes", "*.toml")
    # for bpath in glob.glob("cl_nodes/*.toml"):
    print(glob_filter)
    for bpath in glob.glob(glob_filter):
        try:
            bname = os.path.basename(bpath)
            node_name = "".join(bname.split(".")[:-1])

            with open(bpath, "r") as jf:
                jres = toml.load(jf)
            cl_nodes[node_name] = type("CL_" + node_name.capitalize(), (NodeCLKernel,), {})(
                definition=jres, builder=cl_builder, lazy=True
            )
        except ValueError as e:
            print(f"Failed loading {bpath}:")
            print(e)
            assert node_name not in cl_nodes
    return cl_nodes
