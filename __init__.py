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
    "version": (0, 1, 15),
    "blender": (2, 80, 0),
}

import numpy
import bpy
import copy
from . import image_ops
import importlib

importlib.reload(image_ops)


def compute_shading():
    tri_intersect = """
    // vec3 uvt;
    bool intersectTri(vec3 orig, vec3 dir, vec3 a, vec3 b, vec3 c, out vec3 uvt){
        // möller-trumbore
        const float eps = 0.0000001;
        vec3 ab=b-a;
        vec3 ac=c-a;

        vec3 n=cross(dir,ac);
        float det=dot(ab,n);

        // if the determinant is negative the triangle is backfacing
        // if the determinant is close to 0, the ray misses the triangle
        if(-eps < det && det < eps) { return false; }

        float invdet = 1.0/det;

        vec3 ao=orig-a;
        float u=dot(ao,n) * invdet;
        if(u < 0.0 || u > 1.0) { return false; }

        vec3 e=cross(ao,ab);
        float v=dot(dir,e) * invdet;
        if(v < 0.0 || u+v > 1.0) { return false; }

        float t=-dot(ac,e) * invdet;
        uvt = vec3(u,v,t);
        return true;
    }

    // vec4 local_point = (1.0 - uvt.x - uvt.y) * a + uvt.x * b + uvt.y * c;
    """

    box_intersect = """
    vec2 intersectBox(vec3 orig, vec3 dir, const box b) {
        vec3 tMin = (b.min - orig) / dir;
        vec3 tMax = (b.max - orig) / dir;
        vec3 t1 = min(tMin, tMax);
        vec3 t2 = max(tMin, tMax);
        float tNear = max(max(t1.x, t1.y), t1.z);
        float tFar = min(min(t2.x, t2.y), t2.z);
        return vec2(tNear, tFar);
    }
    """

    import bgl
    import numpy as np

    def install_lib(libname):
        from subprocess import call

        pp = bpy.app.binary_path_python
        call([pp, "-m", "ensurepip", "--user"])
        call([pp, "-m", "pip", "install", "--user", libname])

    # uncomment to install required lib
    # install_lib("moderngl")
    import moderngl

    def get_teximage(context):
        teximage = None
        for area in context.screen.areas:
            if area.type == "IMAGE_EDITOR":
                teximage = area.spaces.active.image
                break
        if teximage is not None and teximage.size[1] != 0:
            return teximage
        else:
            return None

    print("OpenGL supported version (by Blender):", bgl.glGetString(bgl.GL_VERSION))
    ctx = moderngl.create_context(require=430)
    print("GL context version code:", ctx.version_code)
    assert ctx.version_code >= 430
    print("Compute max work group size:", ctx.info["GL_MAX_COMPUTE_WORK_GROUP_SIZE"], end="\n\n")

    basic_shader = """
    #version 430
    #define TILE_WIDTH 8
    #define TILE_HEIGHT 8

    const ivec2 tileSize = ivec2(TILE_WIDTH, TILE_HEIGHT);

    layout(local_size_x=TILE_WIDTH, local_size_y=TILE_HEIGHT) in;
    layout(binding=0) writeonly buffer out_0 { vec4 outp[][1]; };

    uniform uint img_size;

    void main() {
        const ivec2 tile_xy = ivec2(gl_WorkGroupID);
        const ivec2 thread_xy = ivec2(gl_LocalInvocationID);
        const ivec2 pixel_xy = tile_xy * tileSize + thread_xy;

        const float tx = float(pixel_xy.x)/img_size;
        const float ty = float(pixel_xy.y)/img_size;

        vec4 outc = vec4(tx, ty, int(tx) & int(ty), 1.0);

        outp[pixel_xy.x][pixel_xy.y * img_size] = outc;

    }
    """

    image = get_teximage(bpy.context)
    sourcepixels = np.empty((image.size[0], image.size[1], 4), dtype=np.float32)
    compute_shader = ctx.compute_shader(basic_shader)

    print("start compute")
    buffer = ctx.buffer(sourcepixels)
    buffer.bind_to_storage_buffer(0)
    compute_shader.get("img_size", 5).value = image.size[0]
    compute_shader.run(group_x=image.size[0] // 8, group_y=image.size[1] // 8)
    print("end compute")

    image.pixels = np.frombuffer(buffer.read(), dtype=np.float32)

    print("fin.")


class ImageOperations:
    """ Takes in RGBA """

    def __init__(self, image):
        # self.pixels = numpy.copy(image)
        self.pixels = image

    def copy(self):
        n = copy.copy(self)
        n.pixels = numpy.copy(self.pixels)
        return self

    def convolution(self, intens, sfil):
        # source, intensity, convolution matrix
        ssp = self.pixels
        tpx = numpy.zeros(ssp.shape, dtype=float)
        tpx[:, :, 3] = 1.0
        ystep = int(4 * ssp.shape[1])
        norms = 0
        for y in range(sfil.shape[0]):
            for x in range(sfil.shape[1]):
                tpx += (
                    numpy.roll(
                        ssp, (x - int(sfil.shape[1] / 2)) * 4 + (y - int(sfil.shape[0] / 2)) * ystep
                    )
                    * sfil[y, x]
                )
                norms += sfil[y, x]
        if norms > 0:
            tpx /= norms
        return ssp + (tpx - ssp) * intens

    def blur(self, s, intensity):
        self.pixels = self.convolution(intensity, numpy.ones((1 + s * 2, 1 + s * 2), dtype=float))
        return self

    def sharpen(self, intensity):
        self.pixels = self.convolution(
            intensity, numpy.array([[-1, -1, -1], [-1, 9, -1], [-1, -1, -1]])
        )
        return self

    def normalize(self):
        t = self.pixels - numpy.min(self.pixels)
        self.pixels = t / numpy.max(t)
        return self

    def sobel_x(self, intensity):
        gx = numpy.array([[-1, 0, 1], [-2, 0, 2], [-1, 0, 1]])
        self.pixels = self.convolution(intensity, gx)
        return self

    def sobel_y(self, intensity):
        gy = numpy.array([[1, 2, 1], [0, 0, 0], [-1, -2, -1]])
        self.pixels = self.convolution(intensity, gy)
        return self

    def edgedetect(self, s, intensity):
        self.pixels = (
            self.convolution(intensity, numpy.array([[0, 1, 0], [1, -4, 1], [0, 1, 0]]))
        ) * 0.5 + 0.5
        return self

    def gaussian(self, s, intensity):
        fil = numpy.ones((1 + s * 2, 1 + s * 2), dtype=float)
        xs = int(fil.shape[1] / 2)
        ys = int(fil.shape[0] / 2)
        ro = 5.0 ** 2
        for y in range(0, fil.shape[0]):
            for x in range(0, fil.shape[1]):
                fil[y, x] = (1.0 / numpy.sqrt(2 * numpy.pi * ro)) * (
                    2.71828 ** (-((x - xs) ** 2 + (y - ys) ** 2) / (2 * ro))
                )
        self.pixels = self.convolution(intensity, fil)
        return self

    def fast_blur(self, s):
        d = 2 ** s
        tpx = self.pixels
        ystep = tpx.shape[1]
        while d > 1:
            tpx = (tpx * 2 + numpy.roll(tpx, -d * 4) + numpy.roll(tpx, d * 4)) / 4
            tpx = (
                tpx * 2 + numpy.roll(tpx, -d * (ystep * 4)) + numpy.roll(tpx, d * (ystep * 4))
            ) / 4
            d = int(d / 2)
        self.pixels = tpx
        return self

    def normals_simple(self, intensity):
        self.grayscale()
        sshape = self.pixels.shape

        px = self.copy().sobel_x(1.0).pixels
        px[:, :, 2] = px[:, :, 2] * intensity
        px[:, :, 1] = 0
        px[:, :, 0] = 1

        py = self.sobel_y(1.0).pixels
        py[:, :, 2] = py[:, :, 2] * intensity
        py[:, :, 1] = 1
        py[:, :, 0] = 0

        arr = numpy.cross(px[:, :, :3], py[:, :, :3])

        # vec *= 1/len(vec)
        m = 1.0 / numpy.sqrt(arr[:, :, 0] ** 2 + arr[:, :, 1] ** 2 + arr[:, :, 2] ** 2)
        arr[..., 0] *= m
        arr[..., 1] *= m
        arr[..., 2] *= m
        vectors = arr

        retarr = numpy.zeros(sshape)
        retarr[:, :, 0] = 0.5 - vectors[:, :, 0]
        retarr[:, :, 1] = vectors[:, :, 1] + 0.5
        retarr[:, :, 2] = vectors[:, :, 2]
        retarr[:, :, 3] = 1.0
        self.pixels = retarr
        return self

    def sobel(self, s, intensity):
        retarr = numpy.zeros(self.pixels.shape)
        retarr = self.sobel_x(1.0)
        retarr += self.sobel_y(1.0)
        retarr = (retarr * intensity) * 0.5 + 0.5
        retarr[..., 3] = 1.0
        self.pixels = retarr
        return self

    def separate_values(self, s, intensity):
        retarr = numpy.copy(self.pixels)
        retarr[..., :3] = self.pixels[..., :3] ** intensity
        self.pixels = retarr
        return self

    def grayscale(self):
        ssp = self.pixels
        r, g, b = ssp[:, :, 0], ssp[:, :, 1], ssp[:, :, 2]
        gray = 0.2989 * r + 0.5870 * g + 0.1140 * b
        retarr = numpy.copy(self.pixels)
        retarr[..., 0] = gray
        retarr[..., 1] = gray
        retarr[..., 2] = gray
        self.pixels = retarr
        return self


def old():
    class GeneralImageOperator(bpy.types.Operator):
        def init_images(self, context):
            self.input_image = context.scene.seamless_input_image
            self.target_image = context.scene.seamless_generated_name
            self.seamless_powersoftwo = bpy.context.scene.seamless_powersoftwo

            print("Creating images...")
            self.size = bpy.data.images[self.input_image].size
            self.xs = self.size[0]
            self.ys = self.size[1]

            # copy image data into much more performant numpy arrays
            self.sourcepixels = numpy.array(bpy.data.images[self.input_image].pixels)
            self.sourcepixels = self.sourcepixels.reshape((self.ys, self.xs, 4))

            # if limit to powers of two is selected, do it
            offx = 0
            offy = 0
            if self.seamless_powersoftwo:
                print("crop to 2^")
                lxs = int(numpy.log2(self.xs))
                lys = int(numpy.log2(self.ys))
                offx = int((self.xs - 2 ** lxs) / 2)
                offy = int((self.ys - 2 ** lys) / 2)
                print("crop offset:" + repr(offx) + "," + repr(offy))

                if self.xs > 2 ** lxs:
                    self.xs = 2 ** lxs
                if self.ys > 2 ** lys:
                    self.ys = 2 ** lys

            # crop to center
            self.sourcepixels = self.sourcepixels[offy : offy + self.ys, offx : offx + self.xs]
            print("sshape:" + repr(self.sourcepixels.shape))

            # if target image exists, change the size to fit
            if self.target_image in bpy.data.images:
                bpy.data.images[self.target_image].scale(self.xs, self.ys)
                self.image = bpy.data.images[self.target_image]
            else:
                self.image = bpy.data.images.new(self.target_image, width=self.xs, height=self.ys)

            self.pixels = numpy.zeros((self.ys, self.xs, 4))
            self.pixels[:, :, 3] = 1.0  # alpha is always 1.0 everywhere

            print("Start iteration")

        # noinspection PyCallByClass
        def finish_images(self, context):
            print("Assign data")
            # assign pixels
            self.image.pixels = self.pixels.flatten()
            bpy.ops.image.invert(invert_r=False, invert_g=False, invert_b=False, invert_a=False)

    class ConvolutionsOperator(GeneralImageOperator):
        """Image filter operator"""

        bl_idname = "uv.image_convolutions"
        bl_label = "Convolution filters"

        def calculate(self, context):
            return self.selected_filter(
                context.scene.seamless_filter_size, context.scene.seamless_filter_intensity
            )

        def execute(self, context):
            self.init_images(context)
            imgops = ImageOperations(self.sourcepixels)
            self.selected_filter = {
                "BLUR": imgops.blur,
                "EDGEDETECT": imgops.edgedetect,
                "SHARPEN": imgops.sharpen,
                "GAUSSIAN": imgops.gaussian,
                "FASTGAUSSIAN": imgops.fast_gaussian,
                "SOBEL": imgops.sobel,
                "NORMALSSIMPLE": imgops.normals_simple,
                "SEPARATEVALUES": imgops.separate_values,
                "POISSONTILES": imgops.poisson_blending,
                "BILATERAL": imgops.bilateral,
                "GRAYSCALE": imgops.grayscale,
            }[context.scene.seamless_filter_type]

            self.pixels = self.calculate(context)
            self.finish_images(context)

            return {"FINISHED"}

    class MaterialTextureGenerator(bpy.types.Operator):
        bl_idname = "uv.material_texgen"
        bl_label = "Generate textures for a material"

        def execute(self, context):
            self.input_material = context.scene.seamless_input_material
            self.input_image = bpy.data.images[context.scene.seamless_input_image]

            self.xs = self.input_image.size[0]
            self.ys = self.input_image.size[1]

            print("Assign data")
            print(repr(self.input_material))
            mat = bpy.data.materials[self.input_material]
            textures = []
            for t in mat.texture_slots.values():
                if t:
                    textures.append(t)
                    print(t)
            print(textures)
            matn = self.input_material

            difftex = matn + "_t_d"
            normtex = matn + "_t_n"
            spectex = matn + "_t_s"

            diffimg = matn + "_d"
            normimg = matn + "_n"
            specimg = matn + "_s"

            bpy.data.textures.new(difftex, "IMAGE")
            bpy.data.textures.new(normtex, "IMAGE")
            bpy.data.textures.new(spectex, "IMAGE")

            # GENERATE DIFFUSE
            bpy.data.textures[difftex].image = bpy.data.images.new(
                diffimg, width=self.xs, height=self.ys
            )
            sourcepixels = numpy.array(self.input_image.pixels).reshape((self.ys, self.xs, 4))
            bpy.data.textures[difftex].image.pixels = sourcepixels.flatten()

            # GENERATE NORMALS
            bpy.data.textures[normtex].image = bpy.data.images.new(
                normimg, width=self.xs, height=self.ys
            )
            bpy.data.textures[normtex].use_normal_map = True

            # copy image data into much more performant numpy arrays
            pixels = ImageOperations(
                numpy.array(self.input_image.pixels).reshape((self.ys, self.xs, 4))
            ).normals_simple(1, 1.0)

            # assign pixels
            bpy.data.textures[normtex].image.pixels = pixels.flatten()

            # GENERATE SPEC
            bpy.data.textures[spectex].image = bpy.data.images.new(
                specimg, width=self.xs, height=self.ys
            )

            ssp = numpy.array(self.input_image.pixels).reshape((self.ys, self.xs, 4))
            pixels = numpy.ones((self.ys, self.xs, 4))
            r, g, b = ssp[:, :, 0], ssp[:, :, 1], ssp[:, :, 2]
            gray = 0.2989 * r + 0.5870 * g + 0.1140 * b
            gray **= 4
            gray -= numpy.min(gray)
            gray /= numpy.max(gray)
            pixels[..., 0] = gray
            pixels[..., 1] = gray
            pixels[..., 2] = gray
            bpy.data.textures[spectex].image.pixels = pixels.flatten()

            bpy.ops.image.invert(invert_r=False, invert_g=False, invert_b=False, invert_a=False)

            bpy.data.materials[matn].specular_hardness = 30
            bpy.data.materials[matn].specular_intensity = 0
            for i in range(3):
                bpy.data.materials[matn].texture_slots.create(i)
            bpy.data.materials[matn].texture_slots[0].texture = bpy.data.textures[difftex]
            bpy.data.materials[matn].texture_slots[1].texture = bpy.data.textures[normtex]
            bpy.data.materials[matn].texture_slots[1].use_map_color_diffuse = False
            bpy.data.materials[matn].texture_slots[1].use_map_normal = True
            bpy.data.materials[matn].texture_slots[1].normal_factor = 0.5
            bpy.data.materials[matn].texture_slots[2].texture = bpy.data.textures[spectex]
            bpy.data.materials[matn].texture_slots[2].use_map_color_diffuse = False
            bpy.data.materials[matn].texture_slots[2].texture.use_alpha = False
            bpy.data.materials[matn].texture_slots[2].use_map_specular = True

            return {"FINISHED"}

    class SeamlessOperator(GeneralImageOperator):
        """Image seamless texture patcher operator"""

        bl_idname = "uv.seamless_operator"
        bl_label = "Image Seamless Operator"

        maxSSD = 100000000

        def ssd(self, b1, b2):
            if b1.shape == b2.shape:
                return numpy.sum(((b1 - b2) * [0.2989, 0.5870, 0.1140, 0.0]) ** 2)
            else:
                return self.maxSSD

        def stitch(self, x, y):
            dimage = self.pixels
            simage = self.sourcepixels

            winx = self.seamless_window
            winy = self.seamless_window

            if winx + x > self.xs or winy + y > self.ys:
                return

            sxs = self.xs - winx
            sys = self.ys - winy

            bestx, besty = self.seamless_window, self.seamless_window
            bestresult = self.maxSSD
            b1 = dimage[y : y + winy, x : x + winx, :]

            # only test for pixels where the alpha is not zero
            # alpha 0.0 signifies no data, the area to be filled by patching
            # alpha 1.0 is the source data plus the already written patches
            indices = numpy.where(b1[:, :, 3] > 0)

            # random sample through the entire source data to find the best match
            for i in range(self.seamless_samples):
                temx = numpy.random.randint(sxs)
                temy = numpy.random.randint(sys)
                b2 = simage[temy : temy + winy, temx : temx + winx, :]
                result = self.ssd(b1[indices], b2[indices])
                if result < bestresult:
                    bestresult = result
                    bestx, besty = temx, temy

            batch = numpy.copy(simage[besty : besty + winy, bestx : bestx + winx, :])

            if (
                self.seamless_smoothing
                and winx == self.seamless_window
                and winy == self.seamless_window
            ):
                # image edge patches can't have a mask because they are irregular shapes
                mask = numpy.ones((winx, winy))
                for i in range(int(self.seamless_overlap / 2)):
                    val = float(i) / float(self.seamless_overlap / 2)
                    mask[i, :] *= val
                    mask[winy - i - 1, :] *= val
                    mask[:, i] *= val
                    mask[:, winx - i - 1] *= val

                mask[numpy.where(b1[:, :, 3] < 0.5)] = 1.0
                # mask[indices] = 1.0
                batch[:, :, 0] = (
                    dimage[y : y + winy, x : x + winx, 0] * (1.0 - mask) + batch[:, :, 0] * mask
                )
                batch[:, :, 1] = (
                    dimage[y : y + winy, x : x + winx, 1] * (1.0 - mask) + batch[:, :, 1] * mask
                )
                batch[:, :, 2] = (
                    dimage[y : y + winy, x : x + winx, 2] * (1.0 - mask) + batch[:, :, 2] * mask
                )
                batch[:, :, 3] = 1.0

            # copy the new patch to the image
            dimage[y : y + winy, x : x + winx, :] = batch

        def patch_iterate(self):
            # offset both x and y half the size of the image (put edges on center)
            self.pixels = numpy.roll(
                self.sourcepixels, self.xs * 2 + self.xs * 4 * int(self.ys / 2)
            )

            step = self.seamless_window - self.seamless_overlap
            margin = step * self.seamless_lines - self.seamless_overlap

            # erase the data from the are we are about to fill with patches
            self.pixels[int(self.ys // 2 - margin // 2) : int(self.ys // 2 + margin // 2), :, :] = [
                0.0,
                0.0,
                0.0,
                0.0,
            ]
            self.pixels[:, int(self.xs // 2) - margin // 2 : int(self.xs // 2) + margin // 2, :] = [
                0.0,
                0.0,
                0.0,
                0.0,
            ]

            xmax = int(self.xs - 1)
            ymax = int(self.ys - 1)

            # reconstruct the missing area with patching
            xstart = int(self.xs / 2) - margin // 2 - self.seamless_overlap
            ystart = int(self.ys / 2) - margin // 2 - self.seamless_overlap

            for _ in range(1):
                # horizontal
                for x in range(0, xmax, step):
                    for y in range(0, self.seamless_lines):
                        self.stitch(x, y * step + ystart)

                # vertical
                for y in range(0, ymax, step):
                    for x in range(0, self.seamless_lines):
                        self.stitch(x * step + xstart, y)

                        # fill in the last edge cases
            self.pixels = numpy.roll(self.pixels, self.xs * 2)  # half x offset
            for y in range(0, self.seamless_lines):
                self.stitch(int(self.xs / 2) - step, y * step + ystart)
            self.pixels = numpy.roll(self.pixels, self.xs * 2)  # half x offset
            self.pixels = numpy.roll(self.pixels, self.xs * 4 * int(self.ys / 2))  # half y offset
            for x in range(0, self.seamless_lines):
                self.stitch(x * step + xstart, int(self.ys / 2) - step)
            self.pixels = numpy.roll(self.pixels, self.xs * 4 * int(self.ys / 2))  # half y offset

        def execute(self, context):
            self.init_images(context)

            self.seamless_samples = bpy.context.scene.seamless_samples
            self.seamless_window = bpy.context.scene.seamless_window
            self.seamless_overlap = bpy.context.scene.seamless_overlap
            self.seamless_lines = bpy.context.scene.seamless_lines
            self.seamless_smoothing = bpy.context.scene.seamless_smoothing

            self.patch_iterate()
            self.finish_images(context)

            return {"FINISHED"}

        class TextureToolsPanel(bpy.types.Panel):
            bl_space_type = "IMAGE_EDITOR"
            bl_region_type = "UI"
            bl_label = "Seamless Patching"
            bl_category = "Image"

            def draw(self, context):
                layout = self.layout

                row = layout.row()
                row.label(text="Patched:")

                row = layout.row()
                row.prop(context.scene, "seamless_samples")
                row.prop(context.scene, "seamless_window")

                row = layout.row()
                row.prop(context.scene, "seamless_overlap")
                row.prop(context.scene, "seamless_lines")

                row = layout.row()
                row.prop(context.scene, "seamless_smoothing")

                row = layout.row()
                row.operator(SeamlessOperator.bl_idname, text="Make seamless (patches)")

                row = layout.row()
                row.label(text="Fast and simple:")

                # row = layout.row()
                # row.prop(context.scene, "seamless_gimpmargin")

                row = layout.row()
                row.operator(GimpSeamlessOperator.bl_idname, text="Make seamless (fast)")

        class TextureToolsFiltersPanel(bpy.types.Panel):
            bl_space_type = "IMAGE_EDITOR"
            bl_region_type = "UI"
            bl_label = "Image Filters"
            bl_category = "Image"

            def draw(self, context):
                layout = self.layout
                row = layout.row()
                row.prop(context.scene, "seamless_filter_type")

                row = layout.row()
                row.prop(context.scene, "seamless_filter_size")
                row.prop(context.scene, "seamless_filter_intensity")

                row = layout.row()
                row.operator(ConvolutionsOperator.bl_idname, text="Filter")

        class TextureToolsMaterialsPanel(bpy.types.Panel):
            bl_space_type = "IMAGE_EDITOR"
            bl_region_type = "UI"
            bl_label = "Material Tools"
            bl_category = "Image"

            def draw(self, context):
                layout = self.layout
                # row = layout.row()
                # row.label("In a material world.")

                row = layout.row()
                row.prop(context.scene, "seamless_input_material")

                row = layout.row()
                row.operator(MaterialTextureGenerator.bl_idname, text="Generate textures")

        class TextureToolsImageSelectionPanel(bpy.types.Panel):
            bl_space_type = "IMAGE_EDITOR"
            bl_region_type = "UI"
            bl_label = "Image Selection"
            bl_category = "Image"

            def draw(self, context):
                layout = self.layout

                row = layout.row()
                row.prop(context.scene, "seamless_input_image")

                row = layout.row()
                row.prop(context.scene, "seamless_generated_name")

                row = layout.row()
                row.prop(context.scene, "seamless_powersoftwo")

        # ### INITIALIZATION

        # register classes
        regclasses = [
            SeamlessOperator,
            GimpSeamlessOperator,
            ConvolutionsOperator,
            TextureToolsImageSelectionPanel,
            TextureToolsPanel,
            TextureToolsFiltersPanel,
            TextureToolsMaterialsPanel,
            MaterialTextureGenerator,
        ]

        def register():
            # SEAMLESS PANEL
            bpy.types.Scene.seamless_samples = bpy.props.IntProperty(
                name="Samples", default=100, min=1, max=10000
            )
            bpy.types.Scene.seamless_window = bpy.props.IntProperty(
                name="Window", default=32, min=2, max=128
            )
            bpy.types.Scene.seamless_overlap = bpy.props.IntProperty(
                name="Overlap", default=8, min=1, max=64
            )
            bpy.types.Scene.seamless_lines = bpy.props.IntProperty(
                name="Lines", default=2, min=1, max=16
            )
            bpy.types.Scene.seamless_gimpmargin = bpy.props.IntProperty(
                name="Blending margin", default=200, min=1, max=1000
            )
            bpy.types.Scene.seamless_smoothing = bpy.props.BoolProperty(name="Patch smoothing")

            available_objects = []

            def availableObjects(self, context):
                available_objects.clear()
                for im in bpy.data.images:
                    name = im.name
                    available_objects.append((name, name, name))
                return available_objects

            # IMAGE SELECTION & TOOLS
            bpy.types.Scene.seamless_generated_name = bpy.props.StringProperty(
                name="Output image", default="generated"
            )
            bpy.types.Scene.seamless_input_image = bpy.props.EnumProperty(
                name="Input image", items=availableObjects
            )
            bpy.types.Scene.seamless_powersoftwo = bpy.props.BoolProperty(
                name="Crop to powers of two"
            )

            # FILTER PANEL
            bpy.types.Scene.seamless_filter_type = bpy.props.EnumProperty(
                name="Filter type",
                items=[
                    ("BLUR", "Box blur", "", 1),
                    ("SHARPEN", "Sharpen", "", 2),
                    ("EDGEDETECT", "Edge detect", "", 3),
                    ("EMBOSS", "Emboss", "", 4),
                    # ("GAUSSIAN", "Gaussian blur ro:5", "", 5),
                    ("FASTGAUSSIAN", "Fast gaussian", "", 6),
                    ("SOBEL", "Sobel", "", 7),
                    ("NORMALSSIMPLE", "Normal map: simple", "", 8),
                    ("SEPARATEVALUES", "Emphasize whites or blacks", "", 9),
                    # ("POISSONTILES", "Blend image edges", "", 10),
                    # ("BILATERAL", "Bilateral blur", "", 11),
                    ("GRAYSCALE", "Grayscale", "", 12),
                ],
            )
            bpy.types.Scene.seamless_filter_size = bpy.props.IntProperty(
                name="Size", default=1, min=1, max=9
            )
            bpy.types.Scene.seamless_filter_intensity = bpy.props.FloatProperty(
                name="Intensity", default=1.0, min=0.0, max=3.0
            )

            # MATERIALS PANEL
            available_materials = []

            def availableMaterials(self, context):
                available_materials.clear()
                for im in bpy.data.materials:
                    name = im.name
                    available_materials.append((name, name, name))
                return available_materials

            bpy.types.Scene.seamless_input_material = bpy.props.EnumProperty(
                name="Material", items=availableMaterials
            )

            for entry in regclasses:
                bpy.utils.register_class(entry)

        def unregister():
            for entry in regclasses:
                bpy.utils.unregister_class(entry)


class Grayscale_IOP(image_ops.ImageOperatorGenerator):
    def generate(self):
        self.prefix = "grayscale"
        self.info = "Grayscale from RGB"
        self.payload = lambda self, image, context: ImageOperations(image).grayscale().pixels


class Sharpen_IOP(image_ops.ImageOperatorGenerator):
    def generate(self):
        self.props["intensity"] = bpy.props.FloatProperty(name="Intensity", min=0.0, default=1.0)
        self.prefix = "sharpen"
        self.info = "Simple sharpen"
        self.payload = (
            lambda self, image, context: ImageOperations(image).sharpen(self.intensity).pixels
        )


class Fastblur_IOP(image_ops.ImageOperatorGenerator):
    def generate(self):
        self.props["steps"] = bpy.props.IntProperty(name="Steps", min=1, default=2)
        self.prefix = "fast_blur"
        self.info = "Fast blur"
        self.payload = (
            lambda self, image, context: ImageOperations(image).fast_blur(self.steps).pixels
        )


class GaussianBlur_IOP(image_ops.ImageOperatorGenerator):
    def generate(self):
        self.props["width"] = bpy.props.IntProperty(name="Width", min=1, default=2)
        self.props["intensity"] = bpy.props.FloatProperty(name="Intensity", min=0.0, default=1.0)
        self.prefix = "gaussian_blur"
        self.info = "Does a Gaussian blur (semi-broken atm)"
        self.payload = (
            lambda self, image, context: ImageOperations(image)
            .gaussian(self.width, self.intensity)
            .pixels
        )


class Normals_IOP(image_ops.ImageOperatorGenerator):
    def generate(self):
        self.props["intensity"] = bpy.props.FloatProperty(name="Intensity", min=0.0, default=1.0)
        self.prefix = "normals"
        self.info = "(Very rough estimate) normal map from RGB"
        self.payload = (
            lambda self, image, context: ImageOperations(image)
            .normals_simple(self.intensity)
            .pixels
        )


class MGLRender_IOP(image_ops.ImageOperatorGenerator):
    def generate(self):
        self.props["intensity"] = bpy.props.FloatProperty(name="Intensity", min=0.0, default=1.0)
        self.prefix = "test_mgl_render"
        self.info = "Test 0"
        self.category = "Debug"

        def _pl(self, image, context):
            import moderngl
            import numpy as np

            from PIL import Image

            ctx = moderngl.create_context()

            prog = ctx.program(
                vertex_shader="""
                    #version 330
            
                    in vec2 in_vert;
                    in vec3 in_color;
            
                    out vec3 v_color;
            
                    void main() {
                        v_color = in_color;
                        gl_Position = vec4(in_vert, 0.0, 1.0);
                    }
                """,
                fragment_shader="""
                    #version 330
            
                    in vec3 v_color;
            
                    out vec3 f_color;
            
                    void main() {
                        f_color = v_color;
                    }
                """,
            )

            x = np.linspace(-1.0, 1.0, 50)
            y = np.random.rand(50) - 0.5
            r = np.ones(50)
            g = np.zeros(50)
            b = np.zeros(50)

            vertices = np.dstack([x, y, r, g, b])

            vbo = ctx.buffer(vertices.astype("f4").tobytes())
            vao = ctx.simple_vertex_array(prog, vbo, "in_vert", "in_color")

            fbo = ctx.simple_framebuffer((image.shape[0], image.shape[1]))
            fbo.use()
            fbo.clear(0.0, 0.0, 0.0, 1.0)
            vao.render(moderngl.LINE_STRIP)

            isize, idata = fbo.size, fbo.read()

            res = numpy.array(Image.frombytes("RGB", isize, idata, "raw", "RGB", 0, -1))

            image[:, :, 0] = res[:, :, 0]
            image[:, :, 1] = res[:, :, 1]
            image[:, :, 2] = res[:, :, 2]

            return image

        self.payload = _pl


class GimpSeamless_IOP(image_ops.ImageOperatorGenerator):
    # TODO: the smoothing is not complete, it goes only one way
    """Image seamless generator operator"""

    def generate(self):
        self.prefix = "gimp_seamless"
        self.info = "Gimp style seamless image operation"

        def gimpify(self, image, context):
            pixels = numpy.copy(image)
            xs, ys = image.shape[0], image.shape[1]
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
            imask[-sys:ys, 0:sys] = numpy.flipud(imask[0:sys, 0:sxs])
            imask[-sys:ys, sxs : xs - 1] = numpy.flipud(imask[0:sys, sxs : xs - 1])
            imask[sys, :] = imask[sys - 1, :]  # center line

            # apply mask
            amask = numpy.zeros(pixels.shape, dtype=float)
            amask[:, :, 0] = imask
            amask[:, :, 1] = imask
            amask[:, :, 2] = imask
            amask[:, :, 3] = 1.0

            return amask * image + (numpy.ones(amask.shape) - amask) * pixels

        self.payload = gimpify


register, unregister = image_ops.create(locals())
