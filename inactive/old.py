class MGLRender_IOP(image_ops.ImageOperatorGenerator):
    def generate(self):
        self.prefix = "test_mgl_render"
        self.info = "Test 0"
        self.category = "Debug"

        def _pl(self, image, context):
            import moderngl
            import numpy as np

            ctx = moderngl.create_context()
            prog = ctx.program(
                vertex_shader="""
                    #version 330

                    in vec2 in_vert;
                    out vec2 vert_pos;

                    void main() {
                        vert_pos = 0.5 * (in_vert + 1.0);
                        gl_Position = vec4(in_vert, 0.0, 1.0);
                    }
                """,
                fragment_shader="""
                    #version 330

                    in vec2 vert_pos;
                    out vec3 f_color;

                    void main() {
                        // moire patterns
                        float a = length(vert_pos-vec2(0.53, 0.53));
                        float b = length(vert_pos-vec2(0.47, 0.47));
                        float as = (sin(a*9999*exp(-a))+1.0)/2.0;
                        float bs = (sin(b*9999*exp(-b))+1.0)/2.0;
                        float c = as * bs;
                        f_color = vec3(c, c, c);
                    }
                """,
            )

            vertices = np.array([1.0, 1.0, -1.0, 1.0, -1.0, -1.0, -1.0, -1.0, 1.0, -1.0, 1.0, 1.0])

            vbo = ctx.buffer(vertices.astype("f4").tobytes())
            vao = ctx.simple_vertex_array(prog, vbo, "in_vert")

            fbo = ctx.simple_framebuffer((image.shape[0], image.shape[1]))
            fbo.use()
            fbo.clear(0.0, 0.0, 0.0, 1.0)
            vao.render(moderngl.TRIANGLE_STRIP)

            res = numpy.frombuffer(fbo.read(), dtype=np.uint8).reshape((*fbo.size, 3)) / 255.0
            image[:, :, 0] = res[:, :, 0]
            image[:, :, 1] = res[:, :, 1]
            image[:, :, 2] = res[:, :, 2]

            return image

        self.payload = _pl


class MGLRender2_IOP(image_ops.ImageOperatorGenerator):
    def generate(self):
        self.prefix = "test_mgl_render2"
        self.info = "Test 1"
        self.category = "Debug"

        def _pl(self, image, context):
            import moderngl
            import numpy as np

            ctx = moderngl.create_context()
            prog = ctx.program(
                vertex_shader="""
                    #version 330

                    in vec2 in_vert;
                    out vec2 vert_pos;

                    void main() {
                        vert_pos = 0.5 * (in_vert + 1.0);
                        gl_Position = vec4(in_vert, 0.0, 1.0);
                    }
                """,
                fragment_shader="""
                    #version 330

                    in vec2 vert_pos;
                    out vec4 f_color;
                    uniform sampler2D Texture;

                    float normpdf(in float x, in float sigma)
                    {
                        return 0.39894*exp(-0.5*x*x/(sigma*sigma))/sigma;
                    }

                    const vec2 iResolution = vec2(512.0, 512.0);

                    void main()
                    {
                        vec4 c = texture(Texture, vert_pos.xy);

                        const int mSize = 11;
                        const int kSize = (mSize-1)/2;
                        float kernel[mSize];
                        vec3 final_colour = vec3(0.0);

                        // create the 1-D kernel
                        float sigma = 7.0;
                        float Z = 0.0;
                        for (int j = 0; j <= kSize; ++j)
                        {
                            kernel[kSize+j] = kernel[kSize-j] = normpdf(float(j), sigma);
                        }

                        // get the normalization factor (as the gaussian has been clamped)
                        for (int j = 0; j < mSize; ++j)
                        {
                            Z += kernel[j];
                        }

                        // read out the texels
                        for (int i=-kSize; i <= kSize; ++i)
                        {
                            for (int j=-kSize; j <= kSize; ++j)
                            {
                                final_colour += kernel[kSize+j]*kernel[kSize+i] *
                                    texture(Texture, vert_pos.xy +
                                        (vec2(float(i),float(j)))
                                        / iResolution.xy).rgb;
                            }
                        }

                        f_color = vec4(final_colour/(Z*Z), 1.0);
                    }
                """,
            )

            vertices = np.array([1.0, 1.0, -1.0, 1.0, -1.0, -1.0, -1.0, -1.0, 1.0, -1.0, 1.0, 1.0])

            vbo = ctx.buffer(vertices.astype("f4").tobytes())
            vao = ctx.simple_vertex_array(prog, vbo, "in_vert")

            fbo = ctx.simple_framebuffer((*image.shape[:2],), components=4)
            fbo.use()
            fbo.clear(0.0, 0.0, 0.0, 1.0)

            tex = ctx.texture((*image.shape[:2],), 4, (image * 255.0).astype(np.uint8).tobytes())
            tex.use()

            vao.render(moderngl.TRIANGLE_STRIP)

            res = numpy.frombuffer(fbo.read(components=4), dtype=np.uint8).reshape((*fbo.size, 4))
            res = res / 255.0

            image[:, :, 0] = res[:, :, 0]
            image[:, :, 1] = res[:, :, 1]
            image[:, :, 2] = res[:, :, 2]

            return image

        self.payload = _pl


class MGLRender3_IOP(image_ops.ImageOperatorGenerator):
    def generate(self):
        self.prefix = "test_mgl_render3"
        self.info = "Test 2"
        self.category = "Debug"

        def _pl(self, image, context):
            import moderngl
            import numpy as np

            # default: false, false, 3000, 1.00, 0.3, 0.05, 0.06, 0.9
            ctx = moderngl.create_context()

            prog = ctx.program(
                vertex_shader="""
                    #version 330

                    in vec2 in_vert;
                    out vec2 vert_pos;

                    void main() {
                        vert_pos = 0.5 * (in_vert + 1.0);
                        gl_Position = vec4(in_vert, 0.0, 1.0);
                    }
                """,
                fragment_shader="""
                    #version 330

                    in vec2 vert_pos;
                    out vec4 out_vert;

                    uniform sampler2D Texture;

                    uniform float dA = 1.0;
                    uniform float dB = 0.3;
                    uniform float feed = 0.05;
                    uniform float kill = 0.06;
                    uniform float time = 0.9;

                    uniform int imgSize = 256;

                    vec4 tex(int x, int y) {
                        return texelFetch(Texture, ivec2(x % imgSize, y % imgSize), 0);
                    }

                    void main()
                    {
                        ivec2 in_text = ivec2(vert_pos * imgSize);
                        int x = in_text.x;
                        int y = in_text.y;
                        vec4 sc = tex(x, y);

                        vec4 lap = vec4(0.0);
                        lap += tex(x, y-1);
                        lap += tex(x, y+1);
                        lap += tex(x-1, y);
                        lap += tex(x+1, y);
                        lap /= 4.0;

                        // A2 = A + (self.dA * lp(A) - ab2 + (1.0 - A) * self.feed) * t
                        // B2 = B + (self.dB * lp(B) + ab2 - B * kf) * t

                        float a = sc.r;
                        float b = sc.g;
                        float ab2 = a * pow(b, 2.0);
                        out_vert.r = a + (dA * (lap.r - sc.r) - ab2 + (1.0 - a) * feed) * time;
                        out_vert.g = b + (dB * (lap.g - sc.g) + ab2 - b * (kill + feed)) * time;
                    }
                """,
            )

            res = np.ones(shape=image.shape, dtype=np.float32)

            A = np.ones(shape=(*image.shape[:2],), dtype=np.float32)
            B = np.zeros(shape=(*image.shape[:2],), dtype=np.float32)
            w, h = image.shape[0] // 2, image.shape[1] // 2
            B[w - 5 : w + 5, h - 5 : h + 5] = 1.0

            img_in = np.empty((*image.shape[:2], 4))
            img_in[:, :, 0] = A
            img_in[:, :, 1] = B
            img_in[:, :, 2] = 0.0
            img_in[:, :, 3] = 0.0

            vertices = np.array([1.0, 1.0, -1.0, 1.0, -1.0, -1.0, -1.0, -1.0, 1.0, -1.0, 1.0, 1.0])
            vbo = ctx.buffer(vertices.astype("f4").tobytes())
            vao = ctx.simple_vertex_array(prog, vbo, "in_vert")

            prog.get("imgSize", -1).value = img_in.shape[0]
            prog.get("dA", -1).value = 1.0
            prog.get("dB", -1).value = 0.3
            prog.get("feed", -1).value = 0.05
            prog.get("kill", -1).value = 0.06
            prog.get("time", -1).value = 0.9

            pixels = (img_in * 255.0).astype(np.uint8)
            tex = ctx.texture((*img_in.shape[:2],), 4, pixels.tobytes())
            tex.use()

            fbo = ctx.simple_framebuffer((*image.shape[:2],), components=4)
            fbo.use()
            fbo.clear(0.0, 0.0, 0.0, 1.0)

            for _ in range(5000):
                vao.render(moderngl.TRIANGLE_STRIP)
                ctx.copy_framebuffer(tex, fbo)

            res = numpy.frombuffer(fbo.read(), dtype=np.uint8).reshape((*image.shape[:2], 3))
            res = res / 255.0
            total = res[:, :, 1] - res[:, :, 0]
            total -= np.min(total)
            total /= np.max(total)
            image[:, :, 0] = total
            image[:, :, 1] = total
            image[:, :, 2] = total

            return image

        self.payload = _pl


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
        self.pixels = numpy.roll(self.sourcepixels, self.xs * 2 + self.xs * 4 * int(self.ys / 2))

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
        bpy.types.Scene.seamless_powersoftwo = bpy.props.BoolProperty(name="Crop to powers of two")

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

    ray_tracer = """
    float sphere(vec3 ray, vec3 dir, vec3 center, float radius)
    {
        vec3 rc = ray-center;
        float c = dot(rc, rc) - (radius*radius);
        float b = dot(dir, rc);
        float d = b*b - c;
        float t = -b - sqrt(abs(d));
        float st = step(0.0, min(t,d));
        return mix(-1.0, t, st);
    }

    vec3 background(float t, vec3 rd)
    {
        vec3 light = normalize(vec3(sin(t), 0.6, cos(t)));
        float sun = max(0.0, dot(rd, light));
        float sky = max(0.0, dot(rd, vec3(0.0, 1.0, 0.0)));
        float ground = max(0.0, -dot(rd, vec3(0.0, 1.0, 0.0)));
        return \
        (pow(sun, 256.0)+0.2*pow(sun, 2.0))*vec3(2.0, 1.6, 1.0) +
        pow(ground, 0.5)*vec3(0.4, 0.3, 0.2) +
        pow(sky, 1.0)*vec3(0.5, 0.6, 0.7);
    }

    void main(void)
    {
        vec2 uv = (-1.0 + 2.0*gl_FragCoord.xy / iResolution.xy) *
        vec2(iResolution.x/iResolution.y, 1.0);
        vec3 ro = vec3(0.0, 0.0, -3.0);
        vec3 rd = normalize(vec3(uv, 1.0));
        vec3 p = vec3(0.0, 0.0, 0.0);
        float t = sphere(ro, rd, p, 1.0);
        vec3 nml = normalize(p - (ro+rd*t));
        vec3 bgCol = background(iGlobalTime, rd);
        rd = reflect(rd, nml);
        vec3 col = background(iGlobalTime, rd) * vec3(0.9, 0.8, 1.0);
        gl_FragColor = vec4( mix(bgCol, col, step(0.0, t)), 1.0 );
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


class ReactionDiffusion_IOP(image_ops.ImageOperatorGenerator):
    def generate(self):
        self.props["atype"] = bpy.props.EnumProperty(
            name="A Type",
            items=[
                ("RANDOM", "Random", "", 1),
                ("ZERO", "Zero", "", 2),
                ("ONES", "Ones", "", 4),
                ("RANGE", "Range", "", 3),
            ],
        )
        self.props["btype"] = bpy.props.EnumProperty(
            name="B Type",
            items=[
                ("RANDOM", "Random", "", 1),
                ("ZERO", "Zero", "", 2),
                ("ONES", "Ones", "", 4),
                ("MIDDLE", "Middle", "", 3),
            ],
        )

        self.props["output"] = bpy.props.EnumProperty(
            name="Output",
            items=[
                ("B_A", "B-A", "", 1),
                ("A_B", "A-B", "", 2),
                ("A", "A", "", 3),
                ("B", "B", "", 4),
                ("B*A", "B*A", "", 5),
                ("B+A", "B+A", "", 6),
            ],
        )

        self.props["iter"] = bpy.props.IntProperty(name="Iterations", min=1, default=1000)
        self.props["dA"] = bpy.props.FloatProperty(name="dA", min=0.0, default=1.0)
        self.props["dB"] = bpy.props.FloatProperty(name="dB", min=0.0, default=0.5)
        self.props["feed"] = bpy.props.FloatProperty(name="Feed rate (x100)", min=0.0, default=5.5)
        self.props["kill"] = bpy.props.FloatProperty(name="Kill rate (x100)", min=0.0, default=6.2)
        self.props["time"] = bpy.props.FloatProperty(
            name="Timestep", min=0.001, max=1.0, default=0.9
        )
        self.props["scale"] = bpy.props.FloatProperty(
            name="Scale", min=0.01, default=1.0, max=100.0
        )

        # lizard scales: true, true, 5000, 1.0, 0.55, 0.082, 0.06, 0.9
        # default: false, false, 3000, 1.00, 0.3, 0.05, 0.06, 0.9
        # stones: ones, random, b*a, 7556, 1.00, 0.34, 0.07, 0.06, 0.84

        self.prefix = "reaction_diffusion"
        self.info = "Reaction diffusion"
        self.category = "Generator"

        def _pl(self, image, context):
            def laplacian(p):
                ring = -p
                ring[:-1, :] += p[1:, :] * 0.25
                ring[-1, :] += p[0, :] * 0.25
                ring[1:, :] += p[:-1, :] * 0.25
                ring[0, :] += p[-1, :] * 0.25

                ring[:, :-1] += p[:, 1:] * 0.25
                ring[:, -1] += p[:, 0] * 0.25
                ring[:, 1:] += p[:, :-1] * 0.25
                ring[:, 0] += p[:, -1] * 0.25
                return ring

            def laplacian3(p):
                # TODO: broken
                # is Laplacian separatable?
                f = lambda m: np.convolve(m, [1, -2, 1], mode="same")
                p = np.apply_along_axis(f, axis=1, arr=p)
                p = np.apply_along_axis(f, axis=0, arr=p)
                return p

            res = np.ones(shape=image.shape, dtype=np.float32)

            # grid init with A=1, B=0, small area B=1
            if self.rnda:
                gridA = np.random.random(size=(*image.shape[:2],))
                ix = image.shape[0]
                for i in range(ix):
                    gridA[i, :] *= 0.5 + ((i - ix / 2) / (ix * 2.0))
            else:
                gridA = np.ones(shape=(*image.shape[:2],), dtype=np.float32)

            if self.rndb:
                gridB = np.random.random(size=(*image.shape[:2],))
            else:
                gridB = np.zeros(shape=(*image.shape[:2],), dtype=np.float32)
                w, h = image.shape[0] // 2, image.shape[1] // 2
                gridB[w - 5 : w + 5, h - 5 : h + 5] = 1.0

            lp = laplacian

            A = gridA
            B = gridB
            A2 = None
            B2 = None

            print("v2")

            t = self.time
            kf = self.kill + self.feed
            for _ in range(self.iter):
                ab2 = A * B ** 2
                A2 = A + (self.dA * lp(A) - ab2 + (1.0 - A) * self.feed) * t
                B2 = B + (self.dB * lp(B) + ab2 - B * kf) * t
                A = A2
                B = B2

            v = B - A
            v -= np.min(v)
            v /= np.max(v)
            res[:, :, 0] = v
            res[:, :, 1] = v
            res[:, :, 2] = v

            return res

        def _pl2(self, image, context):
            import moderngl
            import numpy as np

            # TODO: floats instead of uint8

            ctx = moderngl.create_context()

            prog = ctx.program(
                vertex_shader="""
                    #version 330

                    in vec2 in_vert;
                    out vec2 vert_pos;

                    void main() {
                        vert_pos = 0.5 * (in_vert + 1.0);
                        gl_Position = vec4(in_vert, 0.0, 1.0);
                    }
                """,
                fragment_shader="""
                    #version 330

                    in vec2 vert_pos;
                    out vec4 out_vert;

                    uniform sampler2D Texture;

                    uniform float dA = 1.0;
                    uniform float dB = 0.3;
                    uniform float feed = 0.05;
                    uniform float kill = 0.06;
                    uniform float time = 0.9;
                    uniform float scale = 1.0;
                    uniform int imgSize = 256;

                    vec4 tex(float x, float y) {
                        return texture(Texture, vec2(mod(x, 1.0), mod(y, 1.0)));
                    }

                    void main()
                    {
                        float fstep = scale/float(imgSize);
                        float x = vert_pos.x;
                        float y = vert_pos.y;
                        vec4 sc = tex(x, y);

                        vec4 lap = vec4(0.0);
                        lap += tex(x, y-fstep);
                        lap += tex(x, y+fstep);
                        lap += tex(x-fstep, y);
                        lap += tex(x+fstep, y);
                        lap += tex(x-fstep, y-fstep)*0.5;
                        lap += tex(x-fstep, y+fstep)*0.5;
                        lap += tex(x+fstep, y-fstep)*0.5;
                        lap += tex(x+fstep, y+fstep)*0.5;
                        lap /= 6.0;

                        // A2 = A + (self.dA * lp(A) - ab2 + (1.0 - A) * self.feed) * t
                        // B2 = B + (self.dB * lp(B) + ab2 - B * kf) * t

                        float a = sc.r;
                        float b = sc.g;
                        float ab2 = a * pow(b, 2.0);
                        out_vert.r = a + (dA * (lap.r - sc.r) - ab2 + (1.0 - a) * feed) * time;
                        out_vert.g = b + (dB * (lap.g - sc.g) + ab2 - b * (kill + feed)) * time;
                    }
                """,
            )

            res = np.ones(shape=image.shape, dtype=np.float32)
            w, h = image.shape[0] // 2, image.shape[1] // 2

            if self.atype == "RANDOM":
                A = np.random.random(size=(*image.shape[:2],))
            elif self.atype == "RANGE":
                A = np.random.random(size=(*image.shape[:2],))
                ix = image.shape[0]
                for i in range(ix):
                    A[i, :] *= 0.5 + ((i - ix / 2) / (ix * 2.0))
            elif self.atype == "ONES":
                A = np.ones(shape=(*image.shape[:2],), dtype=np.float32)
            else:
                A = np.zeros(shape=(*image.shape[:2],), dtype=np.float32)

            if self.btype == "RANDOM":
                B = np.random.random(size=(*image.shape[:2],))
            elif self.btype == "MIDDLE":
                B = np.zeros(shape=(*image.shape[:2],), dtype=np.float32)
                B[w - 5 : w + 5, h - 5 : h + 5] = 1.0
            elif self.btype == "ONES":
                B = np.ones(shape=(*image.shape[:2],), dtype=np.float32)
            elif self.btype == "ZERO":
                B = np.zeros(shape=(*image.shape[:2],), dtype=np.float32)

            img_in = np.empty((*image.shape[:2], 4))
            img_in[:, :, 0] = A
            img_in[:, :, 1] = B
            img_in[:, :, 2] = 0.0
            img_in[:, :, 3] = 0.0

            prog.get("imgSize", -1).value = img_in.shape[0]
            prog.get("dA", -1).value = self.dA
            prog.get("dB", -1).value = self.dB
            prog.get("feed", -1).value = self.feed / 100
            prog.get("kill", -1).value = self.kill / 100
            prog.get("time", -1).value = self.time
            prog.get("scale", -1).value = self.scale

            vertices = np.array([1.0, 1.0, -1.0, 1.0, -1.0, -1.0, -1.0, -1.0, 1.0, -1.0, 1.0, 1.0])
            vbo = ctx.buffer(vertices.astype("f4").tobytes())
            vao = ctx.simple_vertex_array(prog, vbo, "in_vert")

            precision = "f1"
            np_dtype = np.uint8

            pixels = (img_in * 255.0).astype(np_dtype)
            # pixels = img_in.astype(np_dtype)
            tex = ctx.texture((*img_in.shape[:2],), 4, pixels.tobytes(), dtype=precision)
            tex.use()

            fbo = ctx.simple_framebuffer((*image.shape[:2],), components=4, dtype=precision)
            fbo.use()
            fbo.clear(0.0, 0.0, 0.0, 1.0)

            for _ in range(self.iter):
                vao.render(moderngl.TRIANGLE_STRIP)
                ctx.copy_framebuffer(tex, fbo)
                # tex.write(fbo.read(components=4)) # slow

            res = numpy.frombuffer(fbo.read(dtype=precision), dtype=np_dtype).reshape(
                (*image.shape[:2], 3)
            )
            res = res / 255.0

            if self.output == "B_A":
                total = res[:, :, 1] - res[:, :, 0]
            if self.output == "A_B":
                total = res[:, :, 0] - res[:, :, 1]
            if self.output == "A":
                total = res[:, :, 0]
            if self.output == "B":
                total = res[:, :, 1]
            if self.output == "B*A":
                total = res[:, :, 1] * res[:, :, 0]
            if self.output == "B+A":
                total = res[:, :, 1] + res[:, :, 0]

            total -= np.min(total)
            npm = np.max(total)
            if npm > 0.0:
                total /= npm

            image[:, :, 0] = total
            image[:, :, 1] = total
            image[:, :, 2] = total

            return image

        self.payload = _pl2


class Base64_16x16_IOP(image_ops.ImageOperatorGenerator):
    def generate(self):
        self.prefix = "base64"
        self.info = "Base64"
        self.category = "Debug"

        def _pl(self, image, context):
            import zlib
            import base64

            print(image.shape)

            def img_compress(img):
                if img.shape[0] > 32 or img.shape[1] > 32 or img.shape[0] < 8 or img.shape[1] < 8:
                    print("Wrong image size.")
                    return None
                icon = []
                flatdim = img.shape[0] * img.shape[1]
                for v in image.reshape((flatdim, 4)):
                    icon.append(int(round(v[0] * 15.0)))
                    icon.append(int(round(v[1] * 15.0)))
                    icon.append(int(round(v[2] * 7.0)))
                    icon.append(int(round(v[3] * 3.0)))

                compressed = zlib.compress(bytes(icon), level=6)
                print(len(icon), "=>", len(compressed))
                encoded = base64.b64encode(compressed)
                return repr(img.shape[0]) + "," + repr(img.shape[1]) + "," + encoded.decode("utf-8")

            compressed = img_compress(image)
            print(len(compressed), image.shape[0] * image.shape[1] * image.shape[2])
            print(compressed)

            def img_decompress(inp):
                vals = inp.split(",")
                w, h = int(vals[0]), int(vals[1])
                decoded = base64.b64decode(vals[2])
                uncompressed = zlib.decompress(decoded)
                values = []
                for v in range(len(uncompressed))[::4]:
                    values.append(uncompressed[v + 0] / 15)
                    values.append(uncompressed[v + 1] / 15)
                    values.append(uncompressed[v + 2] / 7)
                    values.append(uncompressed[v + 3] / 3)

                return np.array(values).reshape((w, h, 4))

            image = img_decompress(compressed)

            # to image from icon text
            # to_value = {v: i for i, v in enumerate(chars)}
            # values = []
            # counter = 0
            # for v in icon_text:
            #     values.append(to_value[v] / 63)
            #     counter += 1
            #     if counter == 3:
            #         counter = 0
            #         values.append(1.0)

            # # image = image.reshape((image.shape[0] * image.shape[1], 4))
            # image = np.array(values, dtype=image.dtype).reshape(image.shape)

            return image

        self.payload = _pl


class RenderObject_IOP(image_ops.ImageOperatorGenerator):
    def generate(self):
        self.props["object"] = bpy.props.PointerProperty(name="Target", type=bpy.types.Object)

        self.prefix = "render_object"
        self.info = "Simple render of selected object"
        self.category = "Debug"

        def _pl(self, image, context):
            bm = bmesh.new()
            bm.from_mesh(self.object.data)
            bmesh.ops.triangulate(bm, faces=bm.faces[:])

            datatype = np.float32

            # rays
            rays = np.empty((image.shape[0], image.shape[1], 2, 3), dtype=datatype)
            w, h = image.shape[0], image.shape[1]
            for x in range(w):
                for y in range(h):
                    # ray origin
                    rays[x, y, 0, 0] = y * 2 / h - 1.0
                    rays[x, y, 0, 1] = -5.0
                    rays[x, y, 0, 2] = x * 2 / w - 1.0
                    # ray direction
                    rays[x, y, 1, 0] = 0.0
                    rays[x, y, 1, 1] = 1.0
                    rays[x, y, 1, 2] = 0.0

            # mesh
            tris = np.zeros((len(bm.faces), 3, 3), dtype=datatype)
            for fi, f in enumerate(bm.faces):
                vv = f.verts
                # tris[fi] = [i.co for i in vv]
                tris[fi][0][0] = vv[0].co[0]
                tris[fi][0][1] = vv[0].co[1]
                tris[fi][0][2] = vv[0].co[2]
                tris[fi][1][0] = vv[1].co[0]
                tris[fi][1][1] = vv[1].co[1]
                tris[fi][1][2] = vv[1].co[2]
                tris[fi][2][0] = vv[2].co[0]
                tris[fi][2][1] = vv[2].co[1]
                tris[fi][2][2] = vv[2].co[2]
                # v1v0 = vv[1].co - vv[0].co
                # v2v0 = vv[2].co - vv[0].co
                # assert v1v0.length > 0.0
                # assert v2v0.length > 0.0

            bm.faces.ensure_lookup_table()

            # sun_direction = np.array(mu.Vector([0.5, -0.5, 0.5]).normalized(), dtype=datatype)
            # normals = np.array(
            #     [np.array(i.normal, dtype=datatype) for i in bm.faces], dtype=datatype
            # )

            print(image.shape, rays.shape, tris.shape, rays.dtype)
            # result = np.zeros((image.shape[0], image.shape[1]), dtype=datatype)

            def rt_nb(do_a_jit=True):
                import numba

                def intersect_ray(ro, rda, vrt):
                    def cross(a, b):
                        return np.array(
                            [
                                a[1] * b[2] - a[2] * b[1],
                                a[2] * b[0] - a[0] * b[2],
                                a[0] * b[1] - a[1] * b[0],
                            ]
                        )

                    def dot(a, b):
                        return a[0] * b[0] + a[1] * b[1] + a[2] * b[2]

                    def tri_intersect(ro, rd, v0, v1, v2):
                        v1v0 = v1 - v0
                        v2v0 = v2 - v0
                        rov0 = ro - v0
                        n = cross(v1v0, v2v0)
                        q = cross(rov0, rd)
                        rdn = dot(rd, n)
                        if rdn == 0.0:
                            return -1.0
                            # return (-1.0, 0.0, 0.0)
                        d = 1.0 / rdn
                        u = d * (dot(-q, v2v0))
                        v = d * (dot(q, v1v0))
                        t = d * (dot(-n, rov0))
                        if u < 0.0 or u > 1.0 or v < 0.0 or u + v > 1.0:
                            t = -1.0
                        # return (t, u, v)
                        return t

                    c = 1.0e10
                    n = -1
                    for i in range(len(vrt) // 3):
                        iv = i * 3
                        rcast = tri_intersect(ro, rda, vrt[iv], vrt[iv + 1], vrt[iv + 2])
                        if rcast < c and rcast > 0.0:
                            c = rcast
                            n = i

                    return n

                if do_a_jit:
                    intersect_ray = numba.njit(parallel=False)(intersect_ray)

                result = np.empty((image.shape[0], image.shape[1]), dtype=np.float32)

                def rnd_res(ro, rd, verts, normals, sun_direction, res):
                    for x in range(res.shape[0]):
                        print(x)
                        for y in numba.prange(res.shape[1]):
                            r = intersect_ray(ro[x, y], rd, verts)
                            res[x, y] = np.dot(normals[r], sun_direction) if r >= 0 else 0.0

                rnd_res(ro, rd, verts, normals, sun_direction, result)

                return result

            # numba is aboug 20x speedup with single core CPU
            # result = rt_nb(do_a_jit=True)

            def rt_glcompute():
                # in: rays, tris
                # out: distance, u, v, face index

                from .bpy_amb import raycast
                import importlib

                importlib.reload(raycast)

                rc = raycast.Raycaster(tris)
                rw = rays.shape[0]
                res = rc.cast(rays.reshape((rw * rw, 2, 3)))

                return res.reshape((rw, rw, 4))

            result = rt_glcompute()
            dist = result[:, :, 0]
            dist = np.where(dist < 50.0, (dist - 4.0) / 2.0, 1.0)

            image[:, :, 0] = dist
            image[:, :, 1] = result[:, :, 1]
            image[:, :, 2] = result[:, :, 2]

            bm.free()
            return image

        self.payload = _pl
