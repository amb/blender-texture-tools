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
# Created Date: Sunday, August 4th 2019, 8:42:29 pm
# Copyright: Tommi Hyppänen


import bpy  # noqa:F401

import numpy as np

CUDA_ACTIVE = False
# try:
#     import cupy as cup

#     CUDA_ACTIVE = True
# except Exception:
#     CUDA_ACTIVE = False
#     cup = np

from collections import OrderedDict

from .bpy_amb import master_ops
from .bpy_amb import utils
import importlib

importlib.reload(master_ops)
importlib.reload(utils)


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


def get_area_image(context):
    teximage = None
    teximage = context.area.spaces.active.image

    if teximage is not None and teximage.size[1] != 0:
        return teximage
    else:
        return None


def create(lc, additional_classes):
    """ create(locals()) """
    load_these = []
    for name, obj in lc.copy().items():
        if hasattr(obj, "__bases__") and obj.__bases__[0].__name__ == "ImageOperatorGenerator":
            load_these.append(obj)

    # Define common properties and drawing functionality for the image operators panel
    _props = OrderedDict()
    _props["source_enum"] = bpy.props.EnumProperty(
        name="FromEnum", items=[("active", "Active", "", 1), ("defined", "Specified", "", 2)]
    )
    _props["target_enum"] = bpy.props.EnumProperty(
        name="ToEnum", items=[("active", "Active", "", 1), ("defined", "Specified", "", 2)]
    )
    _props["source"] = bpy.props.PointerProperty(name="Image", type=bpy.types.Image)
    _props["target"] = bpy.props.PointerProperty(name="Image", type=bpy.types.Image)

    def _panel_draw(self, context):
        layout = self.layout
        col = layout.column(align=True)
        box = col.box()
        row = box.row()
        row.label(text="Source:")
        row.prop(context.scene.texture_tools, "global" + "_source_enum", expand=True)
        if context.scene.texture_tools.global_source_enum == "defined":
            row = box.row()
            row.prop(context.scene.texture_tools, "global" + "_source")

        box = col.box()
        row = box.row()
        row.label(text="Target:")
        row.prop(context.scene.texture_tools, "global" + "_target_enum", expand=True)
        if context.scene.texture_tools.global_target_enum == "defined":
            row = box.row()
            row.prop(context.scene.texture_tools, "global" + "_target")

    # Build the rest of the panel
    pbuild = master_ops.PanelBuilder(
        "texture_tools",
        load_these,
        _panel_draw,
        _props,
        "OBUILD",
        "IMAGE_EDITOR",
        "UI",
        "Image",
        additional_classes,
    )

    return pbuild.register_params, pbuild.unregister_params


class ImageOperator(master_ops.MacroOperator):
    def payload(self, image, context):
        pass

    def execute(self, context):
        image = get_area_image(bpy.context)

        ctt = context.scene.texture_tools

        if ctt.global_source_enum == "defined":
            source_image = ctt.global_source
        else:
            source_image = image

        if ctt.global_target_enum == "defined":
            target_image = ctt.global_target
        else:
            target_image = image

        input_pixels = np.empty(len(image.pixels), dtype=np.float32)

        image.pixels.foreach_get(input_pixels)
        # print(input_pixels.dtype, type(input_pixels))
        input_pixels = input_pixels.reshape(source_image.size[1], source_image.size[0], 4)

        # with utils.Profile_this(lines=10):
        result = self.payload(input_pixels, context)

        if (
            target_image.size[1] != result.shape[0]
            or target_image.size[0] != result.shape[1]
        ):
            print("Scaling image")
            target_image.scale(result.shape[1], result.shape[0])

        # print(result.dtype, type(result))
        # print(np.max(result), np.min(result))
        target_image.pixels.foreach_set(np.float32(result.ravel()))
        return {"FINISHED"}


class ImageOperatorGenerator(master_ops.OperatorGenerator):
    def __init__(self, master_name):
        self.init_begin(master_name)
        self.force_numpy = False
        self.generate()
        self.init_end()
        self.name = "IMAGE_OT_" + self.name
        self.create_op(ImageOperator, "image")
        self.op.force_numpy = self.force_numpy
