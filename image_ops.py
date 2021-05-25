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

from collections import OrderedDict

from .oklab import linear_to_srgb, srgb_to_linear
from .bpy_amb import master_ops
from .bpy_amb import utils
import importlib

importlib.reload(master_ops)
importlib.reload(utils)


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

    # _props["linear"] = bpy.props.BoolProperty(
    #     name="Linear", default=False, description="Use linear sRGB color space for operations"
    # )

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

        # row = layout.row()
        # row.prop(context.scene.texture_tools, "global_linear")

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


def context_get_source_target_image(context):
    ctt = context.scene.texture_tools
    image = get_area_image(context)

    if ctt.global_source_enum == "defined":
        source_image = ctt.global_source
    else:
        source_image = image

    if ctt.global_target_enum == "defined":
        target_image = ctt.global_target
    else:
        target_image = image

    return source_image, target_image


def image_to_ndarray(source, linear_transform=False):
    assert len(source.pixels) > 0
    input_pixels = np.empty(len(source.pixels), dtype=np.float32)
    source.pixels.foreach_get(input_pixels)
    input_pixels = input_pixels.reshape(source.size[1], source.size[0], 4)
    if linear_transform:
        input_pixels = srgb_to_linear(input_pixels)

    return input_pixels


def ndarray_to_image(target, pixels, linear_transform=False):
    if linear_transform:
        pixels = linear_to_srgb(pixels, clamp=True)

    if target.size[1] != pixels.shape[0] or target.size[0] != pixels.shape[1]:
        target.scale(pixels.shape[1], pixels.shape[0])
    target.pixels.foreach_set(np.float32(pixels.ravel()))


def image_copy_overwrite(source_image, new_name):
    """Copy source_image as new_name, destroy existing images"""
    if new_name in bpy.data.images:
        bpy.data.images.remove(bpy.data.images[new_name])
        # img_n = bpy.data.images[name_normal]
        # img_n.scale(img.size[0], img.size[1])

    img_n = source_image.copy()
    img_n.name = new_name
    return img_n


def image_create_overwrite(new_name, w, h, itype):
    """ Possible itypes:
        ('Filmic Log', 'Linear', 'Linear ACES', 'Non-Color', 'Raw', 'sRGB', 'XYZ') """
    if new_name in bpy.data.images:
        bpy.data.images.remove(bpy.data.images[new_name])

    # img_n = bpy.data.images.new(
    #     name=new_name, width=w, height=h, float_buffer=True, generated_type="BLANK", float=True
    # )
    img_n = bpy.data.images.new(new_name, width=w, height=h, alpha=1.0)
    # Non-Color
    img_n.colorspace_settings.name = itype
    return img_n


class ImageOperator(master_ops.MacroOperator):
    def payload(self, image, context):
        # PyLance complains about unreachable code if this is set
        # assert False, "Empty payload in ImageOperator"
        pass

    def execute(self, context):
        source_image, target_image = context_get_source_target_image(context)

        input_pixels = image_to_ndarray(source_image)

        # with utils.Profile_this(lines=10):
        result = self.payload(input_pixels, context)

        ndarray_to_image(target_image, result)

        return {"FINISHED"}


class ImageOperatorGenerator(master_ops.OperatorGenerator):
    def __init__(self, master_name):
        self.init_begin(master_name)
        self.generate()
        self.init_end()
        self.name = "IMAGE_OT_" + self.name
        self.create_op(ImageOperator, "image")
