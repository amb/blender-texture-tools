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


def create(lc):
    """ create(locals()) """
    load_these = []
    for name, obj in lc.copy().items():
        if hasattr(obj, "__bases__") and obj.__bases__[0].__name__ == "ImageOperatorGenerator":
            load_these.append(obj)

    pbuild = master_ops.PanelBuilder(
        "texture_tools", load_these, "OBUILD", "IMAGE_EDITOR", "UI", "Image"
    )
    return pbuild.register_params, pbuild.unregister_params


class ImageOperator(master_ops.MacroOperator):
    def payload(self, image, context):
        pass

    def execute(self, context):

        image = get_teximage(bpy.context)
        sourcepixels = np.float32(np.array(image.pixels).reshape(image.size[0], image.size[1], 4))
        with utils.Profile_this():
            sourcepixels = self.payload(sourcepixels, context)
        image.pixels = sourcepixels.reshape((image.size[0] * image.size[1] * 4,))

        return {"FINISHED"}


class ImageOperatorGenerator(master_ops.OperatorGenerator):
    def __init__(self, master_name):
        self.init_begin(master_name)
        self.generate()
        self.init_end()
        self.name = "IMAGE_OT_" + self.name
        self.create_op(ImageOperator, "image")
