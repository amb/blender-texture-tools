import numpy
import bpy
import nodeitems_utils
from nodeitems_utils import NodeCategory, NodeItem

#image_processing_nodes
class ImageSocket(bpy.types.NodeSocket):
    bl_idname = 'ImageSocket'
    bl_label = 'Image Node Socket'

    t_value = numpy.zeros((4,2), dtype=numpy.float16)

    # Optional function for drawing the socket input value
    def draw(self, context, layout, node, text):
        layout.label(text)

    # Socket color
    def draw_color(self, context, node):
        return 0.0, 0.8, 0.0, 0.5


class ImageEditNodeTree(bpy.types.NodeTree):
    bl_idname = 'ImageTreeType'
    bl_label = 'Image Edit Node Tree'
    bl_icon = 'COLOR'


class ImageInputNode(bpy.types.Node):
    bl_idname = 'ImageInputNodeType'
    bl_label = 'Image Input Node'

    available_objects = []
    for im in bpy.data.images:
        name = im.name
        available_objects.append((name, name, name))

    input_image = bpy.props.EnumProperty(name="", items=available_objects)

    @classmethod
    def poll(cls, tree):
        return tree.bl_idname == 'ImageTreeType'

    def draw_buttons(self, context, layout):
        # layout.label("Node settings")
        layout.prop(self, "input_image")

    def init(self, context):
        self.outputs.new('ImageSocket', "image out")

    def update(self):
        size = bpy.data.images[self.input_image].size
        pixels = numpy.array(bpy.data.images[self.input_image].pixels)
        pixels = pixels.reshape((size[0], size[1], 4))
        self.outputs['image out'].t_value.resize(pixels.shape)
        self.outputs['image out'].t_value[:] = pixels
        print('OO ' + repr(self.outputs['image out']))
        #print('OO ' + repr(self.outputs['image out'].t_value))
        print(self.input_image)


# noinspection PyAttributeOutsideInit,PyAttributeOutsideInit
class ImageViewNode(bpy.types.Node):
    bl_idname = 'ImageViewNodeType'
    bl_label = 'Image View Node'

    @classmethod
    def poll(cls, tree):
        return tree.bl_idname == 'ImageTreeType'

    def draw_buttons(self, context, layout):
        # print(self.img_name)
        layout.label('foo')

    def init(self, context):
        print("node init")
        self.inputs.new('ImageSocket', "image in")
        self.img_name = "init"

    def update(self):
        self.img_name = 'foo'
        iny = "invalid"
        if self.inputs['image in'].is_linked and self.inputs['image in'].links[0].is_valid:
            iny = self.inputs['image in'].links[0].from_socket.t_value
        else:
            # iny = self.inputs['image in'].default_value
            iny = "error"
        #print("inb: "+self.inputs['image in'].links[0].from_socket.t_value)
        print("II "+repr(iny))

class ImageOutputNode(bpy.types.Node):
    bl_idname = 'ImageOutputNodeType'
    bl_label = 'Image Output Node'

    output_image = bpy.props.StringProperty(name="", default="generated")

    @classmethod
    def poll(cls, tree):
        return tree.bl_idname == 'ImageTreeType'

    def draw_buttons(self, context, layout):
        layout.prop(self, "output_image")

    def init(self, context):
        print("node init")
        self.inputs.new('ImageSocket', "image in")
        self.img_name = "init"

    def update(self):
        if self.inputs['image in'].is_linked and self.inputs['image in'].links[0].is_valid:
            iny = self.inputs['image in'].links[0].from_socket.t_value
        else:
            iny = "error"
        print("O NODE: "+repr(self.inputs['image in'].links[0].from_socket.t_value.shape))

        input_image = self.inputs['image in'].links[0].from_socket.t_value
        # if target image exists, change the size to fit
        if self.output_image in bpy.data.images:
            x,y = int(input_image.shape[0]), int(input_image.shape[1])
            tx, ty = int(bpy.data.images[self.output_image].size[0]), int(bpy.data.images[self.output_image].size[1]) 
            if tx != x or ty != y:
                bpy.data.images[self.output_image].scale(x, y)
            image = bpy.data.images[self.output_image]
        else:
            image = bpy.data.images.new(self.output_image, width=self.xs, height=self.ys)

        # assign pixels
        image.pixels = input_image.flatten()
        bpy.ops.image.invert(invert_r=False, invert_g=False, invert_b=False, invert_a=False)

class MyNodeCategory(NodeCategory):
    @classmethod
    def poll(cls, context):
        return context.space_data.tree_type == 'ImageTreeType'


node_categories = [
    # identifier, label, items list
    MyNodeCategory("IMAGEPNODES", "Image Processing Nodes", items=[
        NodeItem("ImageInputNodeType"),
        NodeItem("ImageViewNodeType"),
        NodeItem("ImageOutputNodeType"),
    ]),
]

# ### INITIALIZATION

# register classes
regclasses = [ImageInputNode, ImageViewNode, ImageOutputNode]

def register():
    for entry in regclasses:
        bpy.utils.register_class(entry)

    nodeitems_utils.register_node_categories("IMAGE_PROCESSING_NODES", node_categories)


def unregister():
    nodeitems_utils.unregister_node_categories("IMAGE_PROCESSING_NODES")

    for entry in regclasses:
        bpy.utils.unregister_class(entry)

if __name__ == "__main__":
    #unregister()
    register()
