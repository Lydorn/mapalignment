import os.path
import numpy as np

import bpy, bmesh

import config

# --- Params --- #

OUTPUT_BASE_DIRPATH = os.path.join(config.PROJECT_DIR, "3d_buildings/leibnitz")

SCALE = 0.1

IMAGE_HEIGHT = 12360 * 0.5  # In meters
IMAGE_WIDTH = 17184 * 0.5  # In meters

UV_SCALE = (1 / (IMAGE_HEIGHT * SCALE), 1 / (IMAGE_WIDTH * SCALE))  # (u, v)

# ---  --- #


def build_buildings(polygon_list, heights):
    bm = bmesh.new()
    uv_layer = bm.loops.layers.uv.new()
        
    for index, (polygon, height) in enumerate(zip(polygon_list, heights)):
        if index % 1000 == 0:
            print("Progress: {}/{}".format(index + 1, len(polygon_list)))
            
        verts = []
        for p in polygon:
            vert = bm.verts.new((p[1], - p[0], 0))
            verts.append(vert)
        face = bm.faces.new(verts)
        
        for p, loop in zip(polygon, face.loops):
            loop[uv_layer].uv = (p[1] * UV_SCALE[0], 1 - p[0] * UV_SCALE[1])

        # Extrude by height
        r = bmesh.ops.extrude_discrete_faces(bm, faces=[face])
        bmesh.ops.translate(bm, vec=(0, 0, height), verts=r['faces'][0].verts)

    bm.normal_update()

    me = bpy.data.meshes.new("polygon")
    bm.to_mesh(me)

    ob = bpy.data.objects.new("building", me)
    bpy.context.scene.objects.link(ob)
    bpy.context.scene.update()


# Load building footprints
polygon_list = np.load(os.path.join(OUTPUT_BASE_DIRPATH, "polygons.npy"))
scaled_polygon_list = [SCALE * polygon for polygon in polygon_list]
heights = np.load(os.path.join(OUTPUT_BASE_DIRPATH, "heights.npy"))
scaled_heights = SCALE * heights

# Build each building one at a time
print("# --- Starting to build buildings: --- #")
build_buildings(scaled_polygon_list, scaled_heights)
print("# --- Finished building buildings --- #")
