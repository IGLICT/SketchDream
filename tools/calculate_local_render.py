import os
import argparse
import numpy as np
import trimesh
from scipy.spatial import ConvexHull

def generate_sphere(radius, center=(0, 0, 0), num_points=100):
    u = np.linspace(0, 2 * np.pi, num_points)
    v = np.linspace(0, np.pi, num_points)
    x = radius * np.outer(np.cos(u), np.sin(v))
    y = radius * np.outer(np.sin(u), np.sin(v))
    z = radius * np.outer(np.ones(np.size(u)), np.cos(v))
    x += center[0]
    y += center[1]
    z += center[2]
    return x, y, z

def save_obj(filename, vertices, faces):
    with open(filename, 'w') as file:
        for vertex in vertices:
            file.write(f"v {vertex[0]} {vertex[1]} {vertex[2]}\n")
        for face in faces:
            file.write(f"f {face[0]} {face[1]} {face[2]}\n")

def main():
    parser = argparse.ArgumentParser(description="Construct coarse 3D meshes")
    
    parser.add_argument('--root_dir', type=str, required=True, default=None, help='Directory of root')
    parser.add_argument('--edit_mesh_path', type=str, required=False, default="model_mask_edit.obj", help='Path of the mask meshes')
    parser.add_argument('--sphere_path', type=str, required=False, default="local_render_visualization.obj", help='Save path of a sphere that visualize the cameras')
    parser.add_argument('--text_path', type=str, required=False, default="local_render_param.txt", help='Save path of the local rendering parameters')
    
    args = parser.parse_args()
    
    mesh = trimesh.load_mesh(os.path.join(args.root_dir, args.edit_mesh_path))
    sphere_path = os.path.join(args.root_dir, args.sphere_path)
    text_path = os.path.join(args.root_dir, args.text_path)
    
    # Read the vertex color
    vertex_colors = mesh.visual.vertex_colors
    vertices = mesh.vertices
    faces = mesh.faces

    print(vertices.shape)

    index = vertex_colors[:,0] < 125
    points_array = vertices[index,:]

    print(points_array.shape)

    # Generate the ball that surrond the editing regions
    hull = ConvexHull(points_array)
    min_sphere_center = np.mean(points_array[hull.vertices], axis=0)
    min_sphere_radius = np.max(np.linalg.norm(points_array - min_sphere_center, axis=1))

    print("Center:", min_sphere_center)
    print("Radius:", min_sphere_radius)

    sketch_fov_deg = 40.0
    sketch_fov = np.deg2rad(sketch_fov_deg / 2.)
    radius_render = min_sphere_radius / np.tan(sketch_fov) * 1.2
    print("Radius Render:", radius_render)

    # Save local rendering parameters into txt files
    with open(text_path, 'w') as file:
        print("Center:", min_sphere_center, file=file)
        print("Radius Render:", radius_render, file=file)
        print("Radius:", min_sphere_radius, file=file)

    # viusulize the local rendering parameters
    radius = min_sphere_radius
    center = min_sphere_center
    num_points = 50

    x, y, z = generate_sphere(radius, center, num_points)

    vertices = np.array([x.flatten(), y.flatten(), z.flatten()]).T
    faces = []

    for i in range(num_points - 1):
        for j in range(num_points - 1):
            face = [
                i * num_points + j + 1,
                i * num_points + j + 2,
                (i + 1) * num_points + j + 2,
                (i + 1) * num_points + j + 1
            ]
            faces.append(face)

    save_obj(sphere_path, vertices, faces)
    
if __name__ == "__main__":
    main()
