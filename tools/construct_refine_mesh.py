import os
import argparse
import trimesh
import numpy as np
import torch
import cubvh

# Save the OBJ
def save_obj_with_color(vertices, colors, faces, filename):
    with open(filename, 'w') as f:
        # Write the vertex information
        for vertex, color in zip(vertices, colors):
            f.write(f'v {vertex[0]} {vertex[1]} {vertex[2]} {color[0]} {color[1]} {color[2]}\n')
        
        # Write the face information
        for face in faces:
            f.write(f'f {face[0]+1} {face[1]+1} {face[2]+1}\n')

def main():
    parser = argparse.ArgumentParser(description="Construct coarse 3D meshes")
    
    parser.add_argument('--coarse_box_path', type=str, required=True, default=None, help='Path of the coarse mask mesh')
    parser.add_argument('--extract_mesh_path', type=str, required=True, default=None, help='Path of the extract mesh')
    parser.add_argument('--refine_box_path', type=str, required=True, default=None, help='Path of the refine mask mesh')
    
    args = parser.parse_args()
    
    save_mesh_path = args.refine_box_path
    mesh_path = args.extract_mesh_path
    coarse_box_path = args.coarse_box_path
    
    print("save_mesh_path:", save_mesh_path)
    print("mesh_path:", mesh_path)
    print("coarse_box_path:", coarse_box_path)

    # Local local mesh
    mesh = trimesh.load(mesh_path)
    vertice = torch.tensor(mesh.vertices, dtype=torch.float32).to('cuda')
    positions = vertice

    # Local coarse mesh box
    mesh_box = trimesh.load(coarse_box_path)
    BVH_mesh = cubvh.cuBVH(mesh_box.vertices, mesh_box.faces) # build with numpy.ndarray/torch.Tensor
    distances, face_id, uvw = BVH_mesh.signed_distance(positions, return_uvw=True, mode='raystab') # [N], [N], [N, 3]

    vertices_np = mesh.vertices
    faces_np = mesh.faces
    color = np.ones(vertices_np.shape)
    vertice_mask = distances.cpu().numpy() < 0.0
    print(vertice_mask.shape)
    color[vertice_mask] = 0.0

    save_obj_with_color(vertices_np, color, faces_np, save_mesh_path)

if __name__ == "__main__":
    main()
