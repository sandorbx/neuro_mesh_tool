# -*- coding: utf-8 -*-
"""
Created on Wed Sep 25 14:31:00 2024

@author: serveradmin
"""

import os
import glob
import open3d as o3d
import numpy as np
from pathlib import Path

def parse_obj_file(file_path, precision=1):
    """Parse an OBJ file and return vertices, textures, normals, and faces."""
    vertices, textures, normals, faces = [], [], [], []

    with open(file_path, 'r') as f:
        for line in f:
            if line.startswith('v '):
                vertices.append(tuple(round(float(c), precision) for c in line.strip().split()[1:]))
            elif line.startswith('vt '):
                textures.append(tuple(round(float(c), precision) for c in line.strip().split()[1:]))
            elif line.startswith('vn '):
                normals.append(tuple(round(float(c), precision) for c in line.strip().split()[1:]))
            elif line.startswith('f '):
                faces.append(line.strip().split()[1:])
    return vertices, textures, normals, faces


def update_global_indices_efficient(local_items, global_items, item_map, item_set):
    """Update the global list with unique items and maintain a mapping using a set for fast checks."""
    for idx, item in enumerate(local_items):
        if item not in item_set:
            global_index = len(global_items) + 1  
            item_map[item] = global_index
            global_items.append(item)
            item_set.add(item)  
        else:
            global_index = item_map[item]
        yield idx + 1, global_index 

def remap_face(face, vertex_indices, texture_indices, normal_indices):
    """Remap the face indices from local to global."""
    new_face = []
    for v in face:
        v_indices = v.split('/')
        v_idx = int(v_indices[0]) if v_indices[0] else 0
        vt_idx = int(v_indices[1]) if len(v_indices) > 1 and v_indices[1] else None
        vn_idx = int(v_indices[2]) if len(v_indices) > 2 and v_indices[2] else None

        # Map to global indices
        new_v_idx = vertex_indices.get(v_idx, 0)
        new_vt_idx = texture_indices.get(vt_idx) if vt_idx else None
        new_vn_idx = normal_indices.get(vn_idx) if vn_idx else None

        if new_vt_idx and new_vn_idx:
            new_v = f'{new_v_idx}/{new_vt_idx}/{new_vn_idx}'
        elif new_vt_idx:
            new_v = f'{new_v_idx}/{new_vt_idx}'
        elif new_vn_idx:
            new_v = f'{new_v_idx}//{new_vn_idx}'
        else:
            new_v = f'{new_v_idx}'

        new_face.append(new_v)
    return 'f ' + ' '.join(new_face)

def write_obj_file_batch(output_file, vertices, textures, normals, faces, batch_size=1000, precision=1):
    """Write the combined data into a single OBJ file using batch writing to improve efficiency."""
    format_str = f"{{:.{precision}f}}"  

    with open(output_file, 'w') as f:
        
        for i in range(0, len(vertices), batch_size):
            f.writelines(['v {} {} {}\n'.format(format_str.format(v[0]), format_str.format(v[1]), format_str.format(v[2]))
                          for v in vertices[i:i + batch_size]])

        
        for i in range(0, len(textures), batch_size):
            f.writelines(['vt {} {}\n'.format(format_str.format(vt[0]), format_str.format(vt[1]))
                          for vt in textures[i:i + batch_size]])

        
        for i in range(0, len(normals), batch_size):
            f.writelines(['vn {} {} {}\n'.format(format_str.format(vn[0]), format_str.format(vn[1]), format_str.format(vn[2]))
                          for vn in normals[i:i + batch_size]])

        
        for i in range(0, len(faces), batch_size):
            f.writelines([face + '\n' for face in faces[i:i + batch_size]])


def merge_obj_files(folder_path, output_file, precision=1):
    """Merge multiple OBJ files from a folder into one output file using efficient methods and batch writing."""
    obj_files = glob.glob(os.path.join(folder_path, '*.obj'))

    global_vertices, global_textures, global_normals, global_faces = [], [], [], []
    vertex_map, texture_map, normal_map = {}, {}, {}
    vertex_set, texture_set, normal_set = set(), set(), set()  

    for file in obj_files:
        vertices, textures, normals, faces = parse_obj_file(file, precision=precision)

        vertex_indices = dict(update_global_indices_efficient(vertices, global_vertices, vertex_map, vertex_set))
        texture_indices = dict(update_global_indices_efficient(textures, global_textures, texture_map, texture_set))
        normal_indices = dict(update_global_indices_efficient(normals, global_normals, normal_map, normal_set))

        for face in faces:
            global_faces.append(remap_face(face, vertex_indices, texture_indices, normal_indices))

   
    write_obj_file_batch(output_file, global_vertices, global_textures, global_normals, global_faces, precision=precision)


def expand_mesh(mesh, expansion_factor=0.01):
    """Expand the mesh by moving vertices along their normals."""
    vertices = np.asarray(mesh.vertices)
    if mesh.has_vertex_normals():
        normals = np.asarray(mesh.vertex_normals)
        expanded_vertices = vertices + expansion_factor * normals
    else:
        expanded_vertices = vertices
    mesh.vertices = o3d.utility.Vector3dVector(expanded_vertices)
    return mesh

def taubin_smoothing(mesh, lambda_val=0.5, mu_val=-0.53, iterations=4):
    """Apply Taubin smoothing to the mesh."""
    for _ in range(iterations):
        smooth_vertices = np.asarray(mesh.filter_smooth_simple(number_of_iterations=1).vertices)
        mesh.vertices = o3d.utility.Vector3dVector((1 + lambda_val) * np.asarray(mesh.vertices) - lambda_val * smooth_vertices)

        smooth_vertices = np.asarray(mesh.filter_smooth_simple(number_of_iterations=1).vertices)
        mesh.vertices = o3d.utility.Vector3dVector((1 + mu_val) * np.asarray(mesh.vertices) - mu_val * smooth_vertices)
    return mesh

def preprocess_point_cloud(pcd, voxel_size=0.01):
    """Down-sample point cloud using a voxel grid"""
    return pcd.voxel_down_sample(voxel_size)

def compute_alpha_shape(pcd, alpha_value):
    """Generate an alpha shape from the point cloud."""
    alpha_shape = o3d.geometry.TriangleMesh.create_from_point_cloud_alpha_shape(pcd, alpha_value)
    return alpha_shape if not alpha_shape.is_empty() else None

def process_mesh(mesh, mesh_name, output_dir, params):
    """Process a mesh: expand, preprocess, compute alpha shape, smooth, simplify, and save."""
    
    if params['expand_mesh']:
        mesh = expand_mesh(mesh, params['expansion_factor'])

    
    pcd = o3d.geometry.PointCloud()
    pcd.points = mesh.vertices

    # Preprocess point cloud
    if params['preprocess_pcd']:
        pcd = preprocess_point_cloud(pcd, params['voxel_size'])

    # Compute alpha shape
    if params['compute_alpha_shape']:
        print(f"Computing alpha shape with alpha = {params['alpha_value']}")
        mesh = compute_alpha_shape(pcd, params['alpha_value'])
        if mesh is None:
            print(f"No alpha shape generated for {mesh_name}. Skipping.")
            return
    else:
        # 
        pass  # expand later

    # Apply smoothing
    if params['apply_smoothing']:
        mesh = taubin_smoothing(mesh, iterations=params['iterations'])

    # Simplify the mesh
    if params['simplify_mesh']:
        target_triangle_count = int(len(mesh.triangles) * params['reduction_factor'])
        mesh = mesh.simplify_quadric_decimation(target_number_of_triangles=target_triangle_count)

    # Name the output file including parameters
    params_str = f"exp{params['expansion_factor']}_vox{params['voxel_size']}_alpha{params['alpha_value']}_red{params['reduction_factor']}"
    output_file = output_dir / f"{mesh_name}_processed_{params_str}.obj"
    output_file.parent.mkdir(parents=True, exist_ok=True)

    # Save the mesh
    o3d.io.write_triangle_mesh(str(output_file), mesh)
    print(f"Processed mesh saved to {output_file}")

def process_subfolder(subfolder_path, output_base_dir, params):
    """Process all OBJ files in a subfolder, optionally merge them, and process the meshes."""
    precision = params['precision']
    subfolder_name = subfolder_path.name
    print(f"Processing subfolder: {subfolder_name}")

    # Output directory for this subfolder
    output_dir = output_base_dir / subfolder_name
    output_dir.mkdir(parents=True, exist_ok=True)

    # List of OBJ files in subfolder
    obj_files = list(subfolder_path.glob('*.obj'))
    if not obj_files:
        print(f"No OBJ files found in {subfolder_name}. Skipping.")
        return

    if params['merge_objs']:
        # Merge OBJ files
        merged_obj_file = output_dir / f'{subfolder_name}_merged.obj'
        merge_obj_files(str(subfolder_path), str(merged_obj_file), precision=precision)
        print(f"Merged OBJ files saved to {merged_obj_file}")

        # Load the merged OBJ file
        mesh = o3d.io.read_triangle_mesh(str(merged_obj_file))

        # Check if the mesh has vertices and faces
        if not mesh.has_vertices() or not mesh.has_triangles():
            print(f"Merged mesh in {subfolder_name} is empty or invalid. Skipping.")
            return
        # Process the merged mesh
        process_mesh(mesh, subfolder_name, output_dir, params)

        # Optionally delete the merged OBJ file
        if params['delete_merged']:
            os.remove(merged_obj_file)
    else:
        # Process each OBJ file individually
        for obj_file in obj_files:
            mesh_name = obj_file.stem
            mesh = o3d.io.read_triangle_mesh(str(obj_file))

            if not mesh.has_vertices() or not mesh.has_triangles():
                print(f"Mesh {mesh_name} is empty or invalid. Skipping.")
                continue

            process_mesh(mesh, mesh_name, output_dir, params)

#USER INPUT#
'''Set your processes and parameters below'''

def main():
    # Parameters
    params = {
    # General Parameters
    'merge_objs': True,
    'delete_merged': True,

    # Mesh Processing Parameters
    'expand_mesh': False,
    'expansion_factor': 0.06,

    'preprocess_pcd': False,
    'voxel_size': 0.01,

    'compute_alpha_shape': False,
    'alpha_value': 8,

    'apply_smoothing': False,
    'iterations': 8,

    'simplify_mesh': True,
    'reduction_factor': 0.5,

    # Precision Parameter
    'precision': 2  # Set the decimal precision
}


    # Define the base directory containing subfolders
    base_dir = Path(r"path")  # Replace with your base directory
    output_base_dir = Path(r"path")  # Replace with your output directory
    output_base_dir.mkdir(parents=True, exist_ok=True)

    
    for subfolder in base_dir.iterdir():
        if subfolder.is_dir():
            process_subfolder(subfolder, output_base_dir, params)

if __name__ == '__main__':
    main()
