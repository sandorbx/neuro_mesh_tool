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
import trimesh
from scipy.ndimage import binary_fill_holes
from scipy import ndimage
import tifffile as tiff
from tqdm import tqdm
from PIL import Image

def translate_mesh(mesh, dx=0.0, dy=0.0, dz=0.0):
    """Translate an Open3D mesh by (dx, dy, dz)."""
    verts = np.asarray(mesh.vertices)
    verts = verts + np.array([dx, dy, dz], dtype=verts.dtype)
    mesh.vertices = o3d.utility.Vector3dVector(verts)
    return mesh

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
    format_str = f"{{:.{precision}f}}"  # Format string for precision

    with open(output_file, 'w') as f:
        # Writing vertices in batches
        for i in range(0, len(vertices), batch_size):
            f.writelines(['v {} {} {}\n'.format(format_str.format(v[0]), format_str.format(v[1]), format_str.format(v[2]))
                          for v in vertices[i:i + batch_size]])

        # Writing texture coordinates in batches
        if textures:
            for i in range(0, len(textures), batch_size):
                f.writelines(['vt {} {}\n'.format(format_str.format(vt[0]), format_str.format(vt[1]))
                              for vt in textures[i:i + batch_size]])

        # Writing normals in batches
        if normals:
            for i in range(0, len(normals), batch_size):
                f.writelines(['vn {} {} {}\n'.format(format_str.format(vn[0]), format_str.format(vn[1]), format_str.format(vn[2]))
                              for vn in normals[i:i + batch_size]])

        # Writing faces in batches
        if len(faces) > 0:  # Ensure the faces list is not empty
            if isinstance(faces[0], (tuple, list, np.ndarray)):  # Processed mesh (integer indices)
                for i in range(0, len(faces), batch_size):
                    f.writelines(['f {} {} {}\n'.format(face[0], face[1], face[2]) for face in faces[i:i + batch_size]])
            else:  # Merged mesh (string format, already remapped)
                for i in range(0, len(faces), batch_size):
                    f.writelines([face + '\n' for face in faces[i:i + batch_size]])



def merge_obj_files(folder_path, output_file, precision=1):
    """Merge multiple OBJ files from a folder into one output file"""
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
    """Process a mesh according to specified parameters."""
    
    # First, compute alpha shape if required
    if params['compute_alpha_shape']:
        # Preprocess point cloud as a parameter of alpha shape
        print(f"Computing alpha shape with alpha = {params['alpha_value']}")
        pcd = o3d.geometry.PointCloud()
        pcd.points = mesh.vertices
        pcd = preprocess_point_cloud(pcd, params['voxel_size'])
        mesh = compute_alpha_shape(pcd, params['alpha_value'])
        if mesh is None:
            print(f"No alpha shape generated for {mesh_name}. Skipping.")
            return
    else:
        # If not computing alpha shape, create point cloud from mesh (justincase)
        pcd = o3d.geometry.PointCloud()
        pcd.points = mesh.vertices

    # Second, simplify the mesh if required
    if params['simplify_mesh']:
        target_triangle_count = int(len(mesh.triangles) * params['reduction_factor'])
        mesh = mesh.simplify_quadric_decimation(target_number_of_triangles=target_triangle_count)

    # Third, apply smoothing if required
    if params['apply_smoothing']:
        # Expand mesh 
        if params['expand_mesh']:
            mesh = expand_mesh(mesh, params['expansion_factor'])
        mesh = taubin_smoothing(mesh, iterations=params['iterations'])

    # flip the mesh
    if params['flip_mesh']:
        print(f"Flipping mesh {mesh_name} horizontally...")
        mesh = flip_mesh_horizontally(mesh, params['brain_size_x'])
        print(f"Mesh {mesh_name} flipped.")

    if params.get('translate_mesh', False):
        dx, dy, dz = params.get('translation_xyz', (0.0, 0.0, 0.0))
        print(f"Translating mesh {mesh_name} by ({dx}, {dy}, {dz})...")
        mesh = translate_mesh(mesh, dx, dy, dz)
        print("Translation applied.")        


    # Build parameters string
    params_str_list = []

    if params['compute_alpha_shape']:
        params_str_list.append(f"alpha{params['alpha_value']}")
        params_str_list.append(f"vox{params['voxel_size']}")

    if params['simplify_mesh']:
        params_str_list.append(f"red{params['reduction_factor']}")

    if params['apply_smoothing']:
        params_str_list.append(f"smooth{params['iterations']}")
        if params['expand_mesh']:
            params_str_list.append(f"exp{params['expansion_factor']}")

    if params['flip_mesh']:
        params_str_list.append("flipped")
        
    if params.get('translate_mesh', False):
        dx, dy, dz = params.get('translation_xyz', (0.0, 0.0, 0.0))
        # Short, filename-safe tag; rounded to your 'precision'
        fmt = f"{{:.{params['precision']}f}}"
        params_str_list.append(
            f"t{fmt.format(dx)}_{fmt.format(dy)}_{fmt.format(dz)}"
        )
        

    if params['voxelize_mesh']:
        params_str_list.append("voxmesh")

    params_str_list.append(f"prec{params['precision']}")
    params_str = '_'.join(params_str_list)

    # Name the output file including active parameters
    output_file = output_dir / f"{mesh_name}_processed_{params_str}.obj"
    output_file.parent.mkdir(parents=True, exist_ok=True)

    # Save the processed mesh as OBJ
    vertices = np.asarray(mesh.vertices)
    faces = np.asarray(mesh.triangles) + 1  # OBJ indices start at 1

    write_obj_file_batch(output_file, vertices, [], [], faces, batch_size=1000, precision=params['precision'])
    print(f"Processed mesh saved to {output_file}")

    # Voxelization process
    if params['voxelize_mesh']:
        print("Starting voxelization process...")
        # Convert Open3D mesh to Trimesh
        trimesh_mesh = trimesh.Trimesh(vertices=vertices, faces=faces - 1)  # Faces start at 0 in Trimesh
        min_coords = trimesh_mesh.vertices.min(axis=0)

        # Calculate transformation factors
        scale_factor = calculate_transformation_factors(params['template_voxel_size'])
        transformed_min_coords = transform_coordinates(min_coords, scale_factor)

        # Voxelize the mesh
        voxel_grid = voxelize_mesh(trimesh_mesh, pitch=params['pitch'])
        print(f"Voxelization complete. Voxel grid shape: {voxel_grid.shape}")

        # Interpolate voxel grid
        voxel_grid = interpolate_voxel_grid(
            voxel_grid,
            downscale_factor_xy=params['downscale_factor_xy'],
            downscale_factor_z=params['downscale_factor_z'],
            fill_volume=params.get('fill_volume', False)
        )
        print(f"Interpolation complete. Voxel grid shape: {voxel_grid.shape}")

        # Integrate into template grid
        final_grid = integrate_into_template_grayscale(
            voxel_grid,
            params['template_dims'],
            params['template_voxel_size'],
            min_coords
        )
        print(f"Integration complete. Final grid shape: {final_grid.shape}")

        # Save as TIFF
        if params.get('save_as_tiff', True):
            output_tiff_file = output_dir / f"{mesh_name}_processed_{params_str}.tif"
            save_as_tiff(final_grid, output_tiff_file)
            print(f"Voxelized mesh saved as TIFF at {output_tiff_file}")


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

        # Read merged mesh
        mesh = o3d.io.read_triangle_mesh(str(merged_obj_file))

        if not mesh.has_vertices() or not mesh.has_triangles():
            print(f"Merged mesh in {subfolder_name} is empty or invalid. Skipping.")
            return

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



def flip_obj_horizontally(input_path, output_path, brain_size_x):
    flipped_lines = []

    with open(input_path, 'r') as obj_file:
        for line in obj_file:
            # Process vertex lines, which start with 'v '
            if line.startswith('v '):
                parts = line.split()
                # Parse the x, y, z coordinates
                x, y, z = map(float, parts[1:4])
                # Perform the horizontal flip (x-axis flip around the center)
                x_flipped = brain_size_x - x
                # Reconstruct the line with the flipped x-coordinate
                flipped_line = f"v {x_flipped} {y} {z}\n"
                flipped_lines.append(flipped_line)
            else:
                flipped_lines.append(line)

    # Write the flipped content to a new OBJ file
    with open(output_path, 'w') as output_file:
        output_file.writelines(flipped_lines)

def flip_mesh_horizontally(mesh, brain_size_x):
    """Flip an Open3D mesh horizontally around the center."""
    vertices = np.asarray(mesh.vertices)
    vertices[:, 0] = brain_size_x - vertices[:, 0]
    mesh.vertices = o3d.utility.Vector3dVector(vertices)
    return mesh


def calculate_transformation_factors(template_voxel_size):
    scale_factor = 1 / template_voxel_size
    return scale_factor

def transform_coordinates(min_coords, scale_factor):
    return min_coords * scale_factor

def voxelize_mesh(mesh, pitch):
    voxel_grid = mesh.voxelized(pitch=pitch).matrix.astype(bool)
    return voxel_grid

def resize_xy_slices(voxel_volume, scale_factor):
    z, y, x = voxel_volume.shape
    new_y = int(np.ceil(y * scale_factor))
    new_x = int(np.ceil(x * scale_factor))
    resized_slices = []

    for i in tqdm(range(z), desc="2D Lanczos Resampling (XY)"):
        slice_ = voxel_volume[i, :, :]
        img = Image.fromarray(slice_)
        img_resized = img.resize((new_x, new_y), resample=Image.LANCZOS)
        resized_slice = np.array(img_resized)
        resized_slices.append(resized_slice)

    resized_volume = np.stack(resized_slices, axis=0)
    return resized_volume

def interpolate_voxel_grid(voxel_grid, downscale_factor_xy=None, downscale_factor_z=None, fill_volume=False):
    voxel_volume = voxel_grid.astype(np.uint8)
    voxel_volume[voxel_volume > 0] = 200

    if fill_volume:
        print("Filling interior voxels...")
        filled = binary_fill_holes(voxel_volume).astype(np.uint8) * 200
        voxel_volume = filled
        print("Interior voxels filled.")

    if downscale_factor_xy and downscale_factor_xy != 1.0:
        print(f"Rescaling XY dimensions by a factor of {downscale_factor_xy}...")
        voxel_volume = resize_xy_slices(voxel_volume, 1 / downscale_factor_xy)
        print("XY rescaling completed.")

    if downscale_factor_z and downscale_factor_z > 1.0:
        print(f"Downscaling Z dimension by a factor of {downscale_factor_z} using linear interpolation...")
        z, y, x = voxel_volume.shape
        new_z = int(np.ceil(z / downscale_factor_z))

        original_z_indices = np.arange(z)
        new_z_indices = np.linspace(0, z - 1, new_z)

        rescaled_volume = np.zeros((new_z, voxel_volume.shape[1], voxel_volume.shape[2]), dtype=voxel_volume.dtype)

        for y_idx in tqdm(range(voxel_volume.shape[1]), desc="1D Linear Resampling (Z)", leave=True):
            for x_idx in range(voxel_volume.shape[2]):
                column = voxel_volume[:, y_idx, x_idx].astype(float)
                resampled_column = np.interp(new_z_indices, original_z_indices, column)
                rescaled_volume[:, y_idx, x_idx] = resampled_column.astype(np.uint8)

        voxel_volume = rescaled_volume
        print("Z downscaling with linear interpolation completed.")
    elif downscale_factor_z and downscale_factor_z <= 1.0:
        raise ValueError("Z downscale factor should be greater than 1.")

    return voxel_volume

def integrate_into_template_grayscale(voxel_grid, template_dims, template_voxel_size, min_coords):
    scale_factor = 1 / template_voxel_size
    target_coords = transform_coordinates(min_coords, scale_factor)
    target_coords = target_coords.astype(int)

    final_grid = np.zeros(template_dims, dtype=np.uint8)

    min_extent = np.minimum(final_grid.shape, voxel_grid.shape + target_coords)

    start_final = np.maximum(target_coords, 0)
    end_final = start_final + voxel_grid.shape

    start_voxel = np.maximum(-target_coords, 0)
    end_voxel = start_voxel + (min_extent - start_final)

    final_slices = tuple(
        slice(start_final[dim], min(end_final[dim], final_grid.shape[dim]))
        for dim in range(3)
    )

    voxel_slices = tuple(
        slice(start_voxel[dim], end_voxel[dim])
        for dim in range(3)
    )

    if voxel_grid.dtype == bool:
        final_grid[final_slices] = np.maximum(final_grid[final_slices], voxel_grid[voxel_slices].astype(np.uint8) * 255)
    else:
        final_grid[final_slices] = np.maximum(final_grid[final_slices], voxel_grid[voxel_slices])

    return final_grid

def save_as_tiff(voxel_grid, file_path):
    voxel_grid = np.transpose(voxel_grid, (2, 1, 0))  # Convert to (X, Y, Z) for TIFF
    tiff.imwrite(str(file_path), voxel_grid.astype(np.uint8), photometric='minisblack',)
    print(f"Saved voxel grid as TIFF at: {file_path}")

#USER INPUT#
'''Set your processes and parameters below'''

def main():
    # Parameters
    params = {
        # General Parameters
        'merge_objs': False,
        

        # Flip Mesh Parameter
        'flip_mesh': False,  # Set to True to flip meshes horizontally
        'brain_size_x': 627.76,
        
    

        # Mesh Processing Parameters
        
        # Translation (final step) -------------------------
        'translate_mesh': True,                  # set True to enable
        'translation_xyz': (10, 10, 10),       # (dx, dy, dz)        


        # Alpha parameters
        'compute_alpha_shape': False,
        'alpha_value': 12,
        'preprocess_pcd': True,  
        'voxel_size': 0.01,
        
        
        #Smoothing
        'apply_smoothing': False,
        'iterations': 32,
        'expand_mesh': False,  
        'expansion_factor': 0.06,
        
        
        #Simplify
        'simplify_mesh': False,
        'reduction_factor': 0.86,

        # Numerical Precision Parameter
        'precision': 4,  # Set the decimal precision
        
        # Voxelization Parameters
        'voxelize_mesh': False,
        'template_voxel_size': 0.4,
        'template_dims': (1156, 2284, 705),  
        'downscale_factor_xy': 2,
        'downscale_factor_z': 2,
        'pitch': 0.4/1,
        'fill_volume': True,
        'save_as_tiff': True,
        
        'delete_merged': True, #Delete the raw unprocessed file to save memory
    }




    # Define the base directory containing subfolders
    base_dir = Path(r"E:\Dropbox\labor\ITO\Image_processing_analysis\neuro-meshtools\OBJ-combine-convert-simplify\lc16-tester\lc16combined")  # Replace with your base directory
    output_base_dir = Path(r"E:\Dropbox\labor\ITO\Image_processing_analysis\neuro-meshtools\OBJ-combine-convert-simplify\lc16-tester")  # Replace with your output directory
    output_base_dir.mkdir(parents=True, exist_ok=True)

    # Gather all folders to process: subfolders and the base folder itself
    folders_to_process = [base_dir] + [subfolder for subfolder in base_dir.iterdir() if subfolder.is_dir()]

    for folder in folders_to_process:
        process_subfolder(folder, output_base_dir, params)

if __name__ == '__main__':
    main()
