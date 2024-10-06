# neuro_mesh_tool
 merge, simplify, smooth and wrap neuron OBJs
 
 input-output : it will batch process a folder of OBJ files, if you organized them into subfolders the merging script will treat these separately and the merged obj will bear the name of the corresponding subfolder, the output folder will mimic the structure of the input directory. 

Functions:

'merge_objs':  merge large batches of OBJ files without vertex conflicts

'simplify_mesh' apply quadratic decimation to streamline the mesh file size, without losing important details, will save lots of memory


'compute_alpha_shape' remesh the surface with Delaunay triangulation at lower values, higher values will shrinkwrap the neurons, revealing abstract block 
                      shape for morphometric evaluation.

'apply_smoothing' custom Taubin smoothing, apply it to blocky meshes generated from your segmentations.


'flip_mesh' flipping between brain halves 


'voxelize_mesh' advanced conversion to image stack, using supersampling and Lanczos interpolation, high visual fidelity, at common viewing distances 
                indistinguishable from the original mesh 

