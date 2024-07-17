# Make sure that the bark has been download and unzip into ./assert/bark

# 1. Rconstruct the 3D models by NeRF (We suggest the front view as 0°)
python launch.py --config assert/lion/configs/recon-system.yaml --train --gpu 1 system.prompt_processor.prompt="recon_lion"

# 2. Predict edit view results
latest_checkpoint=$(ls -dt outputs/reconstruction/recon_lion@*/ckpts/last.ckpt | head -1)
python launch_edit.py --config assert/lion/configs/recon-test-no-material.yaml system.prompt_processor.prompt="test_lion" --gpu 1 resume="$latest_checkpoint" system.material_type="no-material" system.exporter_type=mesh-exporter system.exporter.save_uv=false system.exporter.fmt=obj system.geometry.isosurface_threshold=50.

# 3. Draw the editing sketch and predict depth manually (Do not included here. You can do it by yourself)
# We just use prepared sketches and depth

# 4. Construct the coarse box: Save in the path, "outputs/test/test_lion@*/save/" 
latest_test_dir=$(ls -dt outputs/test/test_lion@*/save | head -1)
python ./tools/construct_coarse_mesh.py --radius 1.7 --root_dir "$latest_test_dir" --edit_mask_path ./assert/lion/edit_mask.png

# 5. Run the coarse editing stage
prompt="a 3D model of realistic bronze lion"
python launch.py --config assert/lion/configs/coarse-edit.yaml --train system.prompt_processor.prompt="$prompt" --gpu 1 system.ref_geometry_weights="$latest_checkpoint" system.new_geometry_weights="$latest_checkpoint"

# 6. Extracted the coarse editing mesh from NeRF: Save in the path, "outputs/test/test_lion@*/save/it5000-export/model_mask_edit.obj"
latest_edit_checkpoint=$(ls -dt outputs/coarse-edit/*/ckpts/epoch=0-step=5000.ckpt | head -1)
python launch_edit.py --config assert/lion/configs/recon-test.yaml system.prompt_processor.prompt="test_lion" --gpu 1 resume="$latest_edit_checkpoint" system.exporter_type=mesh-exporter system.exporter.save_uv=false system.exporter.fmt=obj system.geometry.isosurface_threshold=10.

# 7. Mask the mesh models by coarse box: Save in the path, "outputs/test/test_lion@*/save/it5000-export/model_mask_edit.obj"
latest_test_edit_dir=$(ls -dt outputs/test/test_lion@*/save | head -1)
python ./tools/construct_refine_mesh.py --coarse_box_path "./assert/lion/mesh_coarse_box.obj" --extract_mesh_path "$latest_test_edit_dir/it5000-export/model.obj" --refine_box_path "$latest_test_edit_dir/it5000-export/model_mask_edit.obj"

# The mask mesh can be manually correct by meshlab. Black color marks editing region, while white color marks unediting regions. 

# 8. Generate the local rendering parameters: Save in the path, "outputs/test/test_lion@*/save/it5000-export/local_render_param.txt"
# python ./tools/calculate_local_render.py --root_dir "$latest_test_edit_dir/it5000-export/"

