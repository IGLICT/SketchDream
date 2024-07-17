# Make sure that the lion has been download and unzip into ./assert/lion

# 1. Reconstruct the models by NeRF
python launch.py --config assert/lion/configs/recon-system.yaml --train --gpu 1 system.prompt_processor.prompt="recon_lion"

# 2. Running the refine editing stage
prompt="a 3D model of bronze lion"
latest_checkpoint=$(ls -dt outputs/reconstruction/recon_lion@*/ckpts/last.ckpt | head -1)
python launch.py --config assert/lion/configs/refine-edit.yaml --train system.prompt_processor.prompt="$prompt" --gpu 1 system.ref_geometry_weights="$latest_checkpoint" system.new_geometry_weights="$latest_checkpoint"
