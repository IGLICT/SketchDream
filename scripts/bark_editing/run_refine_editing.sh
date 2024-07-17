# Make sure that the bark has been download and unzip into ./assert/bark

# 1. Reconstruct the models by NeRF
python launch.py --config assert/bark/configs/recon-system.yaml --train --gpu 1 system.prompt_processor.prompt="recon_bark"

# 2. Running the refine editing stage
latest_checkpoint=$(ls -dt outputs/reconstruction/recon_bark@*/ckpts/last.ckpt | head -1)
prompt="a 3D model of realistic mushrooms on bark"
python launch.py --config assert/bark/configs/refine-edit.yaml --train system.prompt_processor.prompt="$prompt" --gpu 1 system.ref_geometry_weights="$latest_checkpoint" system.new_geometry_weights="$latest_checkpoint"
