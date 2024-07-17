python ./tools/depth_predict.py \
  --sketch_path ./assert/golden_fish.png \
  --prompt "a 3D model of realistic golden fish" \
  --output_image ./assert/golden_fish_depth.png \
  --output_np ./assert/golden_fish_depth.npy \
  --output_warp_path ./assert/golden_fish_depth_warp.png \
  --seed 69646
