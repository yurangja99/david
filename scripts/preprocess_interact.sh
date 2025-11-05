categories=(
  "largetable_carry"
  "largetable_lift"
  "smallbox"
  "clothesstand_left_hand"
  "clothesstand_right_hand"
  "clothesstand_two_hands"
)
crop_start=( 30 45 30 15 15 20 )
crop_end=( 35 50 35 15 15 20 )
sudo mkdir logs
for i in "${!categories[@]}"; do
  python src/david/preprocess_interact.py \
    --input_dir data/OMOMO_retarget_new_subsets \
    --pose_dir results/david_retarget/pose_data \
    --obj_dir results/david_retarget/obj_data \
    --dataset FullBodyManip \
    --category ${categories[i]} \
    --crop_start ${crop_start[i]} \
    --crop_end ${crop_end[i]} \
    --device 1
done