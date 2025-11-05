categories=(
  "largetable_carry"
  "largetable_lift"
  "smallbox"
  "clothesstand_left_hand"
  "clothesstand_right_hand"
  "clothesstand_two_hands"
)
hand_info=(0 0 0 1 2 0)
for i in "${!categories[@]}"; do
  CUDA_VISIBLE_DEVICES=1 python src/david/process_omdm.py \
    --dataset FullBodyManip \
    --category ${categories[i]} \
    --obj_dir results/david_retarget/obj_data \
    --hoi_data_dir results/david_retarget/hoi_data \
    --joint_dir results/david_retarget/pose_data \
    --hand_info ${hand_info[i]}
done