categories=(
  "largetable_carry"
  "largetable_lift"
  "smallbox"
  "clothesstand_left_hand"
  "clothesstand_right_hand"
  "clothesstand_two_hands"
)
for i in "${!categories[@]}"; do
  CUDA_VISIBLE_DEVICES=0 python src/david/process_mdm.py \
    --dataset FullBodyManip \
    --category ${categories[i]} \
    --pose_dir results/david_retarget/pose_data \
    --new_joints_dir results/david_retarget/new_joints \
    --new_joint_vecs_dir results/david_retarget/new_joint_vecs \
    --mdm_dir results/david_retarget/mdm
done