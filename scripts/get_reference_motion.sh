categories=(
  "clothesstand_left_hand"
  "clothesstand_right_hand"
  "clothesstand_two_hand"
  "largetable_two_hand_carry"
  "largetable_two_hand_drag"
  "largetable_two_hand_lift"
  "smallbox_two_hand_carry"
  "smallbox_two_hand_drag"
)
for i in "${!categories[@]}"; do
  python src/david/preprocess_interact.py \
    --input_dir data/InterAct \
    --pose_dir results/david_ref/pose_data \
    --obj_dir results/david_ref/obj_data \
    --dataset FullBodyManip \
    --category ${categories[i]} \
    --reference
done
for i in "${!categories[@]}"; do
  CUDA_VISIBLE_DEVICES=1 python src/david/process_mdm.py \
    --dataset FullBodyManip \
    --category ${categories[i]} \
    --pose_dir results/david_ref/pose_data \
    --new_joints_dir results/david_ref/new_joints \
    --new_joint_vecs_dir results/david_ref/new_joint_vecs \
    --mdm_dir results/david_ref/mdm
done
for i in "${!categories[@]}"; do
  CUDA_VISIBLE_DEVICES=1 python src/david/process_omdm.py \
    --dataset FullBodyManip \
    --category ${categories[i]} \
    --obj_dir results/david_ref/obj_data \
    --hoi_data_dir results/david_ref/hoi_data \
    --joint_dir results/david_ref/pose_data
done