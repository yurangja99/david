categories=(
  "largetable_carry"
  "largetable_lift"
  "smallbox"
  "clothesstand_left_hand"
  "clothesstand_right_hand"
)
ref_path=(
  "data/OMOMO_retarget_new_subsets/largetable_carry/sub1_largetable_011.pt"
  "data/OMOMO_retarget_new_subsets/largetable_lift/sub1_largetable_027.pt"
  "data/OMOMO_retarget_new_subsets/smallbox/sub17_smallbox_041.pt"
  "data/OMOMO_retarget_new_subsets/clothesstand_left_hand/sub13_clothesstand_020.pt"
  "data/OMOMO_retarget_new_subsets/clothesstand_right_hand/sub11_clothesstand_021.pt"
)
crop_start=( 30 30 25 15 15 )
blending_frame=( 15 15 15 15 15 )  # 30fps
smoothing_frame=( 8 8 8 8 8 ) # 30fps
hand_info=( 0 0 0 1 2 )
for i in "${!categories[@]}"; do
  python src/david/postprocess_interact.py \
    --dataset FullBodyManip \
    --category ${categories[i]} \
    --output_dir "results/OURS_251105/${categories[i]}" \
    --human_motion_dir results/inference_251105/human_motion \
    --object_motion_dir results/inference_251105/object_motion \
    --ref_path "${ref_path[i]}" \
    --ref_crop_start ${crop_start[i]} \
    --ref_bf ${blending_frame[i]} \
    --ref_sf ${smoothing_frame[i]} \
    --hand_info ${hand_info[i]}
done