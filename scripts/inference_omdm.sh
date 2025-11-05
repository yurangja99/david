categories=(
  "largetable_carry"
  "largetable_lift"
  "smallbox"
  "clothesstand_left_hand"
  "clothesstand_right_hand"
)
epochs=(44100 44800 44400 44700 25500)
ref_idx=(0 0 0 0 0)
blending_frame=(10 10 10 10 10)
hand_info=(0 0 0 1 2)
for i in "${!categories[@]}"; do
  CUDA_VISIBLE_DEVICES=0 python src/david/inference_omdm.py \
    --dataset FullBodyManip \
    --category ${categories[i]} \
    --hoi_data_dir results/david_retarget_251028/hoi_data \
    --omdm_dir results/david_retarget_251028/omdm \
    --inference_human_motion_dir results/inference_251105/human_motion \
    --inference_object_motion_dir results/inference_251105/object_motion \
    --inference_contact_threshold 0.05 \
    --inference_epoch ${epochs[i]} \
    --ref_bf ${blending_frame[i]} \
    --ref_hoi_idx ${ref_idx[i]} \
    --ref_hoi_dir results/david_retarget_251028/hoi_data \
    --hand_info ${hand_info[i]}
done