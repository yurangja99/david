# <p align="center"> DAViD: Modeling Dynamic Affordance of 3D Objects Using Pre-trained Video Diffusion Models (ICCV 2025)</p>

## [Project Page](https://snuvclab.github.io/david/) &nbsp;|&nbsp; [Paper](https://arxiv.org/pdf/2501.08333) 

![demo.png](./assets/teaser.png)

This is the official code for the paper "DAViD: Modeling Dynamic Affordance of 3D Objects Using Pre-trained Video Diffusion Models".

## Installation

To setup the environment for running DAViD, please refer to the instructions provided <a href="INSTALL.md">here</a>.


## Datasets
```
david
  ├ animation
  ├ ...
  ├ data
  │  ├ InterAct
  │  │  ├ sub2_largetable_003.pt
  │  │  ├ sub2_largetable_005.pt
  │  │  └ ...
  │  ├ InterActObjects
  │  │  ├ objects
  │  │  │  ├ clothesstand
  │  │  │  │  ├ clothesstand.obj
  │  │  │  │  └ sample_points.npy
  │  │  │  └ floorlamp
  │  │  │     ├ floorlamp.obj
  │  │  │     └ sample_points.npy
  │  │  ├ clothesstand.urdf
  │  │  ├ floorlamp.urdf
  │  │  └ ...
  │  └ omomo_text_anno_json_data
  │     ├ sub2_largetable_003.json
  │     ├ sub2_largetable_005.json
  │     └ ...
  ├ ...    
  └ imports
     ├ ...
     ├ mdm
     │  ├ body_models
     │  │  ├ dmpls
     │  │  ├ smpl
     │  │  ├ smplh
     │  │  └ smplx
     │  ├ checkpoints
     │  │  ├ humanml_enc_512_50steps
     │  │  │  ├ args.json
     │  │  │  ├ model000750000.pt
     │  │  │  └ opt000750000.pt
     │  │  └ t2m
     │  └ dataset
     │     ├ HumanML3D
     │     ├ humanml_opt.txt
     │     ├ kit_opt.txt
     │     └ t2m_train.npy
     └ ...
```

## Quick Start

### 2D HOI Image Generation

To generate 2D HOI Images of given 3D object (in this case, barbell), use following command.

```shell
bash scripts/generate_2d_hoi_images.sh --dataset "ComAsset" --category "barbell" --device 0 --skip_done
```

<table>
  <tr>
    <td align="center" width="33%">
      <img src="assets/render.png"><br>
      <sub>Rendered Image</sub>
    </td>
    <td align="center" width="33%">
      <img src="assets/canny.png"><br>
      <sub>Canny Edges</sub>
    </td>
    <td align="center" width="33%">
      <img src="assets/2dhoi.png"><br>
      <sub>2D HOI Image</sub>
    </td>
  </tr>
</table>

### Image-to-Video

We leverage commercial image-to-video diffusion model [Kling AI](https://www.klingai.com/) to make 2D HOI videos from 2D HOI images.
Specifically, we use [imgur](https://imgur.com/) and [PiAPI](https://piapi.ai/docs) for uploading image and calling API for Kling AI. Check out `scripts/videos/get.sh`, `scripts/videos/post_i2v.sh` and setup your `X-API-key` of your PiAPI account. Also check out `constants/videos.py` and setup your client id of your imgur account. Note that you need paid version of Kling AI for directly follow our setting.

```shell
CUDA_VISIBLE_DEVICES=0 python src/generation/generate_videos.py --dataset "ComAsset" --category "barbell" --skip_done
```

Otherwise, you can use open-source image-to-video models such as [Wan2.1](https://github.com/Wan-Video/Wan2.1), but we haven't tested it yet.

### 4D HOI Sample Generation

To generate 4D HOI Samples from the generated 2D HOI Images (of the given 3D object, barbell), use following command.

```shell
bash scripts/generate_4d_hoi_samples.sh --dataset "ComAsset" --category "barbell" --device 0 --skip_done
```

<table>
  <tr>
    <td align="center" width="33%">
      <img src="assets/2dhoi.png"><br>
      <sub>2D HOI Image</sub>
    </td>
    <td align="center" width="33%">
      <img src="assets/2dhoivid.gif"><br>
      <sub>2D HOI Video</sub>
    </td>
    <td align="center" width="33%">
      <img src="assets/4dhoi_incam.gif"><br>
      <sub>4D HOI Sample (Camera View)</sub>
    </td>
  </tr>
</table>


### Visualization

To visualize generated 4D HOI Samples, use following command. 

```shell
blenderproc debug src/visualization/visualize_4d_hoi_sample.py --dataset "ComAsset" --category "barbell" --idx 0
```

![4dhoi.gif](./assets/4dhoi.gif)

### Preprocess FullBodyManip Dataset

- [TODO] `train_diffusion_manip_window_120_cano_joints24.p` is preprocessed by window size `120`, so motions (max length `442`) are sliced into context-lost sub-motions.

```shell
python src/david/preprocess_fullbodymanip.py \
  --inputs data/FullBodyManip/train_diffusion_manip_seq_joints24.p data/FullBodyManip/cano_train_diffusion_manip_window_120_joints24.p data/FullBodyManip/test_diffusion_manip_seq_joints24.p data/FullBodyManip/cano_test_diffusion_manip_window_120_joints24.p \
  --pose_dir results/david/pose_data \
  --dataset FullBodyManip \
  --category largetable

python src/david/preprocess_interact.py \
  --input_dir data/InterAct \
  --pose_dir results/david/pose_data \
  --dataset FullBodyManip \
  --category largetable
```

### Train LoRA for MDM

To train LoRA for MDM (of the given 3D object, barbell), use following command.

```shell
bash scripts/train_lora.sh --dataset "ComAsset" --category "barbell" --device 0
bash scripts/train_lora.sh --dataset "FullBodyManip" --category "largetable" --device 0
CUDA_VISIBLE_DEVICES=0 python src/david/process_mdm.py --dataset FullBodyManip --category largetable
CUDA_VISIBLE_DEVICES=0 python src/david/train_lora.py --david_dataset FullBodyManip --category largetable --num_steps 10000
```

### Train Object Motion Diffusion Model

To train Object Motion Diffusion Model (of the given 3D object, barbell), use following command.

```shell
bash scripts/train_omdm.sh --dataset "ComAsset" --category "barbell" --device 0
```

```shell
# OMOMO largetable
bash scripts/train_omdm.sh --dataset "FullBodyManip" --category "largetable" --device 0
```

```shell
# OMOMO largetable
CUDA_VISIBLE_DEVICES=0 python src/david/process_omdm.py --dataset FullBodyManip --category largetable
CUDA_VISIBLE_DEVICES=0 python src/david/train_omdm.py --dataset FullBodyManip --category largetable --n_epochs 100000
```

### Sample Human Motion (Inference)

```shell
bash scripts/generate_human_motion.sh --max_seed 5 --dataset "ComAsset" --category "barbell" --device 0
```

```shell
# OMOMO largetable
bash scripts/generate_human_motion.sh --max_seed 5 --dataset "FullBodyManip" --category "largetable" --device 0
```

```shell
# OMOMO largetable (various settings)
prompts=(
  "A person runs forward, largetable"
  "A person runs backward, largetable"
  "A person side steps, largetable"
  "A person jumps forward, largetable"
)
lora_weight=( 0.9 0.7 0.7 0.7 )
for epoch in $(seq 2000 2000 10000); do
    for i in "${!prompts[@]}"; do
        p="${prompts[i]}"
        w="${lora_weight[i]}"
        for seed in $(seq 0 5); do
            CUDA_VISIBLE_DEVICES=$device python src/david/inference_mdm.py \
                --david_dataset "FullBodyManip" \
                --category "largetable" \
                --text_prompt "$p" \
                --seed $seed \
                --num_samples 1 \
                --num_repetitions 1 \
                --lora_weight $w \
                --inference_epoch $epoch
        done
    done
done
CUDA_VISIBLE_DEVICES=0 python src/david/joint2smplx.py --dataset "FullBodyManip" --category "largetable"
```

### Sample Object Motion (Inference)

```shell
bash scripts/generate_object_motion.sh --dataset "ComAsset" --category "barbell" --device 0
```

```shell
# OMOMO largetable
bash scripts/generate_object_motion.sh --dataset "FullBodyManip" --category "largetable" --device 0
```

```shell
# OMOMO largetable
CUDA_VISIBLE_DEVICES=0 python src/david/inference_omdm.py --dataset "FullBodyManip" --category "largetable" --inference_contact_threshold 0.05 --inference_epoch 100000
```

### Visualize DAViD Output (Inference)

```shell
blenderproc debug src/visualization/visualize_david_output.py --custom-blender-path ../../datasets/blender-3.6.3-linux-x64 --dataset "FullBodyManip" --category "largetable" --idx 0
```

## Regarding Code Release
- We are keep updating the code (including dataset and environment setup)!
- [2025/07/10] Initial skeleton code release!


## Citation
```bibtex
@misc{david,
      title={DAViD: Modeling Dynamic Affordance of 3D Objects Using Pre-trained Video Diffusion Models}, 
      author={Hyeonwoo Kim and Sangwon Baik and Hanbyul Joo},
      year={2025},
      eprint={2501.08333},
      archivePrefix={arXiv},
      primaryClass={cs.CV},
      url={https://arxiv.org/abs/2501.08333}, 
}
```