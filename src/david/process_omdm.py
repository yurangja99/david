import numpy as np
import smplx
import torch
import pickle
import random
import argparse
from glob import glob
import open3d as o3d
import os
import cv2
from constants.david import SELECTED_INDICES
from tqdm import tqdm
from utils.visualize import get_object_vertices, plot_3d_points
from imports.mdm.data_loaders.humanml.utils import paramUtil
SKELETON = paramUtil.t2m_kinematic_chain + paramUtil.t2m_left_hand_chain + paramUtil.t2m_right_hand_chain

COMPATIBILITY_MATRIX = np.array([[1.0, 0.0, 0.0], [0.0, 0.0, 1.0], [0.0, -1.0, 0.0]])

HAND_INFO = dict(
    LEFT=1,
    RIGHT=2,
    NO=0
)

def prepare_dataset(
    dataset,
    category,
    hmr_dir,
    hoi_dir,
    obj_dir,
    hoi_data_dir,
    joint_dir,
    hand_info,
    skip_done
):
    COMPATIBILITY_MATRIX = np.array([[1.0, 0.0, 0.0], [0.0, 0.0, 1.0], [0.0, -1.0, 0.0]])

    
    if hand_info == 1:
        hand_category = f"left_{category}"
    elif hand_info == 2:
        hand_category = f"right_{category}"
    elif hand_info == 0:
        hand_category = f"{category}"

    with open("constants/sampled_human_indices.pkl", "rb") as handle:
        human_sampled_indices = pickle.load(handle)

    joint_pths = sorted(glob(f"{joint_dir}/{dataset}/{category}/*/*/*.npy"))
    obj_pths = sorted(glob(f"{obj_dir}/{dataset}/{category}/*/*/*.npz"))
    assert len(joint_pths) == len(obj_pths)

    pts = []
    RT = []
    body_poses = []
    for joint_pth, obj_pth in zip(joint_pths, tqdm(obj_pths)):
        joints = np.load(joint_pth)
        obj = np.load(obj_pth)
        
        # object info
        obj_R = obj['obj_rot']  # [T, 3, 3]
        obj_t = obj['obj_trans']    # [T, 3]
        poses = obj['poses']    # [T, 72]
        trans = obj['trans']    # [T, 3]
        betas = obj['betas']    # [T, 10]
        motion_global = {
            'poses': poses, # [T, 72]
            'trans': trans, # [T, 3]
            'betas': betas[0:1],   # [1, 10]
        }

        frame_num = motion_global["poses"].shape[0]
        smplxmodel = smplx.create(model_path="imports/mdm/body_models/smplx/SMPLX_NEUTRAL.npz", model_type="smplx", num_pca_comps=45).cuda()
        global_smplxmodel_output = smplxmodel(
            betas=torch.zeros_like(torch.from_numpy(motion_global["betas"].reshape((1, 10))).repeat((frame_num, 1))).to('cuda').float(),
            global_orient=torch.from_numpy(motion_global["poses"][:, :3]).to('cuda').float(),
            body_pose=torch.from_numpy(motion_global["poses"][:, 3 : 3 + 21*3]).to('cuda').float(),
            left_hand_pose=torch.zeros((frame_num, 45), device="cuda").float(),
            right_hand_pose=torch.zeros((frame_num, 45), device="cuda").float(),
            transl=torch.from_numpy(motion_global["trans"][:, :3]).to('cuda').float(),
            expression=torch.zeros((frame_num, 10), device="cuda").float(),
            jaw_pose=torch.zeros((frame_num, 3), device="cuda").float(),
            leye_pose=torch.zeros((frame_num, 3), device="cuda").float(),
            reye_pose=torch.zeros((frame_num, 3), device="cuda").float(),
            return_verts=True,
            return_full_pose=True,
        )
        global_vertices = global_smplxmodel_output.vertices.to(torch.float64).cpu().numpy()
        global_joints = global_smplxmodel_output.joints[:, :].detach().cpu().numpy()

        # human joints/vertices calibration
        min_y = global_vertices.min(axis=(0, 1))[1]
        global_vertices[..., 1] -= min_y
        global_joints[..., 1] -= min_y

        # move object for new wrist positions
        hand_offset = global_joints[..., 20:22, :] - joints[..., 20:22, :]  # (T, 2, 3)
        hand_offset = hand_offset.mean(1)   # (T, 3)
        obj_t_old = obj_t.copy()
        obj_t = obj_t + hand_offset
        
        video_dir = f"{hoi_data_dir}/{dataset}/{hand_category}/animations/"
        os.makedirs(video_dir, exist_ok=True)
        seq_name = os.path.basename(os.path.dirname(obj_pth))
        object_name = ''.join(ch for ch in category.split("_")[0].lower() if ch.isalnum())
        obj_v = get_object_vertices(
            torch.from_numpy(obj_t).to(dtype=torch.float32), 
            torch.from_numpy(obj_R).to(dtype=torch.float32), 
            object=object_name
        ).detach().cpu().numpy()
        obj_v_old = get_object_vertices(
            torch.from_numpy(obj_t_old).to(dtype=torch.float32), 
            torch.from_numpy(obj_R).to(dtype=torch.float32), 
            object=object_name
        ).detach().cpu().numpy()
        # plot_3d_points(os.path.join(video_dir, f"{seq_name}_joints_gt.mp4"), SKELETON, joints, object_name, obj_v_old[:joints.shape[0]], title="Joints(GT)", dataset="humanml", fps=30, show_joints=False)
        plot_3d_points(os.path.join(video_dir, f"{seq_name}_joints.mp4"), SKELETON, global_joints, object_name, obj_v, title="Joints", dataset="humanml", fps=30, show_joints=False)
        plot_3d_points(os.path.join(video_dir, f"{seq_name}_vertices_all.mp4"), [], global_vertices[:, ::50], object_name, obj_v, title="Vertices(All)", dataset="humanml", fps=30, show_joints=True)
        # plot_3d_points(os.path.join(video_dir, f"{seq_name}_vertices_selected.mp4"), [], global_vertices[:, human_sampled_indices], object_name, obj_v, title="Vertices(Hands)", dataset="humanml", fps=30, show_joints=True)

        for frame_vertices, frame_joints, human_euler, frame_obj_R, frame_obj_t, frame_pose in zip(global_vertices, global_joints, motion_global["poses"][:, :3], obj_R, obj_t, motion_global["poses"][:, 3 : 3 + 21*3]):
            sampled_vertices = frame_vertices[human_sampled_indices]

            human_R, _ = cv2.Rodrigues(human_euler)
            # mean = np.mean(sampled_vertices, axis=0)
            mean = np.array(frame_joints[0])    # pelvis joint
            normalized_vertices = sampled_vertices - mean.reshape((1, 3)) # 1024 x 3
            fully_normalized_vertices = normalized_vertices @ (human_R.T).T
            normalized_all_vertices = frame_vertices - mean.reshape((1, 3))
            fully_normalized_all_vertices = normalized_all_vertices @ (human_R.T).T

            compatible_frame_obj_R = human_R.T @ frame_obj_R # 3 x 3. Already transformed, so no transformation
            compatible_frame_obj_t = human_R.T @ (frame_obj_t.reshape((3, 1)) - mean.reshape((3, 1))) # 3 x 1. Already transformed, so no transformation

            compatible_frame_obj_Rt = np.concatenate((compatible_frame_obj_R, compatible_frame_obj_t), axis=1) # 3 x 4

            pts.append(fully_normalized_vertices)
            RT.append(compatible_frame_obj_Rt)
            body_poses.append(frame_pose.reshape((21, 3)))
    
    pts = np.array(pts) # N x 1024 x 3
    RT = np.array(RT) # N x 3 x 4
    body_poses = np.array(body_poses) # N x 63 x 3

    to_save = dict(
        points=pts,
        transform=RT,
        poses=body_poses,
    )
    
    save_pth = f"{hoi_data_dir}/{dataset}/{hand_category}/RT.pkl"
    os.makedirs("/".join(save_pth.split("/")[:-1]), exist_ok=True)

    with open(save_pth, 'wb') as handle:
        pickle.dump(to_save, handle, protocol=pickle.HIGHEST_PROTOCOL)

if __name__ == "__main__":
    parser = argparse.ArgumentParser()

    parser.add_argument("--dataset", type=str, default="FullBodyManip")
    parser.add_argument("--category", type=str, default="largetable")
    parser.add_argument("--hmr_dir", type=str, default="results/generation/hmr")
    parser.add_argument("--hoi_dir", type=str, default="results/generation/hoi")
    parser.add_argument("--obj_dir", type=str, default="results/david/obj_data")
    parser.add_argument("--hoi_data_dir", type=str, default="results/david/hoi_data")
    parser.add_argument("--joint_dir", type=str, default="results/david/pose_data")
    parser.add_argument("--hand_info", type=int, default=0)

    parser.add_argument("--skip_done", action="store_true")
    parser.add_argument("--verbose", action="store_true")

    args = parser.parse_args()
    prepare_dataset(
        dataset=args.dataset,
        category=args.category,
        hmr_dir=args.hmr_dir,
        hoi_dir=args.hoi_dir,
        obj_dir=args.obj_dir,
        hoi_data_dir=args.hoi_data_dir,
        joint_dir=args.joint_dir,
        hand_info=args.hand_info,
        skip_done=args.skip_done
    )
