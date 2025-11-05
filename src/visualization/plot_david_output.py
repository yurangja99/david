import numpy as np
import torch
from glob import glob
import argparse
import pickle
import sys
import os
import smplx
from utils.visualize import get_object_vertices, plot_3d_points
from imports.mdm.data_loaders.humanml.utils import paramUtil

sys.path.append(os.getcwd())

COMPATIBILITY_MATRIX = np.array([[1.0, 0.0, 0.0], [0.0, 0.0, 1.0], [0.0, -1.0, 0.0]], dtype=np.float64)

def print_dict(d, indent=0):
    d = dict(d)
    for k, v in d.items():
        if isinstance(v, (np.ndarray, torch.Tensor)):
            print('\t'*indent, k, type(v), v.shape)
        elif isinstance(v, dict):
            print('\t'*indent, k, type(v))
            print_dict(v, indent+1)
        else:
            print('\t'*indent, k, type(v), v)

def plot_david_output(
    dataset,
    category,
    human_motion_dir,
    object_motion_dir,
    output_motion_dir,
):
    smplx_pose_pths = sorted(glob(f"{human_motion_dir}/{dataset}/{category}/*/*/*/*.npz"))
    print('\n'.join(smplx_pose_pths))
    for ii, smplx_pose_pth in enumerate(smplx_pose_pths):
        object_motion_pth = smplx_pose_pth.replace(human_motion_dir, object_motion_dir).replace(".npz", ".pkl")
        output_pth = smplx_pose_pth.replace(human_motion_dir, output_motion_dir).replace(".npz", ".mp4")
        os.makedirs(os.path.dirname(output_pth), exist_ok=True)

        smplx_poses = np.load(smplx_pose_pth)
        with open(object_motion_pth, "rb") as handle:
            obj_motion = pickle.load(handle)

        frame_num = min(smplx_poses["poses"].shape[0], smplx_poses["trans"].shape[0])
        poses = smplx_poses["poses"][:frame_num]    # (T, 55*3)
        betas = smplx_poses["betas"]    # (10,)
        trans = smplx_poses["trans"][:frame_num]    # (T, 3)
        mocap_frame_rate = smplx_poses["mocap_frame_rate"]  # ()
        gender = smplx_poses["gender"]  # ()

        obj_R = obj_motion["R"][:frame_num] # (T, 3, 3)
        obj_t = obj_motion["t"][:frame_num, :, 0] # (T, 3)
        obj_shifted_t = obj_motion["shifted_t"] # (0,)
        average_scale = obj_motion["average_scale"] # 1.0

        # get joint positions by FK
        smplxmodel = smplx.create(model_path="imports/mdm/body_models/smplx/SMPLX_NEUTRAL.npz", model_type="smplx", num_pca_comps=45).cuda()
        global_smplxmodel_output = smplxmodel(
            betas=torch.from_numpy(betas.reshape((1, 10))).repeat((frame_num, 1)).to('cuda').float(),
            global_orient=torch.from_numpy(poses[:, :3]).to('cuda').float(),
            body_pose=torch.from_numpy(poses[:, 3 : 3 + 21*3]).to('cuda').float(),
            left_hand_pose=torch.zeros((frame_num, 45), device="cuda").float(),
            right_hand_pose=torch.zeros((frame_num, 45), device="cuda").float(),
            transl=torch.from_numpy(trans[:, :3]).to('cuda').float(),
            expression=torch.zeros((frame_num, 10), device="cuda").float(),
            jaw_pose=torch.zeros((frame_num, 3), device="cuda").float(),
            leye_pose=torch.zeros((frame_num, 3), device="cuda").float(),
            reye_pose=torch.zeros((frame_num, 3), device="cuda").float(),
            return_verts=True,
            return_full_pose=True,
        )
        global_vertices = global_smplxmodel_output.vertices.to(torch.float64).cpu().numpy()
        global_joints = global_smplxmodel_output.joints[:, :].detach().cpu().numpy()

        # min_y = global_vertices.min(axis=(0, 1))[1]
        # global_vertices[..., 1] -= min_y
        # global_joints[..., 1] -= min_y

        obj_v = get_object_vertices(
            torch.from_numpy(obj_t).to(dtype=torch.float32),
            torch.from_numpy(obj_R).to(dtype=torch.float32),
            h=0.15,
        ).detach().cpu().numpy()
        plot_3d_points(output_pth, paramUtil.t2m_kinematic_chain, global_joints, obj_v, title="DAViD Output", dataset="humanml", fps=20, show_joints=False)

if __name__ == "__main__":
    parser = argparse.ArgumentParser()

    # visualize configuration
    parser.add_argument("--dataset", type=str, default="ComAsset")
    parser.add_argument("--category", type=str, default="barbell")

    parser.add_argument("--human_motion_dir", type=str, default="results/inference/human_motion")
    parser.add_argument("--object_motion_dir", type=str, default="results/inference/object_motion")
    parser.add_argument("--output_motion_dir", type=str, default="results/inference/plot")

    parser.add_argument("--idx", type=int, default=0)

    args = parser.parse_args()

    plot_david_output(
        dataset=args.dataset,
        category=args.category,
        human_motion_dir=args.human_motion_dir,
        object_motion_dir=args.object_motion_dir,
        output_motion_dir=args.output_motion_dir,
    )