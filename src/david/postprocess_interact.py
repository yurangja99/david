import os, argparse, joblib, numpy as np
from glob import glob
from pathlib import Path
import json
import torch
import pickle
import smplx
from imports.mdm.visualize import vis_utils
from imports.mdm.data_loaders.humanml.utils import paramUtil
from utils.visualize import plot_3d_points, get_object_vertices
from pytorch3d.transforms import (
    quaternion_to_matrix, 
    matrix_to_axis_angle, 
    axis_angle_to_quaternion, 
    axis_angle_to_matrix,
    quaternion_multiply,
    quaternion_invert,
    matrix_to_quaternion,
)

SMPLX_POSE_JOINTS = [
    # 'pelvis', 
    'left_hip', 'right_hip', 'spine1', 
    'left_knee', 'right_knee', 'spine2', 
    'left_ankle', 'right_ankle', 'spine3', 
    'left_foot', 'right_foot', 'neck', 
    'left_collar', 'right_collar', 'head', 
    'left_shoulder', 'right_shoulder', 
    'left_elbow', 'right_elbow', 
    'left_wrist', 'right_wrist', 
    # 'jaw', 'left_eye_smplhf', 'right_eye_smplhf', 
    'left_index1', 'left_index2', 'left_index3', 
    'left_middle1', 'left_middle2', 'left_middle3', 
    'left_pinky1', 'left_pinky2', 'left_pinky3', 
    'left_ring1', 'left_ring2', 'left_ring3', 
    'left_thumb1', 'left_thumb2', 'left_thumb3', 
    'right_index1', 'right_index2', 'right_index3', 
    'right_middle1', 'right_middle2', 'right_middle3', 
    'right_pinky1', 'right_pinky2', 'right_pinky3', 
    'right_ring1', 'right_ring2', 'right_ring3', 
    'right_thumb1', 'right_thumb2', 'right_thumb3'
]
OMOMO_POSE_JOINTS = [
    'L_Hip', 'L_Knee', 'L_Ankle', 'L_Toe',
    'R_Hip', 'R_Knee', 'R_Ankle', 'R_Toe',
    'Torso', 'Spine', 'Chest', 'Neck', 'Head',
    'L_Thorax', 'L_Shoulder', 'L_Elbow', 'L_Wrist', 
    'L_Index1', 'L_Index2', 'L_Index3', 
    'L_Middle1', 'L_Middle2', 'L_Middle3', 
    'L_Pinky1', 'L_Pinky2', 'L_Pinky3', 
    'L_Ring1', 'L_Ring2', 'L_Ring3', 
    'L_Thumb1', 'L_Thumb2', 'L_Thumb3',
    'R_Thorax', 'R_Shoulder', 'R_Elbow', 'R_Wrist', 
    'R_Index1', 'R_Index2', 'R_Index3', 
    'R_Middle1', 'R_Middle2', 'R_Middle3', 
    'R_Pinky1', 'R_Pinky2', 'R_Pinky3', 
    'R_Ring1', 'R_Ring2', 'R_Ring3', 
    'R_Thumb1', 'R_Thumb2', 'R_Thumb3'
]
SMPLX_JOINTS = [
    'pelvis', 
    'left_hip', 'right_hip', 'spine1', 
    'left_knee', 'right_knee', 'spine2', 
    'left_ankle', 'right_ankle', 'spine3', 
    'left_foot', 'right_foot', 'neck', 
    'left_collar', 'right_collar', 'head', 
    'left_shoulder', 'right_shoulder', 
    'left_elbow', 'right_elbow', 
    'left_wrist', 'right_wrist', 
    'jaw', 'left_eye_smplhf', 'right_eye_smplhf', 
    'left_index1', 'left_index2', 'left_index3', 
    'left_middle1', 'left_middle2', 'left_middle3', 
    'left_pinky1', 'left_pinky2', 'left_pinky3', 
    'left_ring1', 'left_ring2', 'left_ring3', 
    'left_thumb1', 'left_thumb2', 'left_thumb3', 
    'right_index1', 'right_index2', 'right_index3', 
    'right_middle1', 'right_middle2', 'right_middle3', 
    'right_pinky1', 'right_pinky2', 'right_pinky3', 
    'right_ring1', 'right_ring2', 'right_ring3', 
    'right_thumb1', 'right_thumb2', 'right_thumb3'
]

OMOMO_JOINTS = [
    'Pelvis',
    'L_Hip', 'L_Knee', 'L_Ankle', 'L_Toe',
    'R_Hip', 'R_Knee', 'R_Ankle', 'R_Toe',
    'Torso', 'Spine', 'Chest', 'Neck', 'Head',
    'L_Thorax', 'L_Shoulder', 'L_Elbow', 'L_Wrist', 
    'L_Index1', 'L_Index2', 'L_Index3', 
    'L_Middle1', 'L_Middle2', 'L_Middle3', 
    'L_Pinky1', 'L_Pinky2', 'L_Pinky3', 
    'L_Ring1', 'L_Ring2', 'L_Ring3', 
    'L_Thumb1', 'L_Thumb2', 'L_Thumb3',
    'R_Thorax', 'R_Shoulder', 'R_Elbow', 'R_Wrist', 
    'R_Index1', 'R_Index2', 'R_Index3', 
    'R_Middle1', 'R_Middle2', 'R_Middle3', 
    'R_Pinky1', 'R_Pinky2', 'R_Pinky3', 
    'R_Ring1', 'R_Ring2', 'R_Ring3', 
    'R_Thumb1', 'R_Thumb2', 'R_Thumb3'
]
OMOMO_TO_SMPLX = {
    'Pelvis': 'pelvis',
    'L_Hip': 'left_hip', 'L_Knee': 'left_knee', 'L_Ankle': 'left_ankle', 'L_Toe': 'left_foot',
    'R_Hip': 'right_hip', 'R_Knee': 'right_knee', 'R_Ankle': 'right_ankle', 'R_Toe': 'right_foot',
    'Torso': 'spine1', 'Spine': 'spine2', 'Chest': 'spine3', 'Neck': 'neck', 'Head': 'head',
    'L_Thorax': 'left_collar', 'L_Shoulder': 'left_shoulder', 'L_Elbow': 'left_elbow', 'L_Wrist': 'left_wrist', 
    'L_Index1': 'left_index1', 'L_Index2': 'left_index2', 'L_Index3': 'left_index3', 
    'L_Middle1': 'left_middle1', 'L_Middle2': 'left_middle2', 'L_Middle3': 'left_middle3', 
    'L_Pinky1': 'left_pinky1', 'L_Pinky2': 'left_pinky2', 'L_Pinky3': 'left_pinky3', 
    'L_Ring1': 'left_ring1', 'L_Ring2': 'left_ring2', 'L_Ring3': 'left_ring3', 
    'L_Thumb1': 'left_thumb1', 'L_Thumb2': 'left_thumb2', 'L_Thumb3': 'left_thumb3',
    'R_Thorax': 'right_collar', 'R_Shoulder': 'right_shoulder', 'R_Elbow': 'right_elbow', 'R_Wrist': 'right_wrist', 
    'R_Index1': 'right_index1', 'R_Index2': 'right_index2', 'R_Index3': 'right_index3', 
    'R_Middle1': 'right_middle1', 'R_Middle2': 'right_middle2', 'R_Middle3': 'right_middle3', 
    'R_Pinky1': 'right_pinky1', 'R_Pinky2': 'right_pinky2', 'R_Pinky3': 'right_pinky3', 
    'R_Ring1': 'right_ring1', 'R_Ring2': 'right_ring2', 'R_Ring3': 'right_ring3', 
    'R_Thumb1': 'right_thumb1', 'R_Thumb2': 'right_thumb2', 'R_Thumb3': 'right_thumb3'
}

T_OMOMO_TO_SMPLX = torch.tensor([
    [ 0,  1,  0],
    [ 0,  0,  1],
    [ 1,  0,  0],
], dtype=torch.float32)
T_SMPLX_TO_OMOMO = torch.tensor([
    [ 0,  0,  1],
    [ 1,  0,  0],
    [ 0,  1,  0],
], dtype=torch.float32)

def transpose_axis_angle_intermimic_to_smplx(axis_angle: torch.Tensor):
    quat = axis_angle_to_quaternion(axis_angle)
    T_quat = matrix_to_quaternion(T_OMOMO_TO_SMPLX.unsqueeze(0))
    T_quat = T_quat.expand(*quat.shape[:-1], 4)
    T_inv_quat = quaternion_invert(T_quat)
    smplx_quat = quaternion_multiply(quaternion_multiply(T_quat, quat), T_inv_quat)
    return matrix_to_axis_angle(quaternion_to_matrix(smplx_quat))

def transpose_transl_intermimic_to_smplx(transl: torch.Tensor):
    return torch.matmul(transl, T_OMOMO_TO_SMPLX.T)

def transpose_axis_angle_smplx_to_intermimic(axis_angle: torch.Tensor):
    quat = axis_angle_to_quaternion(axis_angle)
    T_quat = matrix_to_quaternion(T_SMPLX_TO_OMOMO.unsqueeze(0))
    T_quat = T_quat.expand(*quat.shape[:-1], 4)
    T_inv_quat = quaternion_invert(T_quat)
    smplx_quat = quaternion_multiply(quaternion_multiply(T_quat, quat), T_inv_quat)
    return matrix_to_axis_angle(quaternion_to_matrix(smplx_quat))

def transpose_transl_smplx_to_intermimic(transl: torch.Tensor):
    return torch.matmul(transl, T_SMPLX_TO_OMOMO.T)

def get_absolute_joint_rotations(full_pose: torch.Tensor, smplx_model) -> torch.Tensor:
    T, J, _ = full_pose.shape
    device = full_pose.device

    # Convert to rotation matrices
    rot_mats = axis_angle_to_matrix(full_pose)  # [T, 55, 3, 3]

    # Output tensor for absolute rotations
    abs_rot = torch.zeros_like(rot_mats)  # [T, 55, 3, 3]

    parents = smplx_model.parents.clone().detach()  # list or tensor of shape [55], -1 means root

    for j in range(J):
        parent = parents[j]
        if parent == -1:
            abs_rot[:, j] = rot_mats[:, j]  # root joint
        else:
            abs_rot[:, j] = abs_rot[:, parent] @ rot_mats[:, j]

    return matrix_to_axis_angle(abs_rot)

def normalize(s: str) -> str:
    # 대소문자/언더스코어/하이픈/띄어쓰기 차이를 없애기 위한 정규화
    return ''.join(ch for ch in s.lower() if ch.isalnum())

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--dataset", default="FullBodyManip")
    ap.add_argument("--category", default="largetable")
    ap.add_argument("--output_dir", default="data/InterAct_new", help="InterAct 데이터셋 폴더 주소")
    ap.add_argument("--human_motion_dir", type=str, default="results/inference/human_motion")
    ap.add_argument("--object_motion_dir", type=str, default="results/inference/object_motion")
    args = ap.parse_args()

    total_saved = 0
    total_seen = 0

    human_motion_pths = sorted(glob(f"{args.human_motion_dir}/{args.dataset}/{args.category}/*/*/*/*.npz"))
    obj_motion_pths = [human_pth.replace(args.human_motion_dir, args.object_motion_dir).replace(".npz", ".pkl") for human_pth in human_motion_pths]
    print("\n".join(human_motion_pths))
    os.makedirs(args.output_dir, exist_ok=True)
    for idx, (human_pth, obj_pth) in enumerate(zip(human_motion_pths, obj_motion_pths)):
        total_seen += 1

        # read human/obj motions
        human_motion = np.load(human_pth)
        with open(obj_pth, "rb") as f:
            obj_motion = pickle.load(f)
        
        # human_motion
        # poses: (T, 55*3)
        # betas: (10,)
        # trans: (T, 3)
        # mocap_frame_rate: 30
        # gender: "neutral"
        
        # obj_motion
        # R: (T, 3, 3)
        # t: (T, 3, 1)
        # shifted_t: []
        # average_scale: 1.0

        # initialize data
        frame_num = min(human_motion["poses"].shape[0], obj_motion['R'].shape[0])
        data = torch.zeros((frame_num, 591))

        # forward kinematics
        smplxmodel = smplx.create(model_path="imports/mdm/body_models/smplx/SMPLX_NEUTRAL.npz", model_type="smplx", num_pca_comps=45)
        global_smplxmodel_output = smplxmodel(
            betas=torch.from_numpy(human_motion["betas"].reshape((1, 10))).repeat((frame_num, 1)).float(),
            global_orient=torch.from_numpy(human_motion["poses"][:frame_num, :3]).float(),
            body_pose=torch.from_numpy(human_motion["poses"][:frame_num, 3:3+21*3]).float(),
            left_hand_pose=torch.zeros((frame_num, 45)).float(),
            right_hand_pose=torch.zeros((frame_num, 45)).float(),
            transl=torch.from_numpy(human_motion["trans"][:frame_num, :3]).float(),
            expression=torch.zeros((frame_num, 10)).float(),
            jaw_pose=torch.zeros((frame_num, 3)).float(),
            leye_pose=torch.zeros((frame_num, 3)).float(),
            reye_pose=torch.zeros((frame_num, 3)).float(),
            return_verts=True,
            return_full_pose=True,
        )
        global_vertices = global_smplxmodel_output.vertices.to(torch.float32).cpu() # (T, 10475, 3)
        global_joints = global_smplxmodel_output.joints.to(torch.float32).cpu()[:, :55, :] # (T, 127, 3)
        local_joint_rot = global_smplxmodel_output.full_pose[:, :55*3].reshape(frame_num, 55, 3) # (T, 55, 3)
        global_joint_rot = get_absolute_joint_rotations(local_joint_rot, smplxmodel)

        min_y = global_vertices[..., 1].min()
        global_vertices[..., 1] -= min_y
        global_joints[..., 1] -= min_y

        # SMPLX -> OMOMO
        omomo_body_pos = torch.zeros((frame_num, 52, 3), dtype=torch.float32)
        omomo_body_rot = torch.zeros((frame_num, 52, 3), dtype=torch.float32)
        omomo_dof_pos = torch.zeros((frame_num, 51, 3), dtype=torch.float32)
        for omomo_joint, smplx_joint in OMOMO_TO_SMPLX.items():
            if omomo_joint in OMOMO_JOINTS and smplx_joint in SMPLX_JOINTS:
                oidx = OMOMO_JOINTS.index(omomo_joint)
                sidx = SMPLX_JOINTS.index(smplx_joint)
                if 0 <= sidx < 22 or 25 <= sidx < 40 or 40 <= sidx < 55:
                    omomo_body_pos[:, oidx] = transpose_transl_smplx_to_intermimic(global_joints[:, sidx])
                    omomo_body_rot[:, oidx] = transpose_axis_angle_smplx_to_intermimic(global_joint_rot[:, sidx])
            if omomo_joint in OMOMO_POSE_JOINTS and smplx_joint in SMPLX_POSE_JOINTS:
                oidx = OMOMO_POSE_JOINTS.index(omomo_joint)
                sidx = SMPLX_POSE_JOINTS.index(smplx_joint)
                if 0 <= sidx < 21:
                    omomo_dof_pos[:, oidx] = transpose_axis_angle_smplx_to_intermimic(
                        torch.from_numpy(human_motion["poses"][:frame_num, 3+sidx*3:3+(sidx+1)*3]).float()
                    )

        # root_pos: data[:, 0:3]
        data[:, 0:3] = omomo_body_pos[:, 0, :]

        # root_rot: data[:, 3:7] (xyzw quat)
        data[:, [6, 3, 4, 5]] = axis_angle_to_quaternion(transpose_axis_angle_smplx_to_intermimic(
            torch.from_numpy(human_motion["poses"][:frame_num, :3]).float()
        ))
        
        # dof_pos: data[:, 9:9+51*3]
        data[:, 9:9+51*3] = omomo_dof_pos.reshape(frame_num, 51*3)
        
        # body_pos: data[:, 162:162+52*3]
        data[:, 162:162+52*3] = omomo_body_pos.reshape(frame_num, 52*3)
        
        # obj_pos: data[:, 318:321]
        data[:, 318:321] = transpose_transl_smplx_to_intermimic(
            torch.from_numpy(obj_motion["t"][:frame_num, :, 0]).float()
        )
        
        # obj_rot: data[:, 321:325] (xyzw quat)
        COMPATIBILITY_MATRIX = np.array([[1.0, 0.0, 0.0], [0.0, 0.0, 1.0], [0.0, -1.0, 0.0]])
        ROT_OFS = np.array([[-1.0, 0.0, 0.0], [0.0, 0.0, 1.0], [0.0, 1.0, 0.0]])
        ROT_OFS_INV = np.array([[-1.0, 0.0, 0.0], [0.0, 0.0, 1.0], [0.0, 1.0, 0.0]])
        data[:, [324, 321, 322, 323]] = matrix_to_quaternion(
            torch.from_numpy(COMPATIBILITY_MATRIX.T @ obj_motion["R"][:frame_num] @ COMPATIBILITY_MATRIX @ ROT_OFS_INV).float()
        )
        
        # contact_obj: data[:, 330:331]
        data[:, 330:331] = 1.0
        
        # contact_human: data[:, 331:331+52]
        data[:, 331+17:331+17+16] = 1.0
        data[:, 331+36:331+36+16] = 1.0
        
        # body_rot: data[:, 383:383+52*4] (xyzw quat)
        data[:, 383:383+52*4] = axis_angle_to_quaternion(omomo_body_rot)[..., [1, 2, 3, 0]].reshape(frame_num, 52*4)
        
        # not used: data[:, 7:9], data[:, 325:330]
        data[:, 7:9] = 0.0
        data[:, 325:330] = 0.0

        data_pth = os.path.join(args.output_dir, f"sub0_{args.category}_{idx:03d}.pt")
        torch.save(data, data_pth)
        total_saved += 1
        if total_saved % 50 == 0:
            print(f"[info] saved {total_saved} sequences ...")
    
    print(f"[done] seen={total_seen}, saved={total_saved}")

if __name__ == "__main__":
    main()
