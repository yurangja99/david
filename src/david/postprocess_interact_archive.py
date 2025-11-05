import os, argparse, joblib, numpy as np
from tqdm import tqdm
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

def same_sign_quaternions(q, q_next):
    """
    q, q_next: (..., 4) shape 쿼터니언 텐서
    부호가 반대인지 아닌지 판단하고, 필요 시 부호를 맞춰 반환
    """
    # 가장 큰 절대값 성분의 인덱스 찾기
    idx = torch.argmax(torch.abs(q))
    sign_q = torch.sign(q[idx])
    sign_q_next = torch.sign(q_next[idx])
    
    same_sign = (sign_q == sign_q_next)
    return 1.0 if same_sign else -1.0

def get_reference(path, crop_start, blending_frame):
    # load reference (30FPS)
    ref_data = torch.load(path)
    ref_mask = ref_data[:, 330] > 0.5
    ref_start_idx = int(torch.nonzero(ref_mask.flatten(), as_tuple=True)[0][0]) + crop_start
    ref_data = ref_data[ref_start_idx:ref_start_idx+blending_frame]

    # transl, global orientation, dof_pos
    ref_transl = ref_data[:, 0:3]
    ref_root_quat = ref_data[:, [6, 3, 4, 5]]   # wxyz
    ref_body_pos = ref_data[:, 162:162+52*3].reshape(-1, 52, 3)
    ref_body_rot = ref_data[:, 331+52:331+52+52*4].reshape(-1, 52, 4)
    ref_dof = ref_data[:, 9:9+153].reshape(-1, 51, 3)
    ref_obj_pos = ref_data[:, 318:321]
    ref_obj_quat = ref_data[:, [324, 321, 322, 323]]    # wxyz
    ref_contact_obj = ref_data[:, 330]
    ref_contact_human = ref_data[:, 331:331+52]

    # OMOMO -> SMPLX
    smplx_body_pose = torch.zeros((blending_frame, 21, 3), dtype=torch.float32)
    smplx_left_pose = torch.zeros((blending_frame, 15, 3), dtype=torch.float32)
    smplx_right_pose = torch.zeros((blending_frame, 15, 3), dtype=torch.float32)
    for omomo_joint, smplx_joint in OMOMO_TO_SMPLX.items():
        if omomo_joint not in OMOMO_POSE_JOINTS: continue
        if smplx_joint not in SMPLX_POSE_JOINTS: continue
        omomo_idx = OMOMO_POSE_JOINTS.index(omomo_joint)
        smplx_idx = SMPLX_POSE_JOINTS.index(smplx_joint)
        if 0 <= smplx_idx < 21:
            smplx_body_pose[:, smplx_idx] = transpose_axis_angle_intermimic_to_smplx(ref_dof[:, omomo_idx])
        elif 21 <= smplx_idx < 36:
            smplx_left_pose[:, smplx_idx-21] = transpose_axis_angle_intermimic_to_smplx(ref_dof[:, omomo_idx])
        elif 36 <= smplx_idx < 51:
            smplx_right_pose[:, smplx_idx-36] = transpose_axis_angle_intermimic_to_smplx(ref_dof[:, omomo_idx])

    return ref_data, ref_transl, ref_root_quat, ref_dof, ref_body_pos, ref_body_rot, \
        ref_obj_pos, ref_obj_quat, ref_contact_obj, ref_contact_human, smplx_body_pose, smplx_left_pose, smplx_right_pose

def kabsch_weighted(X, Y, w):
    # X,Y: [N,3], w:[N,1]
    W = w / w.sum()
    Xc = (W * X).sum(dim=0, keepdim=True)
    Yc = (W * Y).sum(dim=0, keepdim=True)
    X0 = X - Xc
    Y0 = Y - Yc
    H  = (X0 * W).T @ Y0
    U, S, Vt = torch.linalg.svd(H)
    R = Vt.T @ U.T
    if torch.det(R) < 0:
        Vt[-1] *= -1
        R = Vt.T @ U.T
    t = (Yc[0] - (R @ Xc[0]))
    return R, t

def solve_frame_6dof(X_obj, pL, RL, pR, RR, cL, cR, owner, w=None, irls_delta=0.02, iters=2):
    """
    X_obj: [N,3] object-local vertices
    pL, pR: [3], RL, RR: [3,3] (world<-hand)
    cL, cR: [N,3] stored hand-local coords (at t=0)
    owner: [N]  (0=L, 1=R, -1=both -> average targets)
    """
    N = X_obj.size(0)
    if w is None: w = torch.ones(N,1, device=X_obj.device)
    Y = torch.empty_like(X_obj)

    useL = (owner==0)
    useR = (owner==1)
    both = (owner<0)

    Y[useL]  = pL + (RL @ cL[useL].T).T
    Y[useR]  = pR + (RR @ cR[useR].T).T
    if both.any():
        YL = pL + (RL @ cL[both].T).T
        YR = pR + (RR @ cR[both].T).T
        Y[both] = 0.5*(YL+YR)

    # IRLS for robustness (Huber-like)
    R, t = kabsch_weighted(X_obj, Y, w)
    for _ in range(iters):
        Yhat = (X_obj @ R.T) + t
        r = torch.norm(Y - Yhat, dim=1, keepdim=True)
        ww = torch.where(r < irls_delta, torch.ones_like(r), irls_delta/torch.clamp(r,1e-8))
        R, t = kabsch_weighted(X_obj, Y, w*ww)

    return R, t  # world pose of the object at this frame

def get_object_trajectory(
    hand_info,
    left_wrist_pos, right_wrist_pos, left_wrist_quat, right_wrist_quat, 
    ref_left_wrist_pos, ref_right_wrist_pos, ref_left_wrist_quat, ref_right_wrist_quat,
    ref_obj_pos, ref_obj_quat):
    # left_wrist_pos        [T, 3]
    # right_wrist_pos       [T, 3]
    # left_wrist_quat       [T, 4], xyzw
    # right_wrist_quat      [T, 4], xyzw
    # ref_obj_pos=ref_data  [3]
    # ref_obj_quat=ref_data [4]
    left_wrist_quat = left_wrist_quat[..., [3, 0, 1, 2]]
    right_wrist_quat = right_wrist_quat[..., [3, 0, 1, 2]]
    ref_left_wrist_quat = ref_left_wrist_quat[..., [3, 0, 1, 2]]
    ref_right_wrist_quat = ref_right_wrist_quat[..., [3, 0, 1, 2]]
    ref_obj_pos = ref_obj_pos.reshape(3)
    ref_obj_quat = ref_obj_quat[..., [3, 0, 1, 2]].reshape(4)

    left_wrist_rot = quaternion_to_matrix(left_wrist_quat)
    right_wrist_rot = quaternion_to_matrix(right_wrist_quat)
    ref_left_wrist_rot = quaternion_to_matrix(ref_left_wrist_quat)
    ref_right_wrist_rot = quaternion_to_matrix(ref_right_wrist_quat)
    ref_obj_rot = quaternion_to_matrix(ref_obj_quat)

    ref_left_wrist_rot_inv = torch.linalg.inv(ref_left_wrist_rot)
    ref_right_wrist_rot_inv = torch.linalg.inv(ref_right_wrist_rot)
    ref_obj_rot_inv = torch.linalg.inv(ref_obj_rot)

    pseudo_vertices = torch.stack([
        (ref_left_wrist_pos - ref_obj_pos) @ ref_obj_rot_inv.T,
        (ref_left_wrist_pos - ref_obj_pos) @ ref_obj_rot_inv.T + torch.tensor([0.1, 0., 0.]),
        (ref_left_wrist_pos - ref_obj_pos) @ ref_obj_rot_inv.T + torch.tensor([0., 0.1, 0.]),
        (ref_left_wrist_pos - ref_obj_pos) @ ref_obj_rot_inv.T + torch.tensor([0., 0., 0.1]),
        (ref_left_wrist_pos - ref_obj_pos) @ ref_obj_rot_inv.T + torch.tensor([-0.1, 0., 0.]),
        (ref_left_wrist_pos - ref_obj_pos) @ ref_obj_rot_inv.T + torch.tensor([0., -0.1, 0.]),
        (ref_left_wrist_pos - ref_obj_pos) @ ref_obj_rot_inv.T + torch.tensor([0., 0., -0.1]),
        (ref_right_wrist_pos - ref_obj_pos) @ ref_obj_rot_inv.T,
        (ref_right_wrist_pos - ref_obj_pos) @ ref_obj_rot_inv.T + torch.tensor([0.1, 0., 0.]),
        (ref_right_wrist_pos - ref_obj_pos) @ ref_obj_rot_inv.T + torch.tensor([0., 0.1, 0.]),
        (ref_right_wrist_pos - ref_obj_pos) @ ref_obj_rot_inv.T + torch.tensor([0., 0., 0.1]),
        (ref_right_wrist_pos - ref_obj_pos) @ ref_obj_rot_inv.T + torch.tensor([-0.1, 0., 0.]),
        (ref_right_wrist_pos - ref_obj_pos) @ ref_obj_rot_inv.T + torch.tensor([0., -0.1, 0.]),
        (ref_right_wrist_pos - ref_obj_pos) @ ref_obj_rot_inv.T + torch.tensor([0., 0., -0.1]),
    ], dim=0)
    global_pseudo_vertices = (pseudo_vertices @ ref_obj_rot) + ref_obj_pos
    saved_left_points = (global_pseudo_vertices - ref_left_wrist_pos) @ ref_left_wrist_rot_inv.T
    saved_right_points = (global_pseudo_vertices - ref_right_wrist_pos) @ ref_right_wrist_rot_inv.T

    owner = torch.tensor([0, 0, 0, 0, 0, 0, 0, 1, 1, 1, 1, 1, 1, 1,]).float()
    
    w = torch.ones((14, 1)).float()
    if hand_info == 1:
        w[7:] = 0.0
    elif hand_info == 2:
        w[:7] = 0.0
    
    result_R, result_t = [], []
    for i in range(0, left_wrist_pos.shape[0]):
        R, t = solve_frame_6dof(
            X_obj=pseudo_vertices, 
            pL=left_wrist_pos[i], RL=left_wrist_rot[i], 
            pR=right_wrist_pos[i], RR=right_wrist_rot[i], 
            cL=saved_left_points, cR=saved_right_points, 
            owner=owner, w=w,
        )
        result_R.append(R)
        result_t.append(t)
    result_R = torch.stack(result_R, dim=0)
    result_t = torch.stack(result_t, dim=0)
    return result_t, matrix_to_quaternion(result_R)[..., [1, 2, 3, 0]]

def main_test():
    ap = argparse.ArgumentParser()
    ap.add_argument("--dataset", default="FullBodyManip")
    ap.add_argument("--category", default="largetable")
    ap.add_argument("--output_dir", default="data/InterAct_new", help="InterAct 데이터셋 폴더 주소")
    ap.add_argument("--human_motion_dir", type=str, default="results/inference/human_motion")
    ap.add_argument("--object_motion_dir", type=str, default="results/inference/object_motion")
    ap.add_argument("--ref_path", type=str, default="data/InterAct/sub3_largetable_006.pt")
    ap.add_argument("--ref_crop_start", type=int, default=0)
    ap.add_argument("--ref_blending_frame", type=int, default=0)
    ap.add_argument("--hand_info", type=int, default=0)
    args = ap.parse_args()

    total_saved = 0
    total_seen = 0

    # load reference
    ref_data, ref_transl, ref_root_quat, ref_dof, ref_body_pos, ref_body_rot, ref_obj_pos, ref_obj_quat, ref_contact_obj, ref_contact_human, \
        ref_smplx_body_pose, ref_smplx_left_pose, ref_smplx_right_pose = get_reference(args.ref_path, args.ref_crop_start, args.ref_blending_frame)

    human_motion_pths = sorted(glob(f"{args.human_motion_dir}/{args.dataset}/{args.category}/human000000000/*/*/*.npz"))
    human_motion_pths = human_motion_pths[:2]
    print("\n".join(human_motion_pths))
    os.makedirs(args.output_dir, exist_ok=True)
    for idx, human_pth in enumerate(tqdm(human_motion_pths)):
        data_pth = os.path.join(args.output_dir, f"sub0_{args.category.split('_')[0]}_{idx:03d}.pt")

        total_seen += 1

        # read human motions
        data = np.load(human_pth)
        
        # human_motion
        # poses: (T, 55*3)
        # betas: (10,)
        # trans: (T, 3)
        # mocap_frame_rate: 30
        # gender: "neutral"

        # interpolation (20FPS -> 30FPS)
        smplx_poses = torch.nn.functional.interpolate(
            torch.from_numpy(data["poses"]).float().transpose(0, 1).unsqueeze(0),   # [1, 55*3, T]
            size=int(data["poses"].shape[0] * 30 / 20),
            mode="linear",
            align_corners=True,
        ).squeeze(0).transpose(0, 1)
        smplx_trans = torch.nn.functional.interpolate(
            torch.from_numpy(data["trans"]).float().transpose(0, 1).unsqueeze(0),     # [1, 3, T]
            size=int(data["trans"].shape[0] * 30 / 20),
            mode="linear",
            align_corners=True,
        ).squeeze(0).transpose(0, 1)
        betas = torch.from_numpy(data["betas"]).float()
        gender = data["gender"]
        
        # initialize data
        frame_num = min(smplx_poses.shape[0], smplx_trans.shape[0])
        data = torch.zeros((frame_num, 591))

        # inpaint arm dof
        if args.hand_info == 0:
            smplx_poses[:, 3+12*3:3+21*3] = ref_smplx_body_pose[-1, 12:21].flatten()
        elif args.hand_info == 1:
            smplx_poses[:, 3+12*3:3+13*3] = ref_smplx_body_pose[-1, 12].flatten()
            smplx_poses[:, 3+15*3:3+16*3] = ref_smplx_body_pose[-1, 15].flatten()
            smplx_poses[:, 3+17*3:3+18*3] = ref_smplx_body_pose[-1, 17].flatten()
            smplx_poses[:, 3+19*3:3+20*3] = ref_smplx_body_pose[-1, 19].flatten()
        elif args.hand_info == 2:
            smplx_poses[:, 3+13*3:3+14*3] = ref_smplx_body_pose[-1, 13].flatten()
            smplx_poses[:, 3+16*3:3+17*3] = ref_smplx_body_pose[-1, 16].flatten()
            smplx_poses[:, 3+18*3:3+19*3] = ref_smplx_body_pose[-1, 18].flatten()
            smplx_poses[:, 3+20*3:3+21*3] = ref_smplx_body_pose[-1, 20].flatten()
        
        # forward kinematics
        smplxmodel = smplx.create(model_path="imports/mdm/body_models/smplx/SMPLX_NEUTRAL.npz", model_type="smplx", num_pca_comps=45)
        global_smplxmodel_output = smplxmodel(
            betas=betas.reshape(1, 10).repeat((frame_num, 1)).float(),
            global_orient=smplx_poses[:frame_num, :3].float(),
            body_pose=smplx_poses[:frame_num, 3:3+21*3].float(),
            left_hand_pose=ref_smplx_left_pose[-1].reshape(1, 45).repeat((frame_num, 1)),
            right_hand_pose=ref_smplx_right_pose[-1].reshape(1, 45).repeat((frame_num, 1)),
            transl=smplx_trans[:frame_num, :3].float(),
            expression=torch.zeros((frame_num, 10)).float(),
            jaw_pose=torch.zeros((frame_num, 3)).float(),
            leye_pose=torch.zeros((frame_num, 3)).float(),
            reye_pose=torch.zeros((frame_num, 3)).float(),
            return_verts=True,
            return_full_pose=True,
        )
        global_joints = global_smplxmodel_output.joints.to(torch.float32).cpu()[:, :55, :] # (T, 127, 3)
        local_joint_rot = global_smplxmodel_output.full_pose[:, :55*3].reshape(frame_num, 55, 3) # (T, 55, 3)
        global_joint_rot = get_absolute_joint_rotations(local_joint_rot, smplxmodel)

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
                        smplx_poses[:frame_num, 3+sidx*3:3+(sidx+1)*3].float()
                    )
                elif 21 <= sidx < 36:
                    omomo_dof_pos[:, oidx] = transpose_axis_angle_smplx_to_intermimic(
                        ref_smplx_left_pose[-1:, sidx-21].float()
                    )
                elif 36 <= sidx < 51:
                    omomo_dof_pos[:, oidx] = transpose_axis_angle_smplx_to_intermimic(
                        ref_smplx_right_pose[-1:, sidx-36].float()
                    )

        # reference, sample 맞추기
        # ref_transl, ref_root_quat, ref_dof, ref_body_pos, ref_body_rot, 
        # ref_obj_pos, ref_obj_quat, ref_contact_obj, ref_contact_human
        
        # root orientation: y축(위쪽) 회전만 해서 ref를 sample에 맞추기
        def rz_torch(theta):  # theta: (...,)
            c, s = torch.cos(theta), torch.sin(theta)
            z = torch.zeros_like(theta)
            o = torch.ones_like(theta)
            R = torch.stack([
                torch.stack([ c, -s, z], dim=-1),
                torch.stack([ s,  c, z], dim=-1),
                torch.stack([ z,  z, o], dim=-1),
            ], dim=-2)  # (...,3,3)
            return R
        def solve_C_global_z_torch(A, B):
            # A,B: (...,3,3)
            Arel = B @ A.transpose(-1, -2)
            a = Arel[..., 1, 0] - Arel[..., 0, 1]
            b = Arel[..., 0, 0] + Arel[..., 1, 1]
            theta = torch.atan2(a, b)
            C = rz_torch(theta)
            return C, theta  # apply: A2 = C @ A
        ref_root_rot = quaternion_to_matrix(ref_root_quat[-1, :])
        root_rot = axis_angle_to_matrix(transpose_axis_angle_smplx_to_intermimic(smplx_poses[args.ref_blending_frame-1:args.ref_blending_frame, :3]))[0]
        C, _ = solve_C_global_z_torch(ref_root_rot, root_rot)

        # trans: ref를 sample에 맞추기
        D = omomo_body_pos[args.ref_blending_frame-1:args.ref_blending_frame, 0, :] - (ref_transl[-1:] @ C.T)
        D[..., 2] = 0.0

        # 높이: sample을 ref에 맞추기
        H = (ref_transl[-1:] @ C.T) - omomo_body_pos[args.ref_blending_frame-1:args.ref_blending_frame, 0, :]
        H[..., :2] = 0.0

        # root_pos: data[:, 0:3]
        data[:, 0:3] = omomo_body_pos[:, 0, :] + H
        data[:args.ref_blending_frame, 0:3] = (ref_transl @ C.T) + D

        # root_rot: data[:, 3:7] (xyzw quat)
        data[:, [6, 3, 4, 5]] = axis_angle_to_quaternion(transpose_axis_angle_smplx_to_intermimic(
            smplx_poses[:frame_num, :3]
        ))
        data[:args.ref_blending_frame, [6, 3, 4, 5]] = quaternion_multiply(matrix_to_quaternion(C), ref_root_quat)
        
        # dof_pos: data[:, 9:9+51*3]
        data[:, 9:9+51*3] = omomo_dof_pos.reshape(frame_num, 51*3)
        data[:args.ref_blending_frame, 9:9+51*3] = ref_dof.reshape(args.ref_blending_frame, 51*3)
        # data[args.ref_blending_frame:, 9+(14)*3:9+(15+17)*3] = data[args.ref_blending_frame-1, 9+(14)*3:9+(15+17)*3]
        # data[args.ref_blending_frame:, 9+(33)*3:9+(34+17)*3] = data[args.ref_blending_frame-1, 9+(33)*3:9+(34+17)*3]
        # data[args.ref_blending_frame:, 9+(15)*3:9+(15+17)*3] = ref_data[args.ref_blending_frame-1, 9+(15)*3:9+(15+17)*3]
        # data[args.ref_blending_frame:, 9+(34)*3:9+(34+17)*3] = ref_data[args.ref_blending_frame-1, 9+(34)*3:9+(34+17)*3]
        
        # body_pos: data[:, 162:162+52*3]
        data[:, 162:162+52*3] = (omomo_body_pos + H).reshape(frame_num, 52*3)
        data[:args.ref_blending_frame, 162:162+52*3] = ((ref_body_pos @ C.T) + D).reshape(args.ref_blending_frame, 52*3)
        
        # contact_obj: data[:, 330:331]
        data[:, 330:331] = 1.0
        data[:args.ref_blending_frame, 330] = ref_contact_obj
        
        # contact_human: data[:, 331:331+52]
        no_touch = [331 + i for i in [0, 1, 2, 5, 6, 9, 10, 11, 12, 13, 14, 15, 16, 33, 34, 35]]
        data[:, no_touch] = -1.0
        if args.hand_info == 0:
            data[:, 331+17:331+17+16] = 1.0
            data[:, 331+36:331+36+16] = 1.0
        elif args.hand_info == 1:
            data[:, 331+17:331+17+16] = 1.0
        elif args.hand_info == 2:
            data[:, 331+36:331+36+16] = 1.0
        data[:args.ref_blending_frame, 331:331+52] = ref_contact_human
        
        # body_rot: data[:, 383:383+52*4] (xyzw quat)
        data[:, 383:383+52*4] = axis_angle_to_quaternion(omomo_body_rot)[..., [1, 2, 3, 0]].reshape(frame_num, 52*4)
        data[:args.ref_blending_frame, 383:383+52*4] = quaternion_multiply(matrix_to_quaternion(C), ref_body_rot[..., [3, 0, 1, 2]])[..., [1, 2, 3, 0]].reshape(args.ref_blending_frame, 52*4)
        
        # not used: data[:, 7:9], data[:, 325:330]
        data[:, 7:9] = 0.0
        data[:, 325:330] = 0.0

        # w = torch.arange(1/15, 16/15, 1/15).unsqueeze(-1)
        # data[args.ref_blending_frame:args.ref_blending_frame+15] = w * data[args.ref_blending_frame:args.ref_blending_frame+15] + (1 - w) * reference[args.ref_blending_frame:args.ref_blending_frame+15]
        # w = torch.arange(1/15, 16/15, 1/15).unsqueeze(-1)
        # data[args.ref_blending_frame:args.ref_blending_frame+15] = w * data[args.ref_blending_frame:args.ref_blending_frame+15] + (1 - w) * data[args.ref_blending_frame-1]

        obj_pos, obj_quat = get_object_trajectory(
            hand_info=args.hand_info,
            left_wrist_pos=data[args.ref_blending_frame:, 162+17*3:162+18*3],
            right_wrist_pos=data[args.ref_blending_frame:, 162+36*3:162+37*3],
            left_wrist_quat=data[args.ref_blending_frame:, 383+17*4:383+18*4],
            right_wrist_quat=data[args.ref_blending_frame:, 383+36*4:383+37*4],
            ref_left_wrist_pos=data[args.ref_blending_frame-1, 162+17*3:162+18*3],
            ref_right_wrist_pos=data[args.ref_blending_frame-1, 162+36*3:162+37*3],
            ref_left_wrist_quat=data[args.ref_blending_frame-1, 383+17*4:383+18*4],
            ref_right_wrist_quat=data[args.ref_blending_frame-1, 383+36*4:383+37*4],
            ref_obj_pos=(ref_data[-1, 318:321] @ C.T) + D,
            ref_obj_quat=quaternion_multiply(matrix_to_quaternion(C), ref_data[args.ref_blending_frame-1, [324, 321, 322, 323]])[..., [1, 2, 3, 0]],
        )

        # obj_pos: data[:, 318:321]
        data[:args.ref_blending_frame, 318:321] = (ref_data[:, 318:321] @ C.T) + D
        data[args.ref_blending_frame:, 318:321] = obj_pos

        data[:args.ref_blending_frame, 321:325] = quaternion_multiply(matrix_to_quaternion(C), ref_data[:, [324, 321, 322, 323]])[..., [1, 2, 3, 0]]
        data[args.ref_blending_frame:, 321:325] = obj_quat

        torch.save(data, data_pth)
        total_saved += 1
        if total_saved % 50 == 0:
            print(f"[info] saved {total_saved} sequences ...")
    
    print(f"[done] seen={total_seen}, saved={total_saved}")

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--dataset", default="FullBodyManip")
    ap.add_argument("--category", default="largetable")
    ap.add_argument("--output_dir", default="data/InterAct_new", help="InterAct 데이터셋 폴더 주소")
    ap.add_argument("--human_motion_dir", type=str, default="results/inference/human_motion")
    ap.add_argument("--object_motion_dir", type=str, default="results/inference/object_motion")
    ap.add_argument("--ref_path", type=str, default="data/InterAct/sub3_largetable_006.pt")
    ap.add_argument("--ref_crop_start", type=int, default=0)
    ap.add_argument("--ref_blending_frame", type=int, default=0)
    args = ap.parse_args()

    total_saved = 0
    total_seen = 0

    human_motion_pths = sorted(glob(f"{args.human_motion_dir}/{args.dataset}/{args.category}/human000000000/*/*/*.npz"))
    obj_motion_pths = [human_pth.replace(args.human_motion_dir, args.object_motion_dir).replace(".npz", ".pkl") for human_pth in human_motion_pths]
    print("\n".join(human_motion_pths))
    os.makedirs(args.output_dir, exist_ok=True)
    for idx, (human_pth, obj_pth) in enumerate(zip(tqdm(human_motion_pths), obj_motion_pths)):
        data_pth = os.path.join(args.output_dir, f"sub0_{args.category.split('_')[0]}_{idx:03d}.pt")
        # if os.path.exists(data_pth): continue
        # if not os.path.exists(obj_pth): continue

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

        if "smallbox" in args.category or "clothesstand_left_hand" in args.category:
            Z_ROT = torch.tensor([
                [-1., 0., 0.],
                [0., -1., 0.],
                [0., 0., 1.]
            ]).float()
        else:
            Z_ROT = torch.tensor([
                [0.0, 1.0, 0.0],
                [-1.0, 0.0, 0.0],
                [0.0, 0.0, 1.0],
            ]).float()
        Z_QUAT = matrix_to_quaternion(Z_ROT)

        # root_pos: data[:, 0:3]
        data[:, 0:3] = omomo_body_pos[:, 0, :] @ Z_ROT.T

        # root_rot: data[:, 3:7] (xyzw quat)
        data[:, [6, 3, 4, 5]] = quaternion_multiply(Z_QUAT, axis_angle_to_quaternion(transpose_axis_angle_smplx_to_intermimic(
            torch.from_numpy(human_motion["poses"][:frame_num, :3]).float()
        )))
        
        # dof_pos: data[:, 9:9+51*3]
        data[:, 9:9+51*3] = omomo_dof_pos.reshape(frame_num, 51*3)
        
        # body_pos: data[:, 162:162+52*3]
        data[:, 162:162+52*3] = (omomo_body_pos @ Z_ROT.T).reshape(frame_num, 52*3)
        
        # obj_pos: data[:, 318:321]
        data[:, 318:321] = transpose_transl_smplx_to_intermimic(
            torch.from_numpy(obj_motion["t"][:frame_num, :, 0]).float()
        ) @ Z_ROT.T
        
        # obj_rot: data[:, 321:325] (xyzw quat)
        # COMPATIBILITY_MATRIX = np.array([[1.0, 0.0, 0.0], [0.0, 0.0, 1.0], [0.0, -1.0, 0.0]])
        # ROT_OFS = np.array([[-1.0, 0.0, 0.0], [0.0, 0.0, 1.0], [0.0, 1.0, 0.0]])
        # ROT_OFS_INV = np.array([[-1.0, 0.0, 0.0], [0.0, 0.0, 1.0], [0.0, 1.0, 0.0]])
        # data[:, [324, 321, 322, 323]] = matrix_to_quaternion(
        #     torch.from_numpy(COMPATIBILITY_MATRIX.T @ obj_motion["R"][:frame_num] @ COMPATIBILITY_MATRIX @ ROT_OFS_INV).float()
        # )
        NEW_ROT_OFS = np.array([[1.0, 0.0, 0.0], [0.0, 0.0, 1.0], [0.0, -1.0, 0.0]])    # x축 기준 -90도 회전
        NEW_ROT_OFS_INV = np.linalg.inv(NEW_ROT_OFS)
        data[:, [324, 321, 322, 323]] = matrix_to_quaternion(
            torch.from_numpy(NEW_ROT_OFS_INV @ obj_motion["R"][:frame_num]).float()
        )
        if "smallbox" in args.category or "clothesstand_left_hand" in args.category:
            data[:, [324, 321, 322, 323]] = quaternion_multiply(
                matrix_to_quaternion(torch.tensor([
                    [0.0, 1.0, 0.0],
                    [-1.0, 0.0, 0.0],
                    [0.0, 0.0, 1.0],
                ]).float()), data[:, [324, 321, 322, 323]]
            )
        
        # contact_obj: data[:, 330:331]
        data[:, 330:331] = 1.0
        
        # contact_human: data[:, 331:331+52]
        no_touch = [331 + i for i in [0, 1, 2, 5, 6, 9, 10, 11, 12, 13, 14, 15, 16, 33, 34, 35]]
        data[:, no_touch] = -1.0
        data[:, 331+17:331+17+16] = 1.0
        data[:, 331+36:331+36+16] = 1.0
        
        # body_rot: data[:, 383:383+52*4] (xyzw quat)
        data[:, 383:383+52*4] = quaternion_multiply(Z_QUAT, axis_angle_to_quaternion(omomo_body_rot))[..., [1, 2, 3, 0]].reshape(frame_num, 52*4)
        
        # not used: data[:, 7:9], data[:, 325:330]
        data[:, 7:9] = 0.0
        data[:, 325:330] = 0.0

        # interpolation: 20FPS -> 30FPS
        data_t = data.transpose(0, 1).unsqueeze(0)  # shape: [1, 591, T]
        target_len = int(frame_num * 30 / 20)
        upsampled = torch.nn.functional.interpolate(data_t, size=target_len, mode='linear', align_corners=True)
        upsampled = upsampled.squeeze(0).transpose(0, 1)

        # inpainting reference
        reference = torch.load(args.ref_path)
        contact_mask = reference[:, 330] > 0.5
        contact_idx = torch.nonzero(contact_mask.flatten(), as_tuple=True)[0]
        t1 = int(contact_idx[0]) + args.ref_crop_start
        reference = reference[t1:]

        upsampled[:args.ref_blending_frame] = reference[:args.ref_blending_frame]
        gap = reference[args.ref_blending_frame, 0:3] - upsampled[args.ref_blending_frame, 0:3]; gap[2] = 0.0   # gap of root trans
        upsampled[args.ref_blending_frame:, 0:3] += gap  # root trans
        upsampled[args.ref_blending_frame:, 162:162+52*3] = (upsampled[args.ref_blending_frame:, 162:162+52*3].reshape(-1, 3) + gap).reshape(-1, 52*3)    # body_pos trans
        obj_gap = reference[args.ref_blending_frame, 318:321] - upsampled[args.ref_blending_frame, 318:321]; obj_gap[2] = 0.0
        upsampled[args.ref_blending_frame:, 318:321] += obj_gap  # obj trans
        if "largetable_carry" in args.category:
            upsampled[args.ref_blending_frame:, 319] -= 0.10
            upsampled[args.ref_blending_frame:, 320] += 0.10
        elif "largetable_lift" in args.category:
            upsampled[args.ref_blending_frame:, 318] += 0.05
        elif "smallbox" in args.category:
            upsampled[args.ref_blending_frame:, 318] -= 0.1
            upsampled[args.ref_blending_frame:, 320] += 0.1
        elif "clothesstand_left_hand" in args.category:
            upsampled[args.ref_blending_frame:, 318] += 0.25
            upsampled[args.ref_blending_frame:, 319] += 0.1
        elif "clothesstand_right_hand" in args.category:
            upsampled[args.ref_blending_frame:, 318] += 0.05
            upsampled[args.ref_blending_frame:, 319] += 0.05
        elif "clothesstand_two_hands" in args.category: 
            upsampled[args.ref_blending_frame:, 318] -= 0.5
            upsampled[args.ref_blending_frame:, 319] += 0.05
        upsampled[args.ref_blending_frame:, 3:7] *= same_sign_quaternions(upsampled[args.ref_blending_frame, 3:7], upsampled[args.ref_blending_frame-1, 3:7])  # root rot
        upsampled[args.ref_blending_frame:, 321:325] *= same_sign_quaternions(upsampled[args.ref_blending_frame, 321:325], upsampled[args.ref_blending_frame-1, 321:325])  # obj rot        
        upsampled[args.ref_blending_frame:, 9+(15)*3:9+(16+16)*3] = upsampled[args.ref_blending_frame-1, 9+(15)*3:9+(16+16)*3]  # DOF left hand
        upsampled[args.ref_blending_frame:, 9+(34)*3:9+(35+16)*3] = upsampled[args.ref_blending_frame-1, 9+(34)*3:9+(35+16)*3]  # DOF right hand

        w = torch.arange(1/15, 16/15, 1/15).unsqueeze(-1)
        upsampled[args.ref_blending_frame:args.ref_blending_frame+15] = w * upsampled[args.ref_blending_frame:args.ref_blending_frame+15] + (1 - w) * reference[args.ref_blending_frame:args.ref_blending_frame+15]

        torch.save(upsampled, data_pth)
        total_saved += 1
        if total_saved % 50 == 0:
            print(f"[info] saved {total_saved} sequences ...")
    
    print(f"[done] seen={total_seen}, saved={total_saved}")

if __name__ == "__main__":
    main_test()
