import numpy as np
import torch
import torch.nn.functional as F

def _sqrt_positive_part(x: torch.Tensor) -> torch.Tensor:
    """
    Returns torch.sqrt(torch.max(0, x))
    but with a zero subgradient where x is 0.
    """
    ret = torch.zeros_like(x)
    positive_mask = x > 0
    if torch.is_grad_enabled():
        ret[positive_mask] = torch.sqrt(x[positive_mask])
    else:
        ret = torch.where(positive_mask, torch.sqrt(x), ret)
    return ret

def matrix_to_quaternion(matrix: torch.Tensor) -> torch.Tensor:
    """
    Convert rotations given as rotation matrices to quaternions.

    Args:
        matrix: Rotation matrices as tensor of shape (..., 3, 3).

    Returns:
        quaternions with real part first, as tensor of shape (..., 4).
    """
    if matrix.size(-1) != 3 or matrix.size(-2) != 3:
        raise ValueError(f"Invalid rotation matrix shape {matrix.shape}.")

    batch_dim = matrix.shape[:-2]
    m00, m01, m02, m10, m11, m12, m20, m21, m22 = torch.unbind(
        matrix.reshape(batch_dim + (9,)), dim=-1
    )

    q_abs = _sqrt_positive_part(
        torch.stack(
            [
                1.0 + m00 + m11 + m22,
                1.0 + m00 - m11 - m22,
                1.0 - m00 + m11 - m22,
                1.0 - m00 - m11 + m22,
            ],
            dim=-1,
        )
    )

    # we produce the desired quaternion multiplied by each of r, i, j, k
    quat_by_rijk = torch.stack(
        [
            # pyre-fixme[58]: `**` is not supported for operand types `Tensor` and
            #  `int`.
            torch.stack([q_abs[..., 0] ** 2, m21 - m12, m02 - m20, m10 - m01], dim=-1),
            # pyre-fixme[58]: `**` is not supported for operand types `Tensor` and
            #  `int`.
            torch.stack([m21 - m12, q_abs[..., 1] ** 2, m10 + m01, m02 + m20], dim=-1),
            # pyre-fixme[58]: `**` is not supported for operand types `Tensor` and
            #  `int`.
            torch.stack([m02 - m20, m10 + m01, q_abs[..., 2] ** 2, m12 + m21], dim=-1),
            # pyre-fixme[58]: `**` is not supported for operand types `Tensor` and
            #  `int`.
            torch.stack([m10 - m01, m20 + m02, m21 + m12, q_abs[..., 3] ** 2], dim=-1),
        ],
        dim=-2,
    )

    # We floor here at 0.1 but the exact level is not important; if q_abs is small,
    # the candidate won't be picked.
    flr = torch.tensor(0.1).to(dtype=q_abs.dtype, device=q_abs.device)
    quat_candidates = quat_by_rijk / (2.0 * q_abs[..., None].max(flr))

    # if not for numerical problems, quat_candidates[i] should be same (up to a sign),
    # forall i; we pick the best-conditioned one (with the largest denominator)
    out = quat_candidates[
        F.one_hot(q_abs.argmax(dim=-1), num_classes=4) > 0.5, :
    ].reshape(batch_dim + (4,))
    return standardize_quaternion(out)

def standardize_quaternion(quaternions: torch.Tensor) -> torch.Tensor:
    """
    Convert a unit quaternion to a standard form: one in which the real
    part is non negative.

    Args:
        quaternions: Quaternions with real part first,
            as tensor of shape (..., 4).

    Returns:
        Standardized quaternions as tensor of shape (..., 4).
    """
    return torch.where(quaternions[..., 0:1] < 0, -quaternions, quaternions)

def rotation(views,idx): 
    rt_matrix = views[idx].obj_rt.clone()
    rt_quat = views[idx].obj_quat.clone()
    if idx < 15:
            rt_matrix[2:] = views[36+idx].obj_rt[2:]
            rt_quat[2:] = views[36+idx].obj_quat[2:]
    else:
        rt_matrix[2:] = views[-1].obj_rt[2:]
        rt_matrix[1] = views[15].obj_rt[1]
        rt_quat[2:] = views[-1].obj_quat[2:]
        rt_quat[1] = views[15].obj_quat[1]
    # rt_matrix[0] = views[0].obj_rt[0]
    # rt_quat[0] = views[0].obj_quat[0]

    Rt_1 = np.zeros((4,4),dtype=np.float32)
    if idx < 15:
        t = 0
        tt = 0
    elif idx  < 33:  
        t = -0.01 *(idx-14) 
        tt = (idx-14)*0.05
    else: 
        t = -0.01*18 + 0.01 *(idx-32)
        tt = (51-idx)*0.05
    # if idx < 25: 
    #     t = -0.01*idx 
    # else:
    #     t = -0.01*24 + 0.01 *(idx-24) 
    c = np.cos(t)
    s = np.sin(t)
    Rt_1[:3,:3] = np.array([[c, 0, s], [0, 1, 0], [-s, 0, c]])
    Rt_1[3,3] = 1
    rt_matrix[2] = rt_matrix[2] @ torch.tensor(Rt_1,device = rt_matrix.device)
    rt_matrix[2,0,3] += tt 

    rt_quat = matrix_to_quaternion(rt_matrix[:,:3,:3].permute(0,2,1))
    #for i in range(len(rt_quat)): 
        #rt_quat[i] = torch.tensor(rotmat2qvec(rt_matrix[i,:3,:3].T))
    return rt_matrix,rt_quat

def rotation_tranport(views,idx): 
    rt_matrix = views[idx].obj_rt.clone()
    rt_quat = views[idx].obj_quat.clone()
    if idx < 15:
            rt_matrix[2:] = views[36+idx].obj_rt[2:]
            rt_quat[2:] = views[36+idx].obj_quat[2:]
    else:
        rt_matrix[2:] = views[-1].obj_rt[2:]
        rt_matrix[1] = views[15].obj_rt[1]
        rt_quat[2:] = views[-1].obj_quat[2:]
        rt_quat[1] = views[15].obj_quat[1]

    Rt_1 = np.zeros((4,4),dtype=np.float32)
    if idx < 15:
        t = 0
        tt = 0
    elif idx  < 24:  
        t = 0
        tt = (idx-14)*0.05
    elif idx < 33: 
        t = 0.04 *(idx-23)  
        tt = 9*0.05
    elif idx < 42: 
        t = 0.04*9 - 0.04 *(idx-32) 
        tt = 9*0.05
    else: 
        t = 0
        tt = (50-idx)*0.05
    # if idx < 25: 
    #     t = -0.01*idx 
    # else:
    #     t = -0.01*24 + 0.01 *(idx-24) 
    c = np.cos(t)
    s = np.sin(t)
    Rt_1[:3,:3] = np.array([[c, 0, s], [0, 1, 0], [-s, 0, c]])
    Rt_1[3,3] = 1
    rt_matrix[2] = rt_matrix[2] @ torch.tensor(Rt_1,device = rt_matrix.device)
    rt_matrix[2,0,3] += tt 

    rt_quat = matrix_to_quaternion(rt_matrix[:,:3,:3].permute(0,2,1))
    #for i in range(len(rt_quat)): 
        #rt_quat[i] = torch.tensor(rotmat2qvec(rt_matrix[i,:3,:3].T))
    return rt_matrix,rt_quat
    
def rotmat2qvec(R):
    Rxx, Ryx, Rzx, Rxy, Ryy, Rzy, Rxz, Ryz, Rzz = R.reshape(-1).cpu()
    K = np.array([
        [Rxx - Ryy - Rzz, 0, 0, 0],
        [Ryx + Rxy, Ryy - Rxx - Rzz, 0, 0],
        [Rzx + Rxz, Rzy + Ryz, Rzz - Rxx - Ryy, 0],
        [Ryz - Rzy, Rzx - Rxz, Rxy - Ryx, Rxx + Ryy + Rzz]]) / 3.0
    eigvals, eigvecs = np.linalg.eigh(K)
    qvec = eigvecs[[3, 0, 1, 2], np.argmax(eigvals)]
    if qvec[0] < 0:
        qvec *= -1
    return qvec

    
    
