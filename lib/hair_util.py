import torch
import numpy as np
import open3d as o3d
import os
import trimesh
from .mesh_util import load_obj_mesh

def hair_synthesis(net, cuda, root_tensor, calib_tensor, num_sample=100, hair_unit=0.006):
    #root:[3, 1024]
    num_strand = root_tensor.shape[2]
    hair_strands = torch.zeros(num_sample, 3, num_strand).to(device=cuda)
    
    curr_node = root_tensor.squeeze()
    hair_strands[0] = curr_node
    for i in range(1,num_sample-1):
        curr_node_orien = net.query(curr_node.unsqueeze(0), calib_tensor)
        curr_node_orien = curr_node_orien.squeeze()
        hair_strands[i] = hair_strands[i-1] + hair_unit * curr_node_orien
        curr_node = hair_strands[i]

    return hair_strands.permute(2, 0, 1).cpu().detach().numpy()

def hair_synthesis_DSH(net, cuda, root_tensor, calib_tensor, num_sample=100, hair_unit=0.006, threshold=[60,150]):
    #growing algorithm in DeepSketchHair
    #root:[3, 1024]
    num_strand = root_tensor.shape[2]
    hair_strands = torch.zeros(num_sample, 3, num_strand).to(device=cuda)
    
    curr_node = root_tensor.squeeze()
    last_node_orien = 0
    hair_strands[0] = curr_node
    for i in range(1,num_sample-1):
        curr_node_orien = net.query(curr_node.unsqueeze(0), calib_tensor).squeeze()

        if i>1:
            len_cd = torch.norm(curr_node_orien,p=2,dim=0)
            len_pd = torch.norm(last_node_orien,p=2,dim=0)
            in_prod = torch.sum(curr_node_orien * last_node_orien, dim=0)
            theta = torch.acos( in_prod/ (len_cd*len_pd))*180/np.pi

            idx_big_theta = theta > threshold[1]
            idx_mid_theta = ((theta > threshold[0]).float() - idx_big_theta.float()).bool().unsqueeze(0)#60<theta<150
            idx_stop = (idx_big_theta + torch.isnan(theta).float()).bool().unsqueeze(0)#orien=0 or theta>150

            idx_stop = torch.cat((idx_stop,idx_stop,idx_stop),dim=0)
            idx_mid_theta = torch.cat((idx_mid_theta,idx_mid_theta,idx_mid_theta),dim=0)
            

            half_node_orien = (curr_node_orien + last_node_orien)/2

            orien_zeros = torch.zeros_like(curr_node_orien).float().to(device=cuda)
            curr_node_orien = torch.where(idx_stop, orien_zeros, curr_node_orien)
            curr_node_orien = torch.where(idx_mid_theta, half_node_orien, curr_node_orien)

        hair_strands[i] = hair_strands[i-1] + hair_unit * curr_node_orien
        curr_node = hair_strands[i]
        last_node_orien = curr_node_orien

    return hair_strands.permute(2, 0, 1).cpu().detach().numpy()

def save_strands_with_mesh(strands, mesh_path, outputpath, err=0.3, is_eval=False):
    import sys
    print(f'[DEBUG] save_strands_with_mesh: strands={strands.shape}, mesh_path={mesh_path}, outputpath={outputpath}, err={err}'); sys.stdout.flush()

    print('[DEBUG] loading mesh...'); sys.stdout.flush()
    mesh = trimesh.load(mesh_path, process=False)
    print(f'[DEBUG] mesh loaded: vertices={len(mesh.vertices)}, faces={len(mesh.faces)}'); sys.stdout.flush()
    #for coarse mesh /1000.0
    if is_eval:
        mesh.vertices = mesh.vertices/1000.0
        print('[DEBUG] mesh vertices scaled by /1000'); sys.stdout.flush()

    lst_pc_all_valid = []
    lst_num_pt = []
    pc_all_valid = []
    lines = []
    sline = 0

    norms0 = np.einsum('ij,ij->i', strands[:, 0], strands[:, 0])
    norms1 = np.einsum('ij,ij->i', strands[:, 1], strands[:, 1])
    valid_mask = ~((norms0 < 0.001) | (norms1 < 0.001))

    valid_indices = np.where(valid_mask)[0]
    valid_strands = strands[valid_indices]  # (N_valid, 100, 3)
    print(f'[DEBUG] valid strands: {len(valid_indices)}/{len(strands)} ({len(valid_indices)/len(strands)*100:.1f}%)'); sys.stdout.flush()

    flat_pts = valid_strands.reshape(-1, 3)
    print(f'[DEBUG] mesh.contains starting: {flat_pts.shape[0]} points ({len(valid_indices)} valid strands)'); sys.stdout.flush()
    chunk_size = 50000
    chunks = [flat_pts[i:i+chunk_size] for i in range(0, len(flat_pts), chunk_size)]
    results = []
    for ci, chunk in enumerate(chunks):
        print(f'[DEBUG] mesh.contains chunk {ci+1}/{len(chunks)} ({len(chunk)} pts)'); sys.stdout.flush()
        results.append(mesh.contains(chunk))
    all_pts_in_out = np.concatenate(results).reshape(len(valid_indices), strands.shape[1])
    print(f'[DEBUG] mesh.contains done: {all_pts_in_out.sum()} points inside mesh'); sys.stdout.flush()

    print('[DEBUG] building strand point lists...'); sys.stdout.flush()
    for k, i in enumerate(valid_indices):
        current_pc_all_valid = []
        num_pt = 2
        current_pc_all_valid.append(strands[i][0])
        current_pc_all_valid.append(strands[i][1])

        pts_in_out = all_pts_in_out[k]

        for j in range(2, strands.shape[1]):
            if pts_in_out[j]:
                current_pc_all_valid.append(strands[i][j])
                num_pt += 1
            else:
                break
        lst_pc_all_valid.append(current_pc_all_valid)
        lst_num_pt.append(num_pt)

    min_num_pts = int(sum(lst_num_pt)/len(lst_num_pt)*err)
    print(f'[DEBUG] min_num_pts threshold: {min_num_pts}, avg strand len: {sum(lst_num_pt)/len(lst_num_pt):.2f}'); sys.stdout.flush()

    kept = 0
    for i in range(len(lst_num_pt)):
        if lst_num_pt[i] < min_num_pts:
            continue
        kept += 1
        pc_all_valid = pc_all_valid + lst_pc_all_valid[i]

        for j in range(lst_num_pt[i]-1):
            lines.append([sline + j, sline + j + 1])
        sline += lst_num_pt[i]

    print(f'[DEBUG] strands kept after length filter: {kept}/{len(lst_num_pt)}, total points: {len(pc_all_valid)}, total lines: {len(lines)}'); sys.stdout.flush()
    print(f'[DEBUG] writing line set to {outputpath}...'); sys.stdout.flush()
    line_set = o3d.geometry.LineSet(points=o3d.utility.Vector3dVector(np.asarray(pc_all_valid)), lines=o3d.utility.Vector2iVector(lines))
    o3d.io.write_line_set(outputpath, line_set)
    print('[DEBUG] write complete'); sys.stdout.flush()

def get_hair_root(filepath='./data/roots10k.obj'):
    root, _ = load_obj_mesh(filepath)
    return root.T

def export_hair_real(net, cuda, data, mesh_path, save_path):
    print('[DEBUG] export_hair_real started')
    print('[DEBUG] mesh_path:', mesh_path)
    print('[DEBUG] save_path:', save_path)

    image_tensor = data['hairstep'].to(device=cuda).unsqueeze(0)
    calib_tensor = data['calib'].to(device=cuda).unsqueeze(0)
    root_tensor = torch.from_numpy(get_hair_root()).to(device=cuda).float().unsqueeze(0)
    print('[DEBUG] image_tensor shape:', tuple(image_tensor.shape))
    print('[DEBUG] calib_tensor shape:', tuple(calib_tensor.shape))
    print('[DEBUG] root_tensor shape:', tuple(root_tensor.shape))

    net.filter(image_tensor)
    print('[DEBUG] net.filter completed')

    strands = hair_synthesis(net, cuda, root_tensor, calib_tensor, num_sample=100, hair_unit=0.006)
    print('[DEBUG] strands shape:', strands.shape)
    save_strands_with_mesh(strands, mesh_path, save_path, 0.3)
    print('[DEBUG] save_strands_with_mesh completed')
