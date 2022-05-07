from operator import imod
import numpy as np
from tqdm import tqdm
from os.path import join
from operator import imod
import numpy as np
from tqdm import tqdm
from os.path import join
import os
import cv2
import time
import torch
import math

from lib.models.smplmodel import load_model, merge_params, select_nf
from lib.dataset import ImageData
from lib.models.estimator import SPIN, init_with_spin
from lib.tools import simple_assign, smooth_pose, get_poses_difference
from lib.pipeline.mirror import multi_stage_optimize
from ortools.linear_solver import pywraplp
import json
def read_json(path):
    with open(path) as f:
        data = json.load(f)
    return data

Radius = 0.4

def demo(image_path,annot_root,out_root,body_model,spin_model,args, counter=1):
    """
    optimize single image
    path: the path of this single image
    body_model: smpl is the default model
    """
    dataset = ImageData(image_path,annot_root,out_root)
    dataset.load_annots()
    camera = dataset.load_camera()
    annots = dataset.annots
    #image = dataset.img
    image = cv2.imread(image_path)
    # initialize the smpl parameters
    body_params_all = []
    bboxes,keypoints2d,pids = [],[],[]
    poses_all = []
    for i,annot in enumerate(annots):
        assert annot['personID'] == i, (i, annot['personID'])
        result = init_with_spin(body_model,spin_model,image,annot['bbox'],annot['keypoints'],camera)
        body_params_all.append(result['body_params'])
        poses_all.append(result['body_params']['poses'])
        bboxes.append(annot['bbox'])
        keypoints2d.append(annot['keypoints'])
        pids.append(annot['personID'])

    poses_all = np.vstack(poses_all)
    N_people = len(pids)
    n_variables = int(N_people*(N_people-1)/2)
    x_dict = {}
    index_dict = {}
    index_matrix = np.zeros((N_people,N_people))

    counter = 0
    start_y = 1
    # get the index matrix of variables for each pair
    for i in range(N_people-1):
        for j in range(start_y,N_people):
            index_matrix[i][j] = counter
            counter+=1
        start_y+=1
    
    end_y = 0
    for i in range(N_people):
        index_matrix[i][i] = -1
        if i>=1:
            for j in range(end_y+1):
                index_matrix[i][j] = index_matrix[j][i]
            end_y +=1
    
    for i in range(N_people):
        indexes = []
        for j in range(N_people):
            if index_matrix[i][j] !=-1:
                indexes.append(index_matrix[i][j])
        index_dict[i] = indexes
    poses_diff_matrix = np.zeros((N_people,N_people))
    start_y = 1
    
    # get the pose differences matrix
    poses_diff_array = np.zeros(n_variables)
    counter = 0
    for i in range(N_people):
        for j in range(start_y,N_people):
            print(poses_all.shape)
            poses_diff_matrix[i][j] = get_poses_difference(i,j, poses_all)
            poses_diff_array[counter] = poses_diff_matrix[i][j]
            counter+=1
        start_y +=1

    # construct the binary integer linear program solver
    solver = pywraplp.Solver('simple_mip_program', pywraplp.Solver.CBC_MIXED_INTEGER_PROGRAMMING) 
    for i in range(n_variables): 
        x_dict[i] = solver.IntVar(0, 1, f'x_{i}')

    for i in range(N_people):
        solver.Add(solver.Sum([-1*x_dict[i] for i in index_dict[i]]) <= -1)
    solver.Minimize(solver.Sum([poses_diff_array[i]*x_dict[i] for i in range(n_variables)]))
    status = solver.Solve()
    if status == pywraplp.Solver.OPTIMAL:
        print('Matched:')
        print('Objective value =', solver.Objective().Value())

        
    matches = np.zeros((N_people,N_people))
    for i in range(N_people):
        for j in range(N_people):
            index_match = index_matrix[i][j]
            print(index_match)
            if index_match!=-1:
                matches[i][j] = x_dict[index_match].solution_value()
    
    match_array = np.zeros(N_people)

    for i in range(N_people):
        for j in range(N_people):
            if matches[i][j] == 1:
                match_array[i] = j
    
    pid_new = []
    render_data = {}
    vertices_all_new = []

    pid_picked = []
    # optimize the pose and shape parameters of each pair of the people and their mirrored people 
    for i in range(N_people):
        if i not in pid_picked:
            bbox_pair = []
            keypoints2d_pair = []
            body_params_pair = []
            bbox_pair.append(bboxes[i])
            pair_id = int(match_array[i])
            bbox_pair.append(bboxes[pair_id])
            keypoints2d_pair.append(keypoints2d[i])
            keypoints2d_pair.append(keypoints2d[pair_id])
            body_params_pair.append(body_params_all[i])
            body_params_pair.append(body_params_all[pair_id])
            pid_new.append(pids[i])
            pid_new.append(pids[pair_id])
            pid_picked.append(pids[i])
            pid_picked.append(pids[pair_id])
            print("keypoints pair: {}".format(np.array(keypoints2d_pair).shape))
            bboxes_new = np.vstack(bbox_pair)
            keypoints2d_new = np.array(keypoints2d_pair)
            body_params_new = merge_params(body_params_pair)

            keypoints3d_new = body_model(return_verts=False,return_tensor=False,**body_params_new)
    # get the id of the real view
            real_id = simple_assign(keypoints3d_new)

            normal = None ## we do not use mirror normal constraint
            body_params = multi_stage_optimize(body_model, body_params_new, bboxes_new, keypoints2d_new, Pall=camera['P'], normal=normal, real_id=real_id, args=args)
            vertices = body_model(return_verts=True, return_tensor=False, **body_params)
            keypoints = body_model(return_verts=False, return_tensor=False, **body_params)

            vertices_all_new.append(vertices[0])
            vertices_all_new.append(vertices[1])

    
    if args.vis_smpl:
        print("pid_new: {}".format(pid_new))
        render_data = {pid_new[i]: {
            'vertices': vertices_all_new[i], 
            'faces': body_model.faces, 
            'vid': 0, 'name': 'human_{}'.format(pid_new[i])} for i in range(len(pid_new))}
        dataset.vis_smpl(render_data, image, camera)
    

if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument('path', type=str, help="the path of images")
    parser.add_argument('annot_path', type=str, help="the path of annotation folder")
    parser.add_argument('output_path', type=str, help="the path of output folder")
    parser.add_argument('--model', type=str, default='smpl', help="type of body model")
    parser.add_argument('--gender', type=str, default='neutral', help="the path of annotation folder")
    parser.add_argument('--vis_smpl', action='store_true', help='set to visualize the smpl result')
    args = parser.parse_args()

    #with Timer('Loading {}, {}'.format(args.model, args.gender)):
    body_model = load_model(args.gender, model_type=args.model)
    #Timer('Loading SPIN'):
    spin_model = SPIN(
            SMPL_MEAN_PARAMS='data/models/smpl_mean_params.npz', 
            checkpoint='data/models/spin_checkpoint.pt', 
            device=body_model.device)
    
    inputlist = sorted(os.listdir(args.path))
    for inp in inputlist:
        if '.jpg' in inp or '.png' in inp:
            demo(join(args.path,inp),args.annot_path,args.output_path,body_model,spin_model,args)
            
            
            