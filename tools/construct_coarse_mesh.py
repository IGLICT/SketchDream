import cv2
import numpy as np
import trimesh
import os
import argparse

def main():
    parser = argparse.ArgumentParser(description="Construct coarse 3D meshes")
    
    parser.add_argument('--radius', type=float, required=False, default=1.5, help='The rendering radius')
    parser.add_argument('--dis_min', type=float, required=False, default=0.2, help='The distance between the min plane and center of coarse mesh (radius scale)')
    parser.add_argument('--dis_max', type=float, required=False, default=0.2, help='The distance between the min plane and center of coarse mesh (radius scale)')
    parser.add_argument('--root_dir', type=str, required=False, default=None, help='The path of the root')
    parser.add_argument('--edit_mask_path', type=str, required=False, default=None, help='The path of the edit mask')
    
    args = parser.parse_args()
    
    radius = args.radius
    # build the front and back faces 
    depth_min = radius - radius * args.dis_min
    depth_max = radius + radius * args.dis_max
    
    # Read input parameters
    root_dir = args.root_dir
    mask_path = args.edit_mask_path
    dict_path = os.path.join(root_dir, 'edit_param.npy')
    obj_box_path = os.path.join(root_dir, 'mesh_coarse_box.obj')
    text_path = os.path.join(root_dir, 'mesh_coarse_box.text')

    with open(text_path, 'w') as file:
        print("depth_min:", depth_min, file=file)
        print("depth_max:", depth_max, file=file)

    img = cv2.imread(mask_path)
    img = cv2.resize(img, (512,512))
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)  # change into gray maps
    ret, binary = cv2.threshold(gray,10,255,cv2.THRESH_BINARY)  # change into binary maps
    contour, hierarchy = cv2.findContours(binary, cv2.RETR_TREE,cv2.CHAIN_APPROX_NONE)  # Find the contour of mask

    ouput_dict = np.load(dict_path, allow_pickle=True).item()

    depth_distance = 0.05

    depth_list= np.arange(depth_min, depth_max, depth_distance).tolist()

    plane_list = []
    for depth in depth_list:
        plane_list.append(depth * ouput_dict['rays_d'] + ouput_dict['rays_o'])
    
    #========================================================================================
    vectices_list = []
    f_all = None

    len_contours = len(contour)

    for contour_id in range(len_contours):
        num_points_before = len(vectices_list)

        # fit the mask contour
        epsilon = 1
        approx = cv2.approxPolyDP(contour[contour_id], epsilon, True)

        num_points = len(approx)
        # Build vertex set
        for depth_pc in plane_list:
            for index in range(num_points):
                x, y = approx[index][0]
                vectices_list.append(depth_pc[0, y, x, :])

        # build side faces
        face_list = []

        for depth_index in range(len(plane_list) - 1):
            face_depth_list = []
            for index in range(num_points):
                if index == num_points - 1:
                    face_depth_list.append([index, 0, index + num_points])
                    face_depth_list.append([0, 0 + num_points, index + num_points])
                else:
                    face_depth_list.append([index, index+1, index + num_points])
                    face_depth_list.append([index+1, index + 1 + num_points, index + num_points])
            face_list.append(np.array(face_depth_list)+depth_index*num_points)

        f = np.concatenate(face_list, axis=0)

        v_2D = approx.reshape(-1,2)
        
        num_point, _ = v_2D.shape
        if num_point < 3:
            continue
        
        from scipy.spatial import Delaunay
        tri = Delaunay(v_2D)
        f_front = tri.simplices

        # calculate barycentric coordinate
        p1 = v_2D[f_front[:,0]]
        p2 = v_2D[f_front[:,1]]
        p3 = v_2D[f_front[:,2]]
        p = (p1 + p2 + p3) / 3.0
        p = p.astype(int)
        # dilate the mask
        kernel = np.ones((5, 5), np.uint8)
        gray_dilate = cv2.dilate(gray, kernel, iterations = 1)
        mask_p = gray_dilate[p[:,1], p[:,0]]
        # remove the outside faces
        f_front = f_front[(mask_p > 125)]

        f_back = f_front + num_points * (len(depth_list)-1)
        if f_all is None:
            f_all = np.concatenate((f, f_front, f_back),axis=0)
        else:
            f_all = np.concatenate((f_all, f + num_points_before, f_front + num_points_before, f_back + num_points_before),axis=0)

    v = np.array(vectices_list)

    obj = trimesh.Trimesh(vertices = v, faces = f_all)
    obj.export(obj_box_path)

if __name__ == "__main__":
    main()

# python construct_coarse_mesh.py --radius 0.7 --root_dir ./outputs/test/test_bark@20240708-205633/save

