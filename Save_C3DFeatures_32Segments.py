import os
import glob
import numpy as np
C3D_path = "/features_10thMar/Abnormal_features/"
C3D_path_seg = "/features10thMar_32/Abnormal_features/"
if not os.path.exists(C3D_path_seg):
    os.mkdir(C3D_path_seg)


for feature_file_path in glob.glob(C3D_path + "*.npy"):

    filename = feature_file_path.split('/')[-1].split('.')[0]
    #print(filename)
    feature = np.load(feature_file_path)
    #print("length of features",len(feature))
    if len(feature) == 0:
        print("error in video",filename)
        continue
    #print("feature shape", feature.shape)
    segment_feature = np.zeros((32, 4096))
    segment32 = np.round(np.linspace(0,len(feature) - 1, 32))
    for index in range(0, len(segment32)):
        #print("index", index)
        start = int(segment32[index])
        #print("lenght of segment32", len(segment32))
        if index == len(segment32) - 1:
            end = int(segment32[index])
        else:
            end = int(segment32[index +1])
        assert end >= start

        if start == end:
            temp_vect = feature[start, :]
        elif end - start == 1:
            temp_vect = feature[start, :]
        else:
            temp_vect = np.mean(feature[start:end, :], axis=0)
        temp_vect = temp_vect/np.linalg.norm(temp_vect)
        if np.linalg.norm(temp_vect) == 0:
            print("Error")
        segment_feature[index, :] = temp_vect

    output_name = filename + "_32" + ".npy"
    #print(segment_feature)
    #print(len(np.array(segment_feature)))
    np.save(C3D_path_seg + output_name, np.array(segment_feature))







