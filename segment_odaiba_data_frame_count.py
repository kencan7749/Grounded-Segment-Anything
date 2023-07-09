import os, sys
from glob import glob 

data_name = "rosbag2_2023_03_03-14_28_12_10"
camera_name = 'c2'

data_name_list = ["rosbag2_2023_03_03-13_40_02_0",
                  "rosbag2_2023_03_03-13_40_02_1",
                  "rosbag2_2023_03_03-13_40_02_2",
                  "rosbag2_2023_03_03-13_40_02_3",
                  "rosbag2_2023_03_03-13_40_02_4",
                  "rosbag2_2023_03_03-13_40_02_5",
                  "rosbag2_2023_03_03-13_40_02_6",
                  "rosbag2_2023_03_03-13_40_02_7",
                  "rosbag2_2023_03_03-13_40_02_8",
                  
                  "rosbag2_2023_03_03-13_54_05_0",
                  "rosbag2_2023_03_03-13_54_05_1",
                  "rosbag2_2023_03_03-13_54_05_2",
                  "rosbag2_2023_03_03-13_54_05_3",
                  "rosbag2_2023_03_03-13_54_05_4",
                  "rosbag2_2023_03_03-13_54_05_5",
                  "rosbag2_2023_03_03-13_54_05_6",
                  "rosbag2_2023_03_03-13_54_05_7",
                  "rosbag2_2023_03_03-13_54_05_8",
                  "rosbag2_2023_03_03-13_54_05_9",
                  "rosbag2_2023_03_03-13_54_05_10",
                  "rosbag2_2023_03_03-13_54_05_11",
                  "rosbag2_2023_03_03-13_54_05_12",
                  
                  "rosbag2_2023_03_03-14_46_14_0",
                  "rosbag2_2023_03_03-14_46_14_1",
                  "rosbag2_2023_03_03-14_46_14_2",
                  "rosbag2_2023_03_03-14_46_14_3",
                  "rosbag2_2023_03_03-14_46_14_4",
                  "rosbag2_2023_03_03-14_46_14_5",
                  "rosbag2_2023_03_03-14_46_14_6",
                  "rosbag2_2023_03_03-14_46_14_7",
                  "rosbag2_2023_03_03-14_46_14_8",
                  "rosbag2_2023_03_03-14_46_14_9",
                  "rosbag2_2023_03_03-14_46_14_10",
                  "rosbag2_2023_03_03-14_46_14_11",
                  
                  
                  
                  
                  ]
camera_name_list = ['c1']

for data_name in data_name_list:
    for camera_name in camera_name_list:
        result_bbox_dir = f'results/bbox/png/{data_name}/{camera_name}'
        result_bbox_concat_dir = f'results/bbox_concat/png/{data_name}/{camera_name}'
        #os.makedirs(result_bbox_dir, exist_ok=True)
        #os.makedirs(result_bbox_concat_dir, exist_ok=True)

        result_dir = f'results/ann/png/{data_name}/{camera_name}'
        result_concat_dir = f'results/ann_concat/png/{data_name}/{camera_name}'
        #os.makedirs(result_dir, exist_ok=True)
        #os.makedirs(result_concat_dir, exist_ok=True)



        # image path

        image_paths = sorted(glob(f'../../data/odaiba_data/results/{data_name}/{camera_name}/image_raw/compressed/*.png'))
        
        print(f'{data_name}')
        print( len(image_paths))
        
        assert len(image_paths) >0
    