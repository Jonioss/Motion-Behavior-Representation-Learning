import pandas as pd
import numpy as np

def transform_dataset(input_file,trajectories_file,labels_file,ids_file):
    '''splits the dataset into three files which it returns. 
        -The trajectories_file is a .npy file with the unnormalized trajectories for model training
        -The labels_file is a .npy file with the encoded labels
        -The ids_file is a .txt file with the id of the trajectory
        In each file each row will correspond to the same row of the other files
    '''

    df=pd.read_parquet(input_file)
    df = df.sort_values(by=['scenario_id','track_id', 'timestep'])
    df["whole_id"] =str(0)+ df['scenario_id']+df["track_id"] #creates one value as an id
    
    #finds the x,y values of each track in timestep 0.0 in order to keep the difference of each timestep to the respective position in t=0.0
    baseline_df = df[(df["track_id"] == 'AV') & (df["timestep"] == 0.0)]

    x_norm_map = baseline_df.set_index("scenario_id")["position_x"]
    y_norm_map = baseline_df.set_index("scenario_id")["position_y"]

    #x_y_values=df.copy() #comment in for the full dataset 
    x_y_values = df[df["track_id"] != 'AV'].copy() #comment in for the dataset without AV
    
    x_y_values["x_norm"] = x_y_values["scenario_id"].map(x_norm_map)
    x_y_values["y_norm"] = x_y_values["scenario_id"].map(y_norm_map)

    x_y_values["position_x"] = (x_y_values["position_x"] - x_y_values["x_norm"]) 
    x_y_values["position_y"] = (x_y_values["position_y"] - x_y_values["y_norm"])

    x_y_values=x_y_values[["object_type","position_x","position_y","timestep","direction","speed_label","whole_id"]]#keeps only certain columns
    
    #for classes with few tracks it adds rotated tracks
    add_to_df=x_y_values[x_y_values['object_type'].isin(['construction','riderless_bicycle','cyclist','motorcyclist','unknown'])]
    add_to_df1=x_y_values[x_y_values['object_type'].isin(['construction','riderless_bicycle','cyclist','motorcyclist','unknown'])]
    add_to_df1['position_x']=add_to_df['position_y']
    add_to_df1['position_y']=-add_to_df['position_x']
    add_to_df1['whole_id']=str(1)+add_to_df['whole_id'][1:]
    add_to_df2=x_y_values[x_y_values['object_type'].isin(['construction','riderless_bicycle','cyclist','motorcyclist','unknown'])]
    add_to_df2['position_x']=-add_to_df['position_x']
    add_to_df2['position_y']=-add_to_df['position_y']
    add_to_df2['whole_id']=str(2)+add_to_df['whole_id'][1:]
    add_to_df3=x_y_values[x_y_values['object_type'].isin(['construction','riderless_bicycle','cyclist','motorcyclist','unknown'])]
    add_to_df3['position_x']=add_to_df['position_y']
    add_to_df3['position_y']=-add_to_df['position_x']
    add_to_df3['whole_id']=str(3)+add_to_df['whole_id'][1:]

    full_list=[x_y_values,add_to_df1,add_to_df3,add_to_df2]
    full=pd.concat(full_list)
    trajectories=full.drop(columns=['object_type','speed_label','direction','timestep',"whole_id"])
    print(trajectories.shape)
    num_trajectories = len(trajectories) // 50
    trajectories = trajectories.values.reshape(num_trajectories, 50, 2).astype(np.float32) 
    origin = trajectories[:, [0], :] 
    trajectories -= origin
    print(trajectories.shape)
    np.save(trajectories_file, trajectories)

    #encodes each label string into an int for compression
    object_type_map={"vehicle":0,"pedestrian":1,"construction":2,"riderless_bicycle":3,"bus":4,"background":5,"static":6,"cyclist":7,"motorcyclist":8,"unknown":9}
    direction_map={"straight":0,"u turn":1,"left turn":2,"right turn":3, "unspecified":4}
    speed_map={"constant speed":0,"decelerate":1,"accelerate":2}
   
    #labels_df = full[full['timestep']==0.0]

    labels_df = x_y_values[x_y_values['timestep']==0.0]
    labels_df['object_type_encoded'] = labels_df['object_type'].map(object_type_map)
    labels_df['direction_encoded'] = labels_df['direction'].map(direction_map)
    labels_df['speed_encoded'] = labels_df['speed_label'].map(speed_map)
    print(labels_df.shape)
    np.save(labels_file, labels_df[["object_type_encoded","direction_encoded","speed_encoded"]])
    labels_df['whole_id'].to_csv(ids_file, index=False, header=False)

if __name__ == "__main__":
    transform_dataset("output_test.parquet","traj_test.npy","labels_test.npy","ids_test.txt")
    transform_dataset("output_train.parquet","traj_train.npy","labels_train.npy","ids_train.txt")
    transform_dataset("output_val.parquet","traj_val.npy","labels_val.npy","ids_val.txt")
