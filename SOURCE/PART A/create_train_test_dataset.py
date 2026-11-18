import numpy as np
from pathlib import Path
current_script_path = Path(__file__).resolve()
parent_dir = current_script_path.parent
arrays_path = parent_dir / "model_arrays"

#loads all datasets
train_tajectories = np.load('traj_train.npy')
train_labels = np.load('labels_train.npy')
test_trajectories = np.load('trajectories_test.npy')
test_labels = np.load('labels_test.npy')
val_trajectories = np.load('traj_val.npy')
val_labels = np.load('labels_val.npy')
with open('ids_test.txt', 'r', encoding='utf-8') as file:
    ids_test = np.array(file.read().split("\n"))
with open('ids_train.txt', 'r', encoding='utf-8') as file:
    ids_train = np.array(file.read().split("\n"))
with open('ids_val.txt', 'r', encoding='utf-8') as file:
    ids_val = np.array(file.read().split("\n"))
#adds all the trajectories and all the labels
trajectories=np.append(train_tajectories,test_trajectories,axis=0)
trajectories=np.append(trajectories,val_trajectories,axis=0)
labels=np.append(train_labels,test_labels,axis=0)
labels=np.append(labels,val_labels,axis=0)
ids=np.append(ids_train,ids_test,axis=0)
ids=np.append(ids,ids_val,axis=0)

print(trajectories.shape)
print(labels.shape)

#changes both datasets in the same random way
rng = np.random.default_rng()
ar=rng.permutation(len(trajectories))
trajectories=trajectories[ar]
labels=labels[ar]
ids=ids[ar]
cut=int(len(labels)*0.9)
train_array=trajectories[:cut]
test_array=trajectories[cut:]
print(test_array.shape)

#finds the largest value in each axis of the train dataset to normalize
max_values = train_array.max(axis=(0, 1))
min_values = train_array.min(axis=(0,1))
max_x = max(max_values[0],-min_values[0])
max_y = max(max_values[1],-min_values[1])

train_array[:,:,0]=train_array[:,:,0]/max_x
train_array[:,:,1]=train_array[:,:,1]/max_y
test_array[:,:,0]=test_array[:,:,0]/max_x
test_array[:,:,1]=test_array[:,:,1]/max_y

max_in_row =  np.abs(trajectories).max(axis=(1, 2))
print(max_in_row)

#creates an extra label for the speed
conditions = [
    max_in_row > 0.01, 
    max_in_row > 0.005,  
    max_in_row > 0.001,
    max_in_row > 0.0005,
    max_in_row > 0   
]

choices = [len(conditions)-i-1 for i in range(len(conditions))]
result = np.select(conditions, choices, default=0)
result = result.reshape(-1, 1)
result_train=result[:cut]
result_test=result[cut:]
ids_result_train=ids[:cut].tolist()
ids_result_test=ids[cut:].tolist()
with open(arrays_path/'ids_array_train.txt', 'w', encoding='utf-8') as file:
    file.write('\n'.join(ids_result_train))
with open(arrays_path/'ids_array_test.txt', 'w', encoding='utf-8') as file:
    file.write('\n'.join(ids_result_test))
np.save(arrays_path/'test_array.npy', test_array)
np.save(arrays_path/'train_array.npy', train_array)
np.save(arrays_path/'train_array_labels.npy', np.append(labels[:cut],result_train,axis=1))
np.save(arrays_path/'test_array_labels.npy',np.append(labels[cut:],result_test,axis=1))

#noisy tests
noise_rate=[0.0005,0.005,0.01]

#calculates the number that each label appears
target1=labels[:,0]
#target2=labels[:,1]
#target3=labels[:,2]
#target4=result
object_type_i_map=["vehicle","pedestrian","construction","riderless_bicycle","bus","background","static","cyclist","motorcyclist","unknown"]
# direction_i_map=["straight","u turn","left turn","right turn", "unspecified"]
# speed_i_map=["constant speed","decelerate","accelerate"]
#speed_i_map=[str(i) for i in range(10)]
unique_values, counts = np.unique(target1, return_counts=True)
for value, count in zip(unique_values, counts):
    print(f"Η τιμή {object_type_i_map[value]} εμφανίζεται {count} φορές")
print('\n')
# unique_values, counts = np.unique(target2, return_counts=True)
# for value, count in zip(unique_values, counts):
#     print(f"Η τιμή {direction_i_map[value]} εμφανίζεται {count} φορές")
# print('\n')
#unique_values, counts = np.unique(target4, return_counts=True)
#for value, count in zip(unique_values, counts):
#    print(f"Η τιμή {speed_i_map[value]} εμφανίζεται {count} φορές")
