import numpy as np
from pathlib import Path
current_script_path = Path(__file__).resolve()
parent_dir = current_script_path.parent
array_folder= parent_dir/ "model_arrays"
test_trajectories=np.load(array_folder/"test_array.npy")
parent_dir = current_script_path.parent
noisy_path = parent_dir / "noisy_tests"
sigma_encoder = 0.01  #
mean_noise = 0.0

noise_encoder = np.random.normal(mean_noise, sigma_encoder, size=test_trajectories.shape)
noisy_encoder = test_trajectories + noise_encoder

sigma_motion_prediction = sigma_encoder * np.sqrt(50 / 49)

noise_motion_prediction = np.zeros_like(test_trajectories) 
noise_motion_prediction[:, :49,:] = np.random.normal(mean_noise, sigma_motion_prediction, size=(test_trajectories.shape[0], 49,2))

noisy_motion_prediction = test_trajectories + noise_motion_prediction
np.save(noisy_path/"noisy_test_array_encoder",noisy_encoder)
np.save(noisy_path/"noisy_test_array_motion_prediction",noisy_motion_prediction)