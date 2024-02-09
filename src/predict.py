import os, sys
import pickle as pkl
import numpy as np
# from utils.data_handler_v3_tfrecord import expand_sequence
from utils.model_utils import model_selector
from scipy.io import loadmat, savemat

import matplotlib.pyplot as plt
import datetime


model_name = "dynamicEllipsoidObstaclesRNN_commonInputMaxPooling_alt"
model_number = 511

class trajectory_predictor():
    def __init__(self, n_robots, n_obstacles, beta, past_horizon=10, prediction_horizon=20, dt=0.05, model_name = model_name, model_number = model_number):
        self.n_robots = n_robots
        self.n_obstacles = n_obstacles
        self.past_horizon = past_horizon
        self.prediction_horizon = prediction_horizon
        self.dt = dt
        
        root_dir= os.path.dirname(sys.path[0])
        model_dir = os.path.join(root_dir, "trained_models", model_name, str(model_number))
        parameters_path = os.path.join(model_dir, "model_parameters.pkl")
        checkpoint_path = os.path.join(model_dir, "model_checkpoint.h5")
        
        assert os.path.isfile(checkpoint_path)    
        args = pkl.load( open( parameters_path, "rb" ) )
        args.prediction_horizon = self.prediction_horizon
        
        assert "scaler" in args
        self.scaler = args.scaler
        
        self.input_data = {}
        if beta == 0:
            self.input_data["beta"] = np.zeros((self.n_robots, self.past_horizon, 1))
        else:
            self.input_data["beta"] = np.ones((self.n_robots, self.past_horizon, 1))
        # self.input_data["beta"] = np.zeros((1, self.past_horizon, 1))
        # self.input_data["beta"] = np.ones((self.n_robots, self.past_horizon, 1))
        # self.input_data["query_input"] = np.zeros((1, self.past_horizon, 3))
        self.input_data["query_input"] = np.zeros((self.n_robots, self.past_horizon, 3))
        if self.n_robots > 1:
            self.input_data["others_input"] = np.zeros((n_robots, self.past_horizon, 6, 1))
        else:
            args.others_input_type = "none"
        if self.n_obstacles > 0:
            self.input_data["obstacles_input"] = np.zeros((self.n_robots, 6, self.n_obstacles))
        else:
            args.obstacles_input_type = "none"

        self.model = model_selector(args)
        self.model.call(self.input_data)
        self.model.built = True
        self.model.load_weights(checkpoint_path)
        
        if self.model.stateful:
            for i in range(len(self.model.layers)):
                self.model.layers[i].stateful = False
        
    def predict(self, robot_data, obstacle_data):
        
        for query_quad_idx in range(self.n_robots):
        # for query_quad_idx in range(1):
            other_quad_idxs = [idx for idx in range(self.n_robots) if idx != query_quad_idx]
            
            self.input_data["query_input"][query_quad_idx] = np.transpose( robot_data[3:6, : , query_quad_idx] )
            
            if self.n_robots > 1:
                self.input_data["others_input"][query_quad_idx] = np.moveaxis( robot_data[0:6, : , other_quad_idxs] - robot_data[0:6, :, query_quad_idx:query_quad_idx+1], 0, 1)
            
            # if self.n_obstacles > 0:
            #     self.input_data["obstacles_input"][query_quad_idx] = obstacle_data - robot_data[0:6, -1, query_quad_idx:query_quad_idx+1]
        
        # scaled_data = self.scaler.transform(self.input_data)
        scaled_data = self.input_data
        
        scaled_data["target"] = self.model.predict(scaled_data)
        # vel_prediction = self.scaler.inverse_transform(scaled_data)["target"]
        vel_prediction = scaled_data["target"]
        
        pos_prediction = np.zeros((self.n_robots, self.prediction_horizon+1, 3))
        pos_prediction[:, 0, :] = np.transpose(robot_data[0:3, -1 , :])
        for step in range(1, self.prediction_horizon+1):
            pos_prediction[:, step, :] = pos_prediction[:, step-1, :] + self.dt * vel_prediction[:, step-1, :]
        
        return np.swapaxes(pos_prediction[:, 1:, :], 0, -1)
        
        
root_dir = os.path.dirname(sys.path[0])
data_master_dir = os.path.join(root_dir, "data", "")
raw_data_dir = os.path.join(data_master_dir, "Raw", "")  
dataset_name = "hr_pred_10_B0_3"      
beta = 1
raw_dataset_path = os.path.join(raw_data_dir, dataset_name + '.mat')

   
past_horizon = 10
pred_horizon = 10
        
data = loadmat(raw_dataset_path)
robot_data = data['log_quad_state_real']



human_positions = robot_data[ 0:2, : , 0]
robot_positions = robot_data[ 0:2, : , 1]

past_human_positions = human_positions[:, 0:past_horizon]
past_robot_positions = robot_positions[:, 0:past_horizon]
ground_truth_human_future = human_positions[:, past_horizon:past_horizon+pred_horizon]

past_robot_data = robot_data[ :, 0:past_horizon , :]
# print(robot_data)

prediction_network_b0 = trajectory_predictor(2, 0, 0, past_horizon, pred_horizon, 0.2)
prediction_network_b1 = trajectory_predictor(2, 0, 1, past_horizon, pred_horizon, 0.2)
res_b0 = prediction_network_b0.predict(past_robot_data, 0)
res_b1 = prediction_network_b1.predict(past_robot_data, 0)

predicted_human_positions_b0 = res_b0[0:2, : , 0]
predicted_human_positions_b1 = res_b1[0:2, : , 0]
        
# plt.plot(past_robot_positions[0], past_robot_positions[1], marker='o', linestyle='-', color='black', label='Past Robot Positions')
plt.plot(past_human_positions[0], past_human_positions[1], marker='o', linestyle='-', color='magenta', label='Past Human Positions')


plt.plot(ground_truth_human_future[0], ground_truth_human_future[1], marker='o', linestyle='-', color='red', label='Ground Truth Human Future Position')


plt.plot(predicted_human_positions_b0[0], predicted_human_positions_b0[1], marker='o', linestyle='-', color='blue', label='Predicted Human Positions B = 0')
plt.plot(predicted_human_positions_b1[0], predicted_human_positions_b1[1], marker='o', linestyle='-', color='green', label='Predicted Human Positions B = 1')

# Set plot labels and title
plt.xlabel('X-axis')
plt.ylabel('Y-axis')
plt.title(dataset_name)
plt.legend()
# plt.xlim(0, 10)
# plt.ylim(0, 10)

# Save the plot with the current date-time as the file name
current_datetime = datetime.datetime.now().strftime("%Y-%m-%d_%H-%M-%S")
plt.savefig(f'{dataset_name}_{current_datetime}.png')

# Show the plot
plt.show()



