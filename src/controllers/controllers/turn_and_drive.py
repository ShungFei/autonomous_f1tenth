import rclpy
import numpy as np
import random
from rclpy.impl import rcutils_logger
from .controller import Controller
from environments.util import get_euler_from_quarternion


def main():
    rclpy.init()
    
    param_node = rclpy.create_node('params')
    
    param_node.declare_parameters(
        '',
        [
            ('car_name', 'f1tenth_two'),
        ]
    )
    
    params = param_node.get_parameters(['car_name'])
    params = [param.value for param in params]
    CAR_NAME = params[0]
    
    controller = Controller('turn_drive_', CAR_NAME, 0.25)
    policy_id = 'turn_drive'
    policy = TurnAndDrive()

    # but index 5 seems to be quaternion angle??
    #odom: [position.x, position.y, orientation.w, orientation.x, orientation.y, orientation.z, lin_vel.x, ang_vel.z], lidar:...
    state = controller.get_observation(policy_id)


    
    # goalx = float(state[0]) + 2
    # goaly = float(state[1]) + 2
    goalx = -5
    goaly = -2
    goal = np.asarray([goalx, goaly])
    
    while True:
        # compute target [linear velocity, angular velocity]
        
        
        state = controller.get_observation(policy_id)
        action = policy.select_action(state, goal)   

        # moves car
        controller.step(action, policy_id)

class TurnAndDrive():

    logger = rcutils_logger.RcutilsLogger(name="tnd_log")

    def __init__(self, turning_lin_vel = 0.2, turning_ang_vel_modifier = 1, straight_lin_vel = 1, angle_diff_tolerance = 0.1):
        self.turning_lin_vel = turning_lin_vel
        self.turning_ang_vel_modifier = turning_ang_vel_modifier
        self.straight_lin_vel = straight_lin_vel
        self.angle_diff_tolerance = angle_diff_tolerance
        

    # Still fixing
    def select_action(self, state, goal):
        location = state[0:2]
        self_angle = get_euler_from_quarternion(state[2],state[3],state[4],state[5])[2]

        self.logger.info("-------------------------------------------------")
        # self.logger.info("LOCATION: "+str(location[0])+" "+str(location[1]))
        self.logger.info("STATE: "+str(location)+" "+str(self_angle))

        
        distance = goal - location
        if ((abs(distance[0]) < 0.2) and (abs(distance[1] < 0.2))):
            lin = 0
            ang = 0
            action = np.asarray([lin, ang])
            return action
        
        angle_to_goal = np.arctan2(distance[1], distance[0])
        if (((angle_to_goal - self_angle) > self.angle_diff_tolerance) or ((angle_to_goal - self_angle) < -self.angle_diff_tolerance)):
            ang = angle_to_goal - self_angle
            # print(state[5])
            # print(ang)
            self.logger.info("ANGLE TO GOAL: "+str(angle_to_goal))
            self.logger.info("SELF ANGLE: "+str(self_angle))
            self.logger.info("DELTA: "+str(ang))

            if ang > 1.5:
                ang = 1.5
            elif ang < -1.5:
                ang = -1.5

            lin = self.turning_lin_vel
        else:
            self.logger.info("ANGLE TO GOAL: "+str(angle_to_goal))
            ang = 0
            lin = self.straight_lin_vel
        
        
        self.logger.info("RESULT LIN_V: "+str(lin))
        self.logger.info("RESULT ANG_V: "+str(ang))
        action = np.asarray([lin, ang*self.turning_ang_vel_modifier])
        return action    

if __name__ == '__main__':
    main()