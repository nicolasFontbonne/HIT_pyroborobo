from pyroborobo import Pyroborobo, Controller, CircleObject
import numpy as np
from datetime import datetime
import os
import random
from tqdm import trange
import sys

class HITController(Controller):

    def __init__(self, wm):
        super().__init__(wm)
        self.rob = Pyroborobo.get()
        #print(dir(Controller))
        self.nb_inputs = 1 + self.nb_sensors * 3
        self.nb_outputs = 2
        self.weights = None
        self.weights_origin = None
        self.weights_ids = None
        self.nb_param = 0

        self.transfer_rate = 0.8
        #self.sigma = 0.01
        self.maturation_time = 400
        self.proba_mutation = 0 #1 / (150 * self.maturation_time)


        self.next_gen_in_it = 0
        self.reward = 0
        self.reward_array = np.zeros(self.maturation_time)
        self.reward_idx = 0

        self.reset()

    def reset(self):
        if self.weights is None:
            self.weights = np.random.uniform(-1, 1, (self.nb_inputs, self.nb_outputs))
            self.weights_origin = np.full((self.nb_inputs, self.nb_outputs),self.get_id())
            self.nb_param = np.prod(self.weights.shape)
            self.weights_ids = []
            for i in range(len(self.weights)):
                for j in range(len(self.weights[i])):
                    self.weights_ids += [[i, j]]
        self.reward = 0
        self.reward_array = np.zeros(self.maturation_time)
        self.reward_idx = 0
        #print(self.weights_ids)

    def step(self):
        self.next_gen_in_it += 1
        if self.next_gen_in_it > self.maturation_time:
            for i in range(self.nb_sensors):
                rob = self.get_robot_controller_at(i)
                if rob:
                    self.share_weights(rob)

        # Movement
        inputs = self.get_inputs()
        trans_speed, rot_speed = inputs @ self.weights
        self.set_translation(np.tanh(trans_speed))
        self.set_rotation(np.tanh(rot_speed))
      
        # Colors !
        self.set_color(0, 0, 255)

        self.reward_idx = (self.reward_idx + 1)%self.maturation_time
        self.reward_array[self.reward_idx] = 0 # clear 
        
        p = random.random()
        if p < self.proba_mutation:
            self.reset()
        

    def get_object(self):
        self.reward_array[self.reward_idx] = 1
        self.reward = np.sum(self.reward_array)

    def share_weights(self, rob):
        rob.receive_weights(self.reward, random.sample(self.weights_ids, int(self.transfer_rate*self.nb_param)), self.weights, self.weights_origin)
        #print(f"Hello {rob.id}, I am {self.id}")

    def receive_weights(self, reward, ids, weights, weights_origin):
        if self.reward < reward:
            for i in ids:
                self.weights[i] = weights[i]
                self.weights_origin[i] = weights_origin[i]
            self.next_gen_in_it = 0
            self.reward = reward
            self.reward_idx = 0
            self.reward_array = np.zeros(self.maturation_time)
            """
            for i in range(len(weights)):
                self.weights[i] += random.gauss(0, self.sigma)
            """
            
    def get_inputs(self):
        dists = self.get_all_distances()
        is_robots = self.get_all_robot_ids() != -1
        is_walls = self.get_all_walls()
        is_objects = self.get_all_objects() != -1

        robots_dist = np.where(is_robots, dists, 1)
        walls_dist = np.where(is_walls, dists, 1)
        objects_dist = np.where(is_objects, dists, 1)

        inputs = np.concatenate([[1], robots_dist, walls_dist, objects_dist])
        assert(len(inputs) == self.nb_inputs)
        return inputs


    def get_score(self):
        if self.next_gen_in_it < self.maturation_time:
            return self.reward
        else:
            self.reward = np.sum(self.reward_array)
            return self.reward


    def inspect(self, prefix=""):
        output = "inputs: \n" + str(self.get_inputs()) + "\n\n"
        #output += "received weights from: \n"
        #output += str(list(self.received_weights.keys()))
        return output

class ResourceObject(CircleObject):
    def __init__(self, id_, data):
        CircleObject.__init__(self, id_)  # Do not forget to call super constructor
        self.regrow_time = 10
        self.cur_regrow = 0
        self.triggered = False
        #self.relocate = True
        self.rob = Pyroborobo.get()  # Get pyroborobo singleton
        #print(dir(ResourceObject))

    def reset(self):
        self.show()
        self.register()
        self.triggered = False
        self.cur_regrow = 0

    def step(self):
        #self.stepPhysicalObject()
        #print(dict(ResourceObject))
        if self.triggered:
            self.cur_regrow -= 1
            if self.cur_regrow <= 0:
                self.show()
                self.register()
                #self.relocate = False
                self.triggered = False

    def is_walked(self, rob_id):
        self.rob.controllers[rob_id].get_object()
        self.hide()
        self.relocate()
        self.unregister()
        self.triggered = True
        self.cur_regrow = self.regrow_time


    def inspect(self, prefix=""):
        return f"""I'm a ResourceObject with id: {self.id}"""


def main():
    rob = Pyroborobo.create("config/hit.properties",
                            controller_class=HITController, 
                            object_class_dict={'_default': ResourceObject})
    rob.start()
    rob.update(100000)
    rob.close()

def launch_exp(log_folder):
    len_exp = 15000
    sample_period = 1
    
    #rob.get("gInitialNumberOfRobots")

    rob = Pyroborobo.create("config/hit.properties",
                            controller_class=HITController, 
                            object_class_dict={'_default': ResourceObject})
    rob.start()
    #print(dir(rob))

    
    nb_param = rob.controllers[0].nb_param
    weights_ids = rob.controllers[0].weights_ids
    #print(weights_ids)


    now = datetime.now()
    date = now.strftime("%Y-%m-%d-%H-%M-%S-%f")
    folder = log_folder+ f"hit_{date}"
    os.mkdir(folder)
    score_file = open(folder+"/score.csv", "w")
    sample_time_file = open(folder+"/sample_time.csv", "w")
    sample_time_parameters_file = open(folder+"/sample_time_parameters.csv", "w")
    origin_parameters_file = []
    value_parameters_file = []
    for i in range(nb_param):
        origin_parameters_file.append(open(folder+f"/origin_parameter_{i}.csv", "w"))
        value_parameters_file.append(open(folder+f"/value_parameters_{i}.csv", "w"))
    position_x_file = open(folder+"/position_x.csv", "w")
    position_y_file = open(folder+"/position_y.csv", "w")

    object_position_x_file = open(folder+"/object_position_x.csv", "w")
    object_position_y_file = open(folder+"/object_position_y.csv", "w")


    for i in trange(0, len_exp, sample_period):
        rob.update(sample_period)
        sample_time_file.write(f"{i},")
        if i%100 == 0:
            sample_time_parameters_file.write(f"{i},")
        for k,ressource_object in enumerate(rob.objects):
            object_position_x_file.write(f"{ressource_object.position[0]}")
            object_position_y_file.write(f"{ressource_object.position[1]}")
            if k != len(rob.objects)-1:
                object_position_x_file.write(',')
                object_position_y_file.write(',')
        object_position_x_file.write('\n')
        object_position_y_file.write('\n')
        for k,robot_controller in enumerate(rob.controllers):
            #print(dir(robot_controller))
            score_file.write(f"{robot_controller.get_score()}")
            if i%100 == 0:
                for j in range(nb_param):
                    origin_parameters_file[j].write(f"{robot_controller.weights_origin[weights_ids[j][0], weights_ids[j][1]]}")
                    value_parameters_file[j].write(f"{robot_controller.weights[weights_ids[j][0], weights_ids[j][1]]}")

            position_x_file.write(f"{robot_controller.absolute_position[0]}")
            position_y_file.write(f"{robot_controller.absolute_position[1]}")
            if k != len(rob.controllers)-1:
                score_file.write(',')
                if i%100 == 0:
                    for j in range(nb_param):
                        origin_parameters_file[j].write(',')
                        value_parameters_file[j].write(',')
                position_x_file.write(',')
                position_y_file.write(',')
        score_file.write("\n")
        if i%100 == 0:
            for j in range(nb_param):
                origin_parameters_file[j].write("\n")
                value_parameters_file[j].write("\n")
        position_x_file.write('\n')
        position_y_file.write('\n')

    
    rob.close()
    score_file.close()
    sample_time_file.close()
    for i in range(nb_param):
        origin_parameters_file[i].close()
    #origin_parameter_0_file.close()
    position_x_file.close()
    position_y_file.close()

if __name__ == "__main__":
    if len(sys.argv) <= 1:
        folder = 'logs/'
    else:
        folder = sys.argv[1]
    launch_exp(folder)
