# original descrete GDICE
import numpy as np
from multiprocessing import Process, Queue
import random

def is_obs_in_list(obs, obs_list):
    is_in = False
    for i in range(len(obs_list)):
        if (obs == obs_list[i]).all():
            is_in = True
    return is_in

class AgentParam(object):
    # for each agent
    def __init__(self, N_node, N_obs, N_action):
        self.N_node = N_node
        self.N_obs = N_obs
        self.N_action = N_action
        self.trans_theta = np.ones((self.N_node, self.N_node, self.N_obs))  # (q, q', obs)
        self.normalize_trans_theta()
        self.emit_theta = np.ones((self.N_node, self.N_action))
        self.normalize_emit_theta()
        self.alpha = 0.1

    def sample_trans_mat(self):
        trans_mat = np.zeros((self.N_node, self.N_obs))
        for i in range(self.N_node):
            for j in range(self.N_obs):
                trans_mat[i, j] = np.random.choice(self.N_node, 1, p=self.trans_theta[i, :, j])
        return trans_mat

    def sample_emit_mat(self):
        emit_mat = np.zeros((self.N_node, 1))
        for i in range(self.N_node):
            emit_mat[i, 0] = np.random.choice(self.N_action, 1, p=self.emit_theta[i, :])
        return emit_mat

    def normalize_trans_theta(self):
        for m in range(self.N_obs):
            for k in range(self.N_node):
                sum = 0
                for l in range(self.N_node):
                    sum = sum + self.trans_theta[k, l, m]
                for l in range(self.N_node):
                    self.trans_theta[k, l, m] = self.trans_theta[k, l, m]/sum

    def normalize_emit_theta(self):
        for k in range(self.N_node):
            sum = 0
            for l in range(self.N_action):
                sum = sum + self.emit_theta[k, l]
            for l in range(self.N_action):
                self.emit_theta[k, l] = self.emit_theta[k, l] / sum

    def reset(self):
        self.trans_theta = np.ones((self.N_node, self.N_node, self.N_obs))  # (q, q', obs)
        self.normalize_trans_theta()
        self.emit_theta = np.ones((self.N_node, self.N_action))
        self.normalize_emit_theta()

    def update(self, trans_mat, emit_mat):
        # update trans_theta
        temp_trans_theta = np.zeros((self.N_node, self.N_node, self.N_obs))

        for i in range(self.N_node):
            for j in range(self.N_obs):
                temp_trans_theta[i, int(trans_mat[i, j]), j] = 1
        self.trans_theta = (1 - self.alpha) * self.trans_theta + self.alpha * temp_trans_theta
        self.normalize_trans_theta()

        # update emit_theta
        temp_emit_theta = np.zeros((self.N_node, self.N_action))
        for i in range(self.N_node):
                temp_emit_theta[i, int(emit_mat[i, 0])] = 1
        self.emit_theta = (1 - self.alpha) * self.emit_theta + self.alpha * temp_emit_theta
        self.normalize_emit_theta()

class ControllerSet(object):
    def __init__(self, N_agent, obs_dict, trans_mat_list, emit_mat_list, max_MC_iter, id):
        # N_agent, number of agents
        # obs_dict, [N_agent][obs_space][h, w, 3]
        # trans_mat_list, [N_agent][N_node, obs_space]
        # emit_mat_list, [N_agent][N_node, 0]
        self.id = id
        self.check_input_validity(N_agent, obs_dict, trans_mat_list, emit_mat_list)
        self.N_agent = N_agent
        self.states = []
        for i in range(self.N_agent):
            self.states.append(0)
        self.trans_mat_list = trans_mat_list
        self.emit_mat_list = emit_mat_list
        self.max_MC_iter = max_MC_iter
        self.obs_dict = obs_dict

    def check_input_validity(self, N_agent, obs_dict, trans_mat_list, emit_mat_list):
        if N_agent != len(trans_mat_list):
            print('length of trans_mat_list does not match with number of agents')
            print('number of agents is', N_agent)
            print('length of trans_mat_list is', len(trans_mat_list))
            return
        if N_agent != len(emit_mat_list):
            print('length of emit_mat_list does not match with number of agents')
            print('number of agents is', N_agent)
            print('length of emit_mat_list is', len(emit_mat_list))
            return
        if len(trans_mat_list) == 0:
            print('length of trans_mat_list is zero')
            return
        if len(emit_mat_list) == 0:
            print('length of emit_mat_list is zero')
            return
        if len(obs_dict[0]) != trans_mat_list[0].shape[1]:
            print('number of observation does not match with trans_mat_dim')
            print('number of observation is', len(obs_dict[0]))
            print('shape of trans_mat_list is', trans_mat_list[0].shape)
            return
        if trans_mat_list[0].shape[0] != emit_mat_list[0].shape[0]:
            print('trans_mat_dim does not match with emit_mat_dim')
            print('trans_mat_dim is', trans_mat_list[0].shape)
            print('emit_mat_dim is', emit_mat_list[0].shape)
            return

    def evaluate(self, env):
        env.reset()
        self.reset_state()
        acc_reward = 0
        for i in range(self.max_MC_iter):
            action_list = self.get_action()
            reward_list, done = env.step(action_list)

            obs_list = env.get_obs()
            self.state_transit(obs_list)
            if done:
                self.reset_state()
                env.reset()
            for sb in range(len(reward_list)):
                acc_reward = acc_reward + reward_list[sb]
        return acc_reward

    def get_obs_index(self, obs, obs_dict_set):
        index = -1
        for i in range(len(obs_dict_set)):
            if (obs == obs_dict_set[i]).all():
                index = i
        '''if index == -1:
            print('warning: unknown observation, state will transit to random node')'''
        # print('index', index)
        return index

    def get_action(self):
        action_list = []
        for i in range(self.N_agent):
            action_list.append(self.emit_mat_list[i][int(self.states[i]), 0])
        return action_list

    def state_transit(self, obs_list):
        N_node = self.trans_mat_list[0].shape[0]
        for i in range(self.N_agent):
            obs = obs_list[i]   # obs of i th agent
            index = self.get_obs_index(obs, self.obs_dict[i])
            if index == -1:
                self.states[i] = random.randint(0, N_node-1)
            else:
                self.states[i] = self.trans_mat_list[i][int(self.states[i]), index]

    def reset_state(self):
        self.states = []
        for i in range(self.N_agent):
            self.states.append(0)

class GDICEopt(object):
    def __init__(self, N_agent, N_node_list, obs_dict, N_action, N_s, N_b, max_opt_iter, max_MC_iter, env):
        self.N_agent = N_agent
        self.N_node_list = N_node_list
        self.obs_dict = obs_dict
        self.N_action = N_action
        self.agt_param_list = []
        self.b_value_list = []
        self.N_s = N_s
        self.N_b = N_b
        self.env_list = []
        for i in range(N_s):
            self.env_list.append(env)
        self.reset_env_list()
        self.max_opt_iter = max_opt_iter
        self.max_MC_iter = max_MC_iter
        self.best_ctrl = []
        self.best_reward = -1000000
        # agt_param_list, [N_agent]
        for i in range(self.N_agent):
            temp = AgentParam(self.N_node_list[i], len(self.obs_dict[i]), N_action)
            self.agt_param_list.append(temp)

        trans_mat_list = []
        emit_mat_list = []
        for j in range(self.N_agent):
            trans_mat_list.append(self.agt_param_list[j].sample_trans_mat())
            emit_mat_list.append(self.agt_param_list[j].sample_emit_mat())
        self.best_ctrl = ControllerSet(self.N_agent, self.obs_dict, trans_mat_list, emit_mat_list, self.max_MC_iter, 0)
        self.best_reward = self.best_ctrl.evaluate(self.env_list[0])
        self.reset_env_list()

    def optimize(self):
        for i in range(self.max_opt_iter):
            print('optimization', i)
            ctrl_set_list = []
            reward_list = []
            for i in range(self.N_s):
                reward_list.append(0)
            self.reset_env_list()

            # samples controller sets
            q = Queue()     # save the reward of each sample
            p_lst = []
            for k in range(self.N_s):   # each sample takes one process
                # sample a controller
                trans_mat_list = []
                emit_mat_list = []
                for j in range(self.N_agent):
                    temp = self.agt_param_list[j].sample_trans_mat()
                    trans_mat_list.append(temp)
                    emit_mat_list.append(self.agt_param_list[j].sample_emit_mat())
                ctrl_set = ControllerSet(self.N_agent, self.obs_dict, trans_mat_list, emit_mat_list, self.max_MC_iter, k)
                ctrl_set_list.append(ctrl_set)

                p = Process(target=self.evaluate, args=(q, ctrl_set, self.env_list[k],), daemon=False)
                p.start()
                p_lst.append(p)

            for p in p_lst:
                p.join()

            for i in range(q.qsize()):
                temp_pair = q.get()
                reward_list[temp_pair.id] = temp_pair.reward

            print(reward_list)

            # keep the best N_b controllers
            index_list = np.argsort(reward_list).tolist()
            index_list.reverse()
            temp_best_list = []
            for k in range(self.N_b):
                index = index_list[k]
                self.update_agt_param_list(ctrl_set_list[index])
                temp_best_list.append(reward_list[index])
            print(temp_best_list)
            # update global best
            if reward_list[index_list[0]] > self.best_reward:
                self.best_reward = reward_list[index_list[0]]
                self.best_ctrl = ctrl_set_list[index_list[0]]
                #self.save_model('GDICE_model_FindGoals')

            # record training curve
            print('global best', self.best_reward)
            self.b_value_list.append(self.best_reward)


            # update global
            self.update_agt_param_list(self.best_ctrl)


            # model collapse
            if self.is_list_even(temp_best_list):
                self.collapse()
                print('model collapse')

    def is_list_even(self, my_list):
        # check if all elements int the list are the same
        is_even = True
        for i in range(len(my_list)-1):
            if my_list[i] != my_list[i+1]:
                is_even = False
        return is_even

    def evaluate(self, q, ctrl_set, env):
        reward = ctrl_set.evaluate(env)
        temp = ID_Reward(ctrl_set.id, reward)
        q.put(temp)

    def update_agt_param_list(self, ctrl_set):
        for i in range(self.N_agent):
            self.agt_param_list[i].update(ctrl_set.trans_mat_list[i], ctrl_set.emit_mat_list[i])

    def reset(self):
        for i in range(self.N_agent):
            self.agt_param_list[i].reset()

        trans_mat_list = []
        emit_mat_list = []
        for j in range(self.N_agent):
            trans_mat_list.append(self.agt_param_list[j].sample_trans_mat())
            emit_mat_list.append(self.agt_param_list[j].sample_emit_mat())
        self.best_ctrl = ControllerSet(self.N_agent, self.obs_dict, trans_mat_list, emit_mat_list, self.max_MC_iter)
        self.best_reward = self.best_ctrl.evaluate(self.env)

        self.b_value_list = []

    def collapse(self):
        for i in range(self.N_agent):
            self.agt_param_list[i].reset()

    def save_model(self, name):
        # save the best controllerset
        for i in range(self.N_agent):
            best_trans_mat = self.best_ctrl.trans_mat_list[i]
            best_emit_mat = self.best_ctrl.emit_mat_list[i]
            np.save(name + '/best_trans_mat_' + str(i) + '.npy', best_trans_mat)
            np.save(name + '/best_emit_mat_' + str(i) + '.npy', best_emit_mat)

        # save the parameters
        for i in range(self.N_agent):
            trans_theta = self.agt_param_list[i].trans_theta
            emit_theta = self.agt_param_list[i].emit_theta
            np.save(name+ '/trans_theta_' + str(i) + '.npy', trans_theta)
            np.save(name + '/emit_theta_' + str(i) + '.npy', emit_theta)

        # save observation dictionary
        for i in range(self.N_agent):
            img_list = self.obs_dict[i][0]
            for j in range(1, len(self.obs_dict[i])):
                img_list = np.concatenate((img_list, self.obs_dict[i][j]), axis=2)
            np.save(name + '/obs_dict_' + str(i) + '.npy', img_list)

    def load_model(self, name):
        # load observation dictionary
        self.obs_dict = []
        for i in range(self.N_agent):
            img_list = np.load(name + '/obs_dict_' + str(i) + '.npy')
            img_num = int(img_list.shape[2] / 3)
            temp_list = []
            for j in range(img_num):
                temp_img = np.array(img_list[:, :, 3 * j:3 * j + 3])
                temp_list.append(img_list[:, :, 3 * j:3 * j + 3])
            self.obs_dict.append(temp_list)
        for i in range(self.N_agent):
            self.agt_param_list[i].N_obs = len(self.obs_dict[i])

        # load the parameters
        for i in range(self.N_agent):
            self.agt_param_list[i].trans_theta = np.load(name + '/trans_theta_' + str(i) + '.npy')
            self.agt_param_list[i].emit_theta = np.load(name + '/emit_theta_' + str(i) + '.npy')

        # load the best controllerset
        best_trans_mat_list = []
        best_emit_mat_list = []
        for i in range(self.N_agent):
            best_trans_mat = np.load(name + '/best_trans_mat_' + str(i) + '.npy')
            best_emit_mat = np.load(name + '/best_emit_mat_' + str(i) + '.npy')
            best_trans_mat_list.append(best_trans_mat)
            best_emit_mat_list.append(best_emit_mat)


        self.best_ctrl = ControllerSet(self.N_agent, self.obs_dict, best_trans_mat_list, best_emit_mat_list, self.max_opt_iter)
        self.best_reward = self.best_ctrl.evaluate(self.env)

        print('load  best controllerset finished')
        print('best controllerset', self.best_ctrl.trans_mat_list[0].shape)
        print('best controllerset', self.best_ctrl.trans_mat_list[1].shape)
        print('best controllerset', self.best_ctrl.emit_mat_list[0].shape)
        print('best controllerset', self.best_ctrl.emit_mat_list[1].shape)
        print('best_reward', self.best_reward)

    def reset_env_list(self):
        for i in range(self.N_s):
            self.env_list[i].reset()

    def save_train_curve(self):
        np.save("b_value_list.npy", np.array(self.b_value_list))

class ID_Reward(object):
    def __init__(self, id, reward):
        self.id = id
        self.reward = reward

