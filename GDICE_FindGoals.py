from env_FindGoals import EnvFindGoals
from GDICE_MP import GDICEopt, is_obs_in_list
import random
import matplotlib.pyplot as plt


if __name__ == '__main__':
    # running GDICE
    alpha = 0.05
    N_agent = 2
    N_s = 20
    N_b = 5     # no bigger than N_s
    N_action = 5
    N_node_list = [10, 10]
    max_opt_iter = 100
    max_MC_iter = 300

    # get obs_dict
    obs_list1 = []
    obs_list2 = []
    env = EnvFindGoals()
    print('start')
    for j in range(100):
        print(j)
        env.reset()
        for i in range(100):
            reward_list, done = env.step([random.randint(0, 4), random.randint(0, 4)])
            obs1 = env.get_obs()[0]
            obs2 = env.get_obs()[1]
            if is_obs_in_list(obs1, obs_list1) == False:
                obs_list1.append(obs1)
            if is_obs_in_list(obs2, obs_list2) == False:
                obs_list2.append(obs2)
        print(len(obs_list1))
        print(len(obs_list2))
    obs_dict = [obs_list1, obs_list2]
    print('here')

    #obs_dict = [[0], [0], [0]]
    env = EnvFindGoals()
    optimizer = GDICEopt(N_agent, N_node_list, obs_dict, N_action, N_s, N_b, max_opt_iter, max_MC_iter, env)
    #optimizer.load_model('GDICE_model_FindGoals')

    optimizer.optimize()
    best_ctrl = optimizer.best_ctrl
    # optimizer.save_train_curve()
    env.reset()
    best_ctrl.reset_state()
    for i in range(max_MC_iter):
        action_list = best_ctrl.get_action()
        reward_list, done = env.step(action_list)
        obs_list = env.get_obs()

        best_ctrl.state_transit(obs_list)
        if done:
            best_ctrl.reset_state()
            env.reset()
        env.plot_scene()

        '''# save global plot
        plt.figure()
        full_obs = env.get_full_obs()
        plt.imshow(full_obs)
        plt.savefig('/home/shuo/Figures/' + str(i) + '.png')
        plt.close()'''
