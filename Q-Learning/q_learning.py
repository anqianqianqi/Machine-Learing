import numpy as np
import time
import pandas as pd


from environment import MountainCar


def train(mode,weight_out,episodes,max_iterations,gamma,learning_rate,epsilon):
    environment = MountainCar( mode )
    # initialize the weight parameter w with 0 values, which has the shape of |A| x (|S|+1)
    # A -> {0,1,2}, 0 = pushing the car left; 1 = do nothing; 2 = pushing the car right
    w = np.zeros( (3,environment.state_space + 1) )
    for _ in range( episodes ):
        count = 0
        # initialize state vector s, fold the bias term 1 at index 0
        # reshape s into (|S|+1,1)
        state = np.append( [1],np.zeros( environment.state_space ) )
        state = state.reshape( -1,1 )
        # compute reward taking action a at state s based on curent parameters
        # q.shape = (|A|,1)
        q_compute_product = np.matmul( w,state )
        q_compute = np.max( q_compute_product )
        # take the action that gives the highest reward at state s
        action = np.argmax( q_compute_product )
        feedback_environment = environment.step( action )
        if feedback_environment[2]: print( feedback_environment )
        # do forever until the episode reaches the maximum iterations or the car reaches the top
        while not feedback_environment[2] and count < max_iterations:
            # at mode "raw"
            ##position,velocity =feedback_environment[0].values()
            ##state = np.array([1,position,velocity])
            gradient_w = np.zeros( (3,environment.state_space + 1) )
            gradient_w[action] = state.reshape( -1 )
            # derive new state vector from enviroment feedback
            if mode == 'raw':
                state = np.array(list( feedback_environment[0].values() ))
            elif mode =='tile':
                state = np.zeros( environment.state_space )
                state[list( feedback_environment[0].keys() )] = 1
            state = np.append( [1],state )
            state = state.reshape( -1,1 )
            # reward given by the environment by taking action a at state s
            q_environment = feedback_environment[1]
            #compute the maximum reward at new state s' with current parameter
            q_compute_product = np.matmul( w,state )
            q_compute_next_state = np.max( q_compute_product )
            # update parameter w
            w -= learning_rate * (q_compute - (q_environment+gamma*q_compute_next_state)) * gradient_w
            # compute new feedback with updated parameters at new state s`
            q_compute_product = np.matmul( w,state )
            q_compute = np.max( q_compute_product )
            if_explore = np.random.uniform(0,1)
            if if_explore<epsilon:
                #with probability epsilon, we explore by taking random step
                action = np.random.randint(0,3)
            else:
                #with probability 1- epsilon, we adopt greedy search strategy
                action = np.argmax( q_compute_product )
            feedback_environment = environment.step( action )
            count += 1
    w_DF = pd.DataFrame( w )
    w_DF.T.to_csv( weight_out )



if __name__ == "__main__":
    start_time = time.time()
    #mode = "tile"
    mode = "raw"
    weight_out = "weight_out.csv"
    return_out = "return_out.csv"
    #the number of epsidoes the program should train the agent for
    episodes = 10
    #the maximum of the length of an episode. When it is reached, we terminate the current episode
    max_iterations = 2000
    #the value for epsilon-greedy strategy
    epsilon = 0.05
    #the discount factor
    gamma = 0.99
    #the learning rate alpha for Q-learning algorithem
    learning_rate = 0.01
    train(mode,weight_out,episodes,max_iterations,gamma,learning_rate,epsilon)
    print("------{}minutes------".format((time.time()-start_time)/60.0))


