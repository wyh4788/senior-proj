import gym
import numpy as np

def downsample(image):
    # Take only alternate pixels - basically halves the resolution of the image (which is fine for us)
    return image[::2, ::2, :]

def remove_color(image):
    """Convert all color (RGB is the third dimension in the image)"""
    return image[:, :, 0]

def remove_background(image):
    image[image == 144] = 0
    image[image == 109] = 0
    return image

def preprocess_observations(input_observation, prev_processed_observation, input_dimensions):
    """ convert the 210x160x3 uint8 frame into a 6400 float vector """
    processed_observation = input_observation[35:195] # crop
    processed_observation = downsample(processed_observation)
    processed_observation = remove_color(processed_observation)
    processed_observation = remove_background(processed_observation)
    processed_observation[processed_observation != 0] = 1 # everything else (paddles, ball) just set to 1
    # Convert from 80 x 80 matrix to 1600 x 1 matrix
    processed_observation = processed_observation.astype(np.float).ravel()

    # subtract the previous frame from the current one so we are only processing on changes in the game
    if prev_processed_observation is not None:
        input_observation = processed_observation - prev_processed_observation
    else:
        input_observation = np.zeros(input_dimensions)
    # store the previous frame so we can subtract from it next time
    prev_processed_observations = processed_observation
    return input_observation, prev_processed_observations


def sigmoid(x):
    return 1.0/(1.0 + np.exp(-x))

def relu(vector):
    vector[vector < 0] = 0
    return vector

def apply_neural_nets(observation_matrix, weights):
    """ Based on the observation_matrix and weights, compute the new hidden layer values and the new output layer values"""
    hidden_layer_values = np.dot(weights['1'], observation_matrix)
    hidden_layer_values = relu(hidden_layer_values)
    output_layer_values = np.dot(hidden_layer_values, weights['2'])
    output_layer_values = sigmoid(output_layer_values)
    return hidden_layer_values, output_layer_values

def choose_action(probability):
    if probability >= 0.5:
        # signifies up in openai gym
        return 2
    else:
         # signifies down in openai gym
        return 3


def main():
    env = gym.make("Pong-v0")
    observation = env.reset() # This gets us the image

    # hyperparameters
    episode_number = 0
    batch_size = 10
    gamma = 0.99 # discount factor for reward
    decay_rate = 0.99
    num_hidden_layer_neurons = 200
    input_dimensions = 80 * 80
    learning_rate = 1e-4
    output_reward = []

    episode_number = 0
    reward_sum = 0
    running_reward = None
    prev_processed_observations = None

    # w1 = np.loadtxt("weight1-1.out")
    # w2 = np.loadtxt("weight2-1.out")

    w1 = np.loadtxt("weight1-2.out")
    w2 = np.loadtxt("weight2-2.out")
    # w1 = np.loadtxt("weight1-3.out")
    # w2 = np.loadtxt("weight2-3.out")

    weights = {
        '1': w1,
        '2': w2
    }

    while True:
        env.render()
        processed_observations, prev_processed_obcdservations = preprocess_observations(observation, prev_processed_observations, input_dimensions)
        hidden_layer_values, up_probability = apply_neural_nets(processed_observations, weights)

        action = choose_action(up_probability)

        # carry out the chosen action
        observation, reward, done, info = env.step(action)

        reward_sum += reward


        if done: # an episode finished
            episode_number += 1

            observation = env.reset() # reset env
            running_reward = reward_sum if running_reward is None else running_reward * 0.99 + reward_sum * 0.01
            output_reward.append(running_reward)
            print 'resetting env. episode reward total was %f. running mean: %f. num: %i' % (reward_sum, running_reward, episode_number)


            reward_sum = 0
            prev_processed_observations = None

            if episode_number == 100:
                break

main()
