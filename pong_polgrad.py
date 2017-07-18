#!/bin/python2
import gym
import numpy as np
from cv2 import cvtColor, COLOR_RGB2GRAY
from scipy.misc import imresize
import time
import itertools
import random
import sys
from collections import deque
from keras.models import Sequential
from keras.layers import Dense, Flatten, Activation
from keras.layers.convolutional import Conv2D
#from keras.layers.pooling import MaxPooling2D
#from keras.layers.normalization import BatchNormalization
#from keras.layers.advanced_activations import LeakyReLU
from keras.utils.generic_utils import get_custom_objects
from keras.initializers import glorot_normal, Constant
from keras.optimizers import RMSprop 
from keras import backend as K
from keras.callbacks import TensorBoard
from matplotlib import pyplot as plt

# Custom Activation: Softmax for the first 6 policy gradient neurons, and
#                    linear for the last state value finction neuron
def custom_activation(x):
    return K.concatenate([K.softmax(x[:, :-1]), x[:, -1:]], axis=-1)
get_custom_objects().update({'custom_activation': Activation(
                             custom_activation)})

# Custom Loss: Loss for policy gradients
# (0.5 * (V_p - Vs)^2)*alpha + (-log(p_a) * (R - Vs))*(1 - alpha); here alpha=0.5
def custom_loss(y_true, y_pred):
    return 0.5 * K.pow(y_true[:, -1] - y_pred[:, -1], 2) - K.squeeze(
           K.batch_dot(K.log(K.batch_dot(y_pred[:, :-1], y_true[:, :-1],
           axes=1)), K.expand_dims(y_true[:, -1] - y_pred[:, -1], axis=-1),
           axes=1), axis=-1)

# Neural Network Model
if 'model_pong.h5' in os.listdir('.'):
    model = load_model('model_pong.h5')
    print "\nLoaded model\n"
else:
    model = Sequential()
    model.add(Conv2D(32, input_shape=(4, 84, 84), kernel_size=8, data_format=\
                    'channels_first', activation='relu'))
    #model.add(LeakyReLU())
    #model.add(MaxPooling2D())
    #model.add(BatchNormalization())
    model.add(Conv2D(64, kernel_size=4, activation='relu'))
    #model.add(LeakyReLU())
    #model.add(MaxPooling2D())
    #model.add(BatchNormalization())
    model.add(Conv2D(64, kernel_size=2, activation='relu'))
    #model.add(LeakyReLU())
    #model.add(MaxPooling2D())
    #model.add(BatchNormalization())
    model.add(Flatten())
    model.add(Dense(128, kernel_initializer=glorot_normal(), bias_initializer=\
                    Constant(0.1), activation='relu'))
    #model.add(LeakyReLU())
    model.add(Dense(6 + 1, kernel_initializer=glorot_normal(), bias_initializer=\
                    Constant(0.001), activation='relu')) # First 6 for policy, last one for value fn.
    model.add(Activation(custom_activation))

    opt = RMSprop()
    model.compile(loss=custom_loss, optimizer=opt)

tb_callback = TensorBoard(log_dir='/home/rharish/Programs/Python/RL'\
                          '/Tensorboard/' + str(time.time()))

pong_experience = deque()

def pong_learn(num_episodes=20, exp_size=200000, discount_factor=0.99,
               save_model_every=1800, reset_fixed_every=5,
               exp_sample_size=32, save_video_every=5, tensorboard=True,
               render=False, verbose=True):

    env = gym.make('Pong-v4')
    env = gym.wrappers.Monitor(env, 'Videos', resume=True, video_callable=
                            lambda count: count % save_video_every == 0)

    experience = deque(maxlen=exp_size)
    if 'model_pong.data' in os.listdir('.'):
        print "\n\nLoading model experience..."
        exp_file = open('model_pong.data')
        exp = pickle.load(exp_file)
        exp_file.close()
        experience.extend(exp)
        print "Model experience loaded\nExperience size = %d\n" % len(
              experience)
    global pong_experience
    pong_experience = experience

    fixed_model = Sequential.from_config(model.get_config())
    now = time.time()

    for ep_num in range(num_episodes):
        state = deque(maxlen=4)
        state.append(imresize(cvtColor(env.reset(), COLOR_RGB2GRAY), (84, 84)).\
                     astype(np.uint8))
        done = False
        if ep_num % reset_fixed_every == 0:
            fixed_model.set_weights(model.get_weights())

        epoch = 0
        for step_num in itertools.count():
            if step_num < 3:
                action = env.action_space.sample()
            else:
                prediction = model.predict(np.array([state]), batch_size=1)
                action = np.random.choice(range(6), p=prediction[0][:-1])

            state_new, reward, done, _ = env.step(action)
            state_new = imresize(cvtColor(state_new, COLOR_RGB2GRAY), (84, 84)).\
                        astype(np.uint8)

            if len(state) < 4:
                state.append(state_new)
                if done:
                    break
                else:
                    continue

            experience.append((np.array(state), action, reward, np.array(
                               state_new)))
            state.append(state_new)

            # Experience Replay
            if len(experience) >= exp_sample_size:
                states, actions, rewards, states_new = zip(*random.sample(
                                                           experience,
                                                           exp_sample_size))

                states_newer = []
                for (exp_state, exp_state_new) in zip(states, states_new):
                    exp_state = exp_state[1:]
                    states_newer.append(np.append(exp_state, [exp_state_new],
                                                  axis=0))
                del states_new
                states_newer = np.array(states_newer)

                td_targets = rewards + discount_factor * fixed_model.predict(
                            states_newer, batch_size=exp_sample_size)[:, -1]
                to_predict = np.array([np.append(np.identity(6)[exp_action],
                                       td_target) for exp_action, td_target in\
                                       zip(actions, td_targets)])

                if tensorboard:
                    if verbose:
                        model.fit(np.array(states), to_predict, epochs=(
                                  epoch + 1), batch_size=exp_sample_size,
                                  callbacks=[tb_callback], initial_epoch=epoch)
                    else:
                        model.fit(np.array(states), to_predict, epochs=(
                                  epoch + 1), batch_size=exp_sample_size,
                                  callbacks=[tb_callback], verbose=0,
                                  initial_epoch=epoch)
                else:
                    if verbose:
                        model.fit(np.array(states), to_predict, epochs=(
                                  epoch + 1), batch_size=exp_sample_size,
                                  initial_epoch=epoch)
                    else:
                        model.fit(np.array(states), to_predict, epochs=(
                                  epoch + 1), batch_size=exp_sample_size,
                                  verbose=0, initial_epoch=epoch)
                epoch += 1

            if render:
                env.render()

            if not verbose:
                sys.stdout.write("\r%d steps taken\r" % (step_num + 1))
                sys.stdout.flush()

            if done:
                break
        
        if time.time() - now > save_model_every:
            print "\n\nSaving model..."
            model.save('model_pong.h5')
            exp_file = open('model_pong.data', 'wb')
            pickle.dump(experience, exp_file)
            exp_file.close()
            print "Model saved\n"
            now = time.time()

        sys.stdout.write("%d episode(s) done out of %d" % (ep_num + 1,
                                                           num_episodes))
        sys.stdout.flush()

    print "\n\nSaving model..."
    model.save('model_pong.h5')
    exp_file = open('model_pong.data', 'wb')
    pickle.dump(experience, exp_file)
    exp_file.close()
    print "Model saved\n"

try:
    pong_learn(exp_sample_size=16, tensorboard=False, verbose=False)
except KeyboardInterrupt:
    print "\n\nSaving model..."
    model.save('model_pong.h5')
    exp_file = open('model_pong.data', 'wb')
    pickle.dump(experience, exp_file)
    exp_file.close()
    print "Model saved\n"
except:
    print "\n\nSaving model..."
    model.save('model_pong.h5')
    exp_file = open('model_pong.data', 'wb')
    pickle.dump(experience, exp_file)
    exp_file.close()
    print "Model saved\n"
    raise

