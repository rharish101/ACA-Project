#!/bin/python2
import gym
import numpy as np
from cv2 import cvtColor, COLOR_RGB2GRAY
from scipy.misc import imresize
from datetime import datetime
import time
import itertools
import random
import os
import sys
from shutil import copy
import pickle
from collections import deque
from keras.models import Sequential, load_model
from keras.layers import Dense, Flatten
from keras.layers.convolutional import Conv2D
from keras.utils.generic_utils import get_custom_objects
from keras.initializers import glorot_normal, Constant
from keras.optimizers import RMSprop 
from keras import backend as K
from keras.callbacks import TensorBoard

# Custom Loss: Loss for DQN using MSE
def custom_loss(y_true, y_pred):
    return K.square((K.squeeze(K.batch_dot(y_true, y_pred, axes=1),
           axis=-1) / K.sum(y_true, axis=-1)) - K.max(y_true, axis=-1))

# Neural Network Model
if 'model_pong.h5' in os.listdir('.'):
    model = load_model('model_pong.h5')
    print "\nLoaded model\n"
else:
    model = Sequential()
    model.add(Conv2D(32, input_shape=(4, 84, 84), kernel_size=8, data_format=\
                    'channels_first', activation='relu'))
    model.add(Conv2D(64, kernel_size=4, activation='relu'))
    model.add(Conv2D(64, kernel_size=2, activation='relu'))
    model.add(Flatten())
    model.add(Dense(128, kernel_initializer=glorot_normal(), bias_initializer=\
                    Constant(0.1), activation='relu'))
    model.add(Dense(6, kernel_initializer=glorot_normal(), bias_initializer=\
                    Constant(0.001)))

    opt = RMSprop()
    model.compile(loss='mse', optimizer=opt)
    print "\nCreated model\n"

tb_callback = TensorBoard(log_dir='/home/rharish/Programs/Python/RL'\
                          '/Tensorboard/' + str(time.time()))

pong_experience = deque()
pong_policy_epsilon = 0.6
pong_ep_num = 0

def saver():
    print "\n\nSaving model..."
    model.save('model_pong.h5')
    exp_file = open('model_pong.data', 'wb')
    pickle.dump((pong_policy_epsilon, pong_experience), exp_file)
    exp_file.close()
    copy('model_pong.h5', '../')
    copy('model_pong.data', '../')
    print "Model saved\n"

def pong_learn(num_episodes=20000, exp_size=200000, discount_factor=0.99,
               policy_epsilon=0.6, epsilon_decay_factor=0.9,
               epsilon_decay_episodes=100, min_epsilon=0.001, 
               save_model_every=1800, reset_fixed_every=5, exp_sample_size=32,
               save_video_every=50, tensorboard=True, render=False,
               verbose=True, video_record=True):

    env = gym.make('Pong-v4')
    if video_record:
        env = gym.wrappers.Monitor(env, 'Videos', resume=True, video_callable=
                                lambda count: count % save_video_every == 0)

    experience = deque(maxlen=exp_size)
    if 'model_pong.data' in os.listdir('.'):
        print "\nLoading model experience..."
        exp_file = open('model_pong.data')
        policy_epsilon, exp = pickle.load(exp_file)
        exp_file.close()
        experience.extend(exp)
        print "Model experience loaded\nExperience size = %d\n" % len(
              experience)
    global pong_experience
    pong_experience = experience
    global pong_policy_epsilon
    pong_policy_epsilon = policy_epsilon

    fixed_model = Sequential.from_config(model.get_config())
    now = time.time()

    global pong_ep_num
    for ep_num in range(pong_ep_num, num_episodes):
        pong_ep_num = ep_num
        state = deque(maxlen=4)
        state.append(imresize(cvtColor(env.reset(), COLOR_RGB2GRAY), (84, 84)).\
                     astype(np.uint8))
        done = False
        if ep_num % reset_fixed_every == 0:
            fixed_model.set_weights(model.get_weights())

        if (ep_num + 1) % epsilon_decay_episodes == 0 and policy_epsilon >\
        min_epsilon:
            policy_epsilon *= epsilon_decay_factor

        epoch = 0
        for step_num in itertools.count():
            if step_num < 3:
                action = env.action_space.sample()
            else:
                prediction = model.predict(np.array([state]), batch_size=1)
                policy = [(policy_epsilon / 6) for i in range(6)]
                policy[np.argmax(prediction[0])] += 1 - policy_epsilon
                action = np.random.choice(range(6), p=policy)

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
                exp_sample = random.sample(experience, exp_sample_size)

                states = []
                to_predict = []
                for exp_state, exp_action, exp_reward, exp_state_new in\
                exp_sample: 
                    states.append(exp_state)
                    exp_state = exp_state[1:]
                    exp_state_new_full = np.append(exp_state, [exp_state_new],
                                                   axis=0)
                    td_target = exp_reward + discount_factor *\
                                np.amax(fixed_model.predict(np.array(
                                [exp_state_new_full]), batch_size=1)[0])
                    to_predict.append(np.identity(6)[exp_action] * td_target)
                states = np.array(states)
                to_predict = np.array(to_predict)

                if tensorboard:
                    if verbose:
                        model.fit(states, to_predict, epochs=(epoch + 1),
                                  batch_size=exp_sample_size, callbacks=[
                                  tb_callback], initial_epoch=epoch)
                    else:
                        model.fit(states, to_predict, epochs=(epoch + 1),
                                  batch_size=exp_sample_size, callbacks=[
                                  tb_callback], initial_epoch=epoch,
                                  verbose=0)
                else:
                    if verbose:
                        model.fit(states, to_predict, epochs=(epoch + 1),
                                  batch_size=exp_sample_size,
                                  initial_epoch=epoch)
                    else:
                        model.fit(states, to_predict, epochs=(epoch + 1),
                                  batch_size=exp_sample_size, verbose=0,
                                  initial_epoch=epoch)
                epoch += 1

            if render:
                env.render()

            if not verbose:
                sys.stdout.write("\r%d steps taken\r" % (step_num + 1))
                sys.stdout.flush()

            if done:
                break
        
        if time.time() - now > save_model_every:
            saver()
            now = time.time()

        sys.stdout.write("\n%d episode(s) done out of %d\n" % (ep_num + 1,
                                                               num_episodes))
        sys.stdout.flush()

    saver()

while True:
    render = False
    try:
        pong_learn(exp_sample_size=16, num_episodes=10000,
                   save_video_every=100, tensorboard=False, verbose=False,
                   video_record=True, render=render)
        break
    except KeyboardInterrupt:
        try:
            response = raw_input('Toggle render? (y/N): ')
            if response.lower in ('y', 'yes', 'yup', 'ya', 'ja'):
                render = not render
                continue
        except KeyboardInterrupt:
            saver()
            break
    except:
        saver()
        print 'Time of error: ' + str(datetime.fromtimestamp(
                                      time.time()).strftime('%H%M on %a, %d '\
                                      '%b \'%y')) + '\n'
        raise

