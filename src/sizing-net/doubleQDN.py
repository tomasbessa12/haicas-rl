#!/usr/bin/python3.6

import itertools
import numpy as np
import os
import random
import sys
import tensorflow as tf
import plotting
import matplotlib.pyplot as plt

if "../" not in sys.path:
  sys.path.append("../")

from collections import deque, namedtuple

import circuit.vcamp as vcamp

class Estimator():
    """Q-Value Estimator neural network.

    This network is used for both the Q-Network and the Target Network.
    """

    def __init__(self, state_size, action_size, scope="estimator", summaries_dir=None):
        self.scope = scope
        # Writes Tensorboard summaries to disk
        self.summary_writer = None
        with tf.compat.v1.variable_scope(scope):
            # Build the graph
            self._build_model(state_size, action_size)
            if summaries_dir:
                summary_dir = os.path.join(summaries_dir, "summaries_{}".format(scope))
                if not os.path.exists(summary_dir):
                    os.makedirs(summary_dir)
                self.summary_writer = tf.summary.FileWriter(summary_dir)

    def _build_model(self, state_size, action_size):
        """
        Builds the Tensorflow graph.
        """

        # Placeholders for our input
        # Our inputs are observations and actions Q(s,a) 
        self.state_pl = tf.compat.v1.placeholder(shape=[None, state_size], dtype=tf.float32, name="state")
        self.action_pl = tf.compat.v1.placeholder(shape=[None,action_size], dtype=tf.float32, name="action")
        # The TD target value
        self.y_pl = tf.compat.v1.placeholder(shape=[None], dtype=tf.float32, name="y")
        
        X = tf.concat([self.state_pl,self.action_pl], 1 , name="X")
        
        # Fully connected layer
        fc1 = tf.layers.fully_connected(X, 256, activation_fn=tf.nn.relu)
        fc2 = tf.layers.fully_connected(fc1, 128, activation_fn=tf.nn.relu)
        fc3 = tf.layers.fully_connected(fc2, 128)
        self.predictions = tf.layers.fully_connected(fc3, 1, activation_fn=None)
        
        # Calculate the loss
        self.losses = tf.squared_difference(self.y_pl, self.predictions)
        self.loss = tf.reduce_mean(self.losses)

        # Optimizer Parameters from original paper
        self.optimizer = tf.train.RMSPropOptimizer(0.00025, 0.99, 0.0, 1e-6)
        self.train_op = self.optimizer.minimize(self.loss, global_step=tf.train.get_global_step())
        
        # Summaries for Tensorboard
        self.summaries = tf.summary.merge([
            tf.summary.scalar("loss", self.loss),
            tf.summary.histogram("target_values", self.y_pl),
            tf.summary.histogram("loss_hist", self.losses),
            tf.summary.histogram("q_values", self.predictions)
        ])



    def predict(self, sess, s, a):
        """
        Predicts expected reward Q for state action pair.

        Args:
          sess: Tensorflow session
          s: State input of shape [batch_size, 49]
          a: Action input of shape [batch_size, 18]

        Returns:
          Tensor of shape [batch_size, 1] containing the estimated 
          expected reward values.
        """
        if(np.isnan(s).any()):
            raise Exception('State with nan. The value of s was: {}'.format(s))
        if(np.isnan(a).any()):
            raise Exception('Action with nan. The value of a was: {}'.format(a))
        
        
        
        return sess.run(self.predictions, { self.state_pl: s, self.action_pl : a })

    def update(self, sess, s, a, y):
        """
        Updates the estimator towards the given targets.

        Args:
          sess: Tensorflow session object
          s: State input of shape [batch_size, 49]
          a: Chosen actions of shape [batch_size, 18]
          y: Targets of shape [batch_size]

        Returns:
          The calculated loss on the batch.
        """
        if(np.isnan(s).any()):
            raise Exception('State with nan. The value of s was: {}'.format(s))
        if(np.isnan(a).any()):
            raise Exception('Action with nan. The value of a was: {}'.format(a))
        if(np.isnan(y).any()):
            raise Exception('Target with nan. The value of Y was: {}'.format(y))
        
            
        feed_dict = { self.state_pl: s, self.y_pl: y, self.action_pl: a }
        summaries, global_step, _, loss = sess.run(
            [self.summaries, tf.train.get_global_step(), self.train_op, self.loss],
            feed_dict)
        if self.summary_writer:
            self.summary_writer.add_summary(summaries, global_step)
        return loss

def copy_model_parameters(sess, estimator1, estimator2):
    """
    Copies the model parameters of one estimator to another.

    Args:
      sess: Tensorflow session instance
      estimator1: Estimator to copy the paramters from
      estimator2: Estimator to copy the parameters to
    """
    e1_params = [t for t in tf.trainable_variables() if t.name.startswith(estimator1.scope)]
    e1_params = sorted(e1_params, key=lambda v: v.name)
    e2_params = [t for t in tf.trainable_variables() if t.name.startswith(estimator2.scope)]
    e2_params = sorted(e2_params, key=lambda v: v.name)

    update_ops = []
    for e1_v, e2_v in zip(e1_params, e2_params):
        op = e2_v.assign(e1_v)
        update_ops.append(op)

    sess.run(update_ops)




def deep_q_learning(sess,
                    env,
                    q_estimator,
                    target_estimator,
                    num_episodes,
                    experiment_dir,
                    replay_memory_size=500000,
                    replay_memory_init_size=50000,
                    update_target_estimator_every=10000,
                    discount_factor=0.99,
                    epsilon_start=1.0,
                    epsilon_end=0.1,
                    epsilon_decay_steps=500000,
                    batch_size=32,
                    record_video_every=50):
    """
    Q-Learning algorithm for off-policy TD control using Function Approximation.
    Finds the optimal greedy policy while following an epsilon-greedy policy.

    Args:
        sess: Tensorflow Session object
        env: OpenAI ish environment
        q_estimator: Estimator object used for the q values
        target_estimator: Estimator object used for the targets
        num_episodes: Number of episodes to run for
        experiment_dir: Directory to save Tensorflow summaries in
        replay_memory_size: Size of the replay memory
        replay_memory_init_size: Number of random experiences to sampel when initializing 
          the reply memory.
        update_target_estimator_every: Copy parameters from the Q estimator to the 
          target estimator every N steps
        discount_factor: Gamma discount factor
        epsilon_start: Chance to sample a random action when taking an action.
          Epsilon is decayed over time and this is the start value
        epsilon_end: The final minimum value of epsilon after decaying is done
        epsilon_decay_steps: Number of steps to decay epsilon over
        batch_size: Size of batches to sample from the replay memory
        record_video_every: Record a video every N episodes

    Returns:
        An EpisodeStats object with two numpy arrays for episode_lengths and episode_rewards.
    """

    Transition = namedtuple("Transition", ["state", "action", "reward", "next_state", "done"])

    # The replay memory
    replay_memory = []

    # Keeps track of useful statistics
    stats = plotting.EpisodeStats(
        episode_lengths=np.zeros(num_episodes),
        episode_rewards=np.zeros(num_episodes))

    # Create directories for checkpoints and summaries
    checkpoint_dir = os.path.join(experiment_dir, "checkpoints")
    checkpoint_path = os.path.join(checkpoint_dir, "model")
    monitor_path = os.path.join(experiment_dir, "monitor")

    if not os.path.exists(checkpoint_dir):
        os.makedirs(checkpoint_dir)
    if not os.path.exists(monitor_path):
        os.makedirs(monitor_path)

    saver = tf.train.Saver()
    # Load a previous checkpoint if we find one
    latest_checkpoint = tf.train.latest_checkpoint(checkpoint_dir)
    if latest_checkpoint:
        print("Loading model checkpoint {}...\n".format(latest_checkpoint))
        saver.restore(sess, latest_checkpoint)
    
    # Get the current time step
    total_t = sess.run(tf.contrib.framework.get_global_step())

    # The epsilon decay schedule
    epsilons = np.linspace(epsilon_start, epsilon_end, epsilon_decay_steps)

    # Populate the replay memory with initial experience
    print("Populating replay memory using random off-policy sampling...")
    state = env.reset()
    state = np.nan_to_num(state)
    state = np.divide(state ,env.state_scale)
    for i in range(replay_memory_init_size):
        action = env.sample_action()
        next_state, reward, done, log = env.step(action)
        
        next_state = np.nan_to_num(next_state)
        next_state = np.divide(next_state ,env.state_scale)
        
        print(str(i) + ": " + str(log))
        replay_memory.append(Transition(state, action, reward, next_state, done))
        if done:
            state = env.reset()
            state = np.nan_to_num(state)
            state = np.divide(state ,env.state_scale)
        else:
            state = next_state
    
    print("Replay memory populated...")
    for i_episode in range(num_episodes):

        # Save the current checkpoint
        saver.save(tf.get_default_session(), checkpoint_path)

        # Reset the environment
        state = env.reset()
        state = np.nan_to_num(state)
        state = np.divide(state ,env.state_scale)
        loss = None

        # One step in the environment
        for t in itertools.count():

            # Epsilon for this time step
            epsilon = epsilons[min(total_t, epsilon_decay_steps-1)]

            # Add epsilon to Tensorboard
            episode_summary = tf.Summary()
            episode_summary.value.add(simple_value=epsilon, tag="epsilon")
            q_estimator.summary_writer.add_summary(episode_summary, total_t)

            # Maybe update the target estimator
            if total_t % update_target_estimator_every == 0:
                copy_model_parameters(sess, q_estimator, target_estimator)
                print("\nCopied model parameters to target network.")



            # Take a step
            #evaluate a set of random actions
            actions = env.sample_action(2048)
            q_values = q_estimator.predict(sess, np.tile(state, [actions.shape[0],1]), actions)
            best_action = np.argmax(q_values)
       
            next_state, reward, done, log = env.step(actions[best_action])
            
            next_state = np.nan_to_num(next_state)
         
            next_state = np.divide(next_state,env.state_scale)


            # Print out which step we're on, useful for debugging.
            print("\rStep {} ({}) @ Episode {}/{}, performance: {}, reward: {},q: {}, loss: {}                   ".format(
                    t, total_t, i_episode + 1, num_episodes,log['performance'], reward, np.max(q_values), loss), end="")
            
            sys.stdout.flush()

            # If our replay memory is full, pop the first element
            if len(replay_memory) == replay_memory_size:
                replay_memory.pop(0)

            # Save transition to replay memory
            replay_memory.append(Transition(state, actions[best_action], reward, next_state, done))   

            # Update statistics
            stats.episode_rewards[i_episode] += reward
            stats.episode_lengths[i_episode] = t

            # Sample a minibatch from the replay memory
            samples = random.sample(replay_memory, batch_size)
            states_batch, action_batch, reward_batch, next_states_batch, done_batch = map(np.array, zip(*samples))

            # Calculate q values and targets
            # This is where Double Q-Learning comes in!
            
            action_samples = 2048
            next_actions = env.sample_action(action_samples*batch_size)
            action_size = next_actions.shape[1]
        
            next_states_for_sample = np.repeat(next_states_batch,action_samples, axis=0)
            q_values_next = q_estimator.predict(sess, next_states_for_sample, next_actions)
            
            q_values_next = q_values_next.reshape((batch_size,action_samples,1))
            
            max_loc = np.argmax(q_values_next, axis=1)
            max_loc = max_loc.reshape(batch_size)
            
            for i in range(batch_size):
                max_loc[i] = max_loc[i] + i*action_samples
                
            best_actions = next_actions[max_loc,:]
        
            q_values_next_target = target_estimator.predict(sess, next_states_batch, best_actions)
            q_values_next_target = q_values_next_target.reshape(batch_size)
            targets_batch = reward_batch + np.invert(done_batch).astype(np.float32) * discount_factor * q_values_next_target
                
            # Perform gradient descent update
            states_batch = np.array(states_batch)
            loss = q_estimator.update(sess, states_batch, action_batch, targets_batch)

            if done:
                break

            state = next_state
            total_t += 1

        # Add summaries to tensorboard
        episode_summary = tf.Summary()
        episode_summary.value.add(simple_value=stats.episode_rewards[i_episode], node_name="episode_reward", tag="episode_reward")
        episode_summary.value.add(simple_value=stats.episode_lengths[i_episode], node_name="episode_length", tag="episode_length")
        q_estimator.summary_writer.add_summary(episode_summary, total_t)
        q_estimator.summary_writer.flush()
        print(env.values)
        print(env.measures)
        yield total_t, plotting.EpisodeStats(
            episode_lengths=stats.episode_lengths[:i_episode+1],
            episode_rewards=stats.episode_rewards[:i_episode+1])

    monitor.close()
    return stats


if __name__ == '__main__':
    env = vcamp.VcAmpRLEnv()
    seed = 17
    np.random.seed(seed)
    print(env.state_scale)

    
    # For Testing....
    tf.compat.v1.reset_default_graph()
    tf.compat.v1.disable_eager_execution()

    global_step = tf.Variable(0, name="global_step", trainable=False)

    e = Estimator(env.state_size, env.action_size, scope="test")

    with tf.Session() as sess:
        sess.run(tf.global_variables_initializer())
        
        # Example observation batch
        
        observation = env.reset()
        print(observation)
        observation = np.nan_to_num(observation)
        observation = np.divide(observation ,env.state_scale)
        observation = np.array([observation.transpose()])
        
        print(observation)
        action = np.expand_dims(env.sample_action(), axis=0)
        y = np.array([10.0])
        
        
        print(np.max(observation))
        # Test Prediction
        print(e.predict(sess, observation, action))

        # Test training step
        print(e.update(sess, observation, action, y))
        
    # runnit:
    tf.reset_default_graph()

    # Where we save our checkpoints and graphs
    experiment_dir = os.path.abspath("./experiments/{}".format("vcamp"))

    # Create a glboal step variable
    global_step = tf.Variable(0, name='global_step', trainable=False)
    # Create estimators
    q_estimator = Estimator(env.state_size, env.action_size, scope="q", summaries_dir=experiment_dir)
    target_estimator = Estimator(env.state_size, env.action_size, scope="target_q")

    # Run it!
    with tf.Session() as sess:
        sess.run(tf.initialize_all_variables())
        for t, stats in deep_q_learning(sess,
                                        env,
                                        q_estimator=q_estimator,
                                        target_estimator=target_estimator,
                                        experiment_dir=experiment_dir,
                                        num_episodes=1000,
                                        replay_memory_size=5000,
                                        replay_memory_init_size=128,
                                        update_target_estimator_every=1000,
                                        epsilon_start=1.0,
                                        epsilon_end=0.1,
                                        epsilon_decay_steps=500,
                                        discount_factor=0.98,
                                        batch_size=32):

            print("\nEpisode Reward: {} test".format(stats.episode_rewards[-1]))

