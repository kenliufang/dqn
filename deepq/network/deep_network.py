import tensorflow as tf
import tensorflow.contrib.layers as layers

import deepq.common.replay_buffer as rb
import deepq.common.linear_scheduler as ls

import tempfile

import numpy as np
import os

class DeepQNet(object):
    def __init__(self, env):
        self.env = env


    def _make_dqn(self, convs, hiddens, dueling, inpt, num_actions, scope, reuse=False):
        # print(inpt)
        with tf.variable_scope(scope, reuse=reuse):
            out = inpt
            with tf.variable_scope("convnet"):
                for num_outputs, kernel_size, stride in convs:
                    out = layers.convolution2d(out,
                                               num_outputs=num_outputs,
                                               kernel_size=kernel_size,
                                               stride=stride,
                                               activation_fn=tf.nn.relu)
                    # print(out)

            out = layers.flatten(out)
            # print(out)
            with tf.variable_scope("action_value"):
                action_out = out
                for hidden in hiddens:
                    action_out = layers.fully_connected(action_out, num_outputs=hidden, activation_fn=tf.nn.relu)
                print(out)
                action_scores = layers.fully_connected(action_out, num_outputs=num_actions, activation_fn=None)

            # print(action_scores)
            if dueling:
                with tf.variable_scope("state_value"):
                    state_out = out
                    for hidden in hiddens:
                        state_out = layers.fully_connected(state_out, num_outputs=hidden, activation_fn=tf.nn.relu)
                    state_score = layers.fully_connected(state_out, num_outputs=1, activation_fn=None)
                action_scores_mean = tf.reduce_mean(action_scores, 1)
                action_scores_centered = action_scores - tf.expand_dims(action_scores_mean, 1)
                return state_score + action_scores_centered
            else:
                return action_scores

    def make_dqn(self, inpt, num_actions, scope, reuse=False):
        return self._make_dqn(convs=[(32, 8, 4), (64, 4, 2), (64, 3, 1)],
                                  hiddens = [256],
                                  dueling=False,
                                  inpt=inpt,
                                  num_actions=num_actions,
                                  scope=scope,
                                  reuse=reuse)


    def build_act(self, scope="deepq", reuse=None):
        """Creates the act function:

        Parameters
        ----------
        make_obs_ph: str -> tf.placeholder or TfInput
            a function that take a name and creates a placeholder of input with that name
        q_func: (tf.Variable, int, str, bool) -> tf.Variable
            the model that takes the following inputs:
                observation_in: object
                    the output of observation placeholder
                num_actions: int
                    number of actions
                scope: str
                reuse: bool
                    should be passed to outer variable scope
            and returns a tensor of shape (batch_size, num_actions) with values of every action.
        num_actions: int
            number of actions.
        scope: str or VariableScope
            optional scope for variable_scope.
        reuse: bool or None
            whether or not the variables should be reused. To be able to reuse the scope must be given.

        Returns
        -------
        act: (tf.Variable, bool, float) -> tf.Variable
            function to select and action given observation.
    `       See the top of the file for details.
        """
        with tf.variable_scope(scope, reuse=reuse):
            observations_ph =  tf.placeholder(tf.float32, [None] + list(self.env.observation_space.shape))
            stochastic_ph = tf.placeholder(tf.bool, (), name="stochastic")
            update_eps_ph = tf.placeholder(tf.float32, (), name="update_eps")

            eps = tf.get_variable("eps", (), initializer=tf.constant_initializer(0))

            q_values = self.make_dqn(observations_ph, self.env.action_space.n, scope="q_func")
            deterministic_actions = tf.argmax(q_values, axis=1)

            batch_size = tf.shape(observations_ph)[0]
            random_actions = tf.random_uniform(tf.stack([batch_size]), minval=0, maxval=self.env.action_space.n, dtype=tf.int64)
            chose_random = tf.random_uniform(tf.stack([batch_size]), minval=0, maxval=1, dtype=tf.float32) < eps
            stochastic_actions = tf.where(chose_random, random_actions, deterministic_actions)

            output_actions = tf.cond(stochastic_ph, lambda: stochastic_actions, lambda: deterministic_actions)
            # print("output_actions{}".format(output_actions))
            update_eps_expr = eps.assign(tf.cond(update_eps_ph >= 0, lambda: update_eps_ph, lambda: eps))

            # act = U.function(inputs=[observations_ph, stochastic_ph, update_eps_ph],
            #                  outputs=output_actions,
            #                  givens={update_eps_ph: -1.0, stochastic_ph: True},
            #                  updates=[update_eps_expr])
            #feed_dict = {}
            return lambda ob, update_eps :\
                tf.get_default_session().run([output_actions, update_eps_expr],
                                             feed_dict={observations_ph: ob, update_eps_ph: update_eps, stochastic_ph: True})

    def get_scoped_vars(self, scope_name):
        full_scope_name = tf.get_variable_scope().name + "/" + scope_name
        return tf.get_collection(tf.GraphKeys.GLOBAL_VARIABLES, scope=full_scope_name)

    def huber_loss(self, x, delta=1.0):
        """Reference: https://en.wikipedia.org/wiki/Huber_loss"""
        return tf.where(
            tf.abs(x) < delta,
            tf.square(x) * 0.5,
            delta * (tf.abs(x) - 0.5 * delta)
        )

    def build_train(self, optimizer, grad_norm_clipping=None, gamma=1.0, double_q=True,
                scope="deepq", reuse=None):
        num_actions = self.env.action_space.n
        act_f = self.build_act(scope=scope, reuse=reuse)
        with tf.variable_scope(scope, reuse=reuse):
            # set up placeholders
            obs_shape = [None] + list(self.env.observation_space.shape)
            obs_t_input = tf.placeholder(tf.float32, obs_shape, "obs_t")
            # print("obs_t_input{}".format(obs_t_input))
            act_t_ph = tf.placeholder(tf.int32, [None], name="action")
            # print("act_t_ph{}".format(act_t_ph))
            rew_t_ph = tf.placeholder(tf.float32, [None], name="reward")
            obs_tp1_input = tf.placeholder(tf.float32, obs_shape, "obs_tp1")
            done_mask_ph = tf.placeholder(tf.float32, [None], name="done")
            importance_weights_ph = tf.placeholder(tf.float32, [None], name="weight")

            # q network evaluation
            q_t = self.make_dqn(obs_t_input, num_actions, scope="q_func", reuse=True)  # reuse parameters from act
            q_func_vars = self.get_scoped_vars("q_func")

            # target q network evalution
            q_tp1 = self.make_dqn(obs_tp1_input, num_actions, scope="target_q_func")
            target_q_func_vars = self.get_scoped_vars("target_q_func")

            # q scores for actions which we know were selected in the given state.
            q_t_selected = tf.reduce_sum(q_t * tf.one_hot(act_t_ph, num_actions), 1)

            # compute estimate of best possible value starting from state at t + 1


            q_tp1_best = tf.reduce_max(q_tp1, 1)
            q_tp1_best_masked = (1.0 - done_mask_ph) * q_tp1_best

            # compute RHS of bellman equation
            q_t_selected_target = rew_t_ph + gamma * q_tp1_best_masked

            # compute the error (potentially clipped)
            td_error = q_t_selected - tf.stop_gradient(q_t_selected_target)
            errors = self.huber_loss(td_error)
            weighted_error = tf.reduce_mean(importance_weights_ph * errors)
            # compute optimization op (potentially with gradient clipping)

            optimize_expr = optimizer.minimize(weighted_error, var_list=q_func_vars)

            # update_target_fn will be called periodically to copy Q network to target Q network
            update_target_expr = []
            for var, var_target in zip(sorted(q_func_vars, key=lambda v: v.name),
                                       sorted(target_q_func_vars, key=lambda v: v.name)):
                update_target_expr.append(var_target.assign(var))
            update_target_expr = tf.group(*update_target_expr)


            # Create callable functions
            train = lambda obs_t, act_t, rew_t, obs_tp1, done_mask, importance_weights :\
                tf.get_default_session().run([td_error, optimize_expr],
                                             feed_dict={obs_t_input:obs_t, act_t_ph:act_t,
                                                        rew_t_ph:rew_t, obs_tp1_input:obs_tp1,
                                                        done_mask_ph:done_mask,
                                                        importance_weights_ph:importance_weights})
            #         importance_weights_ph U.function(
            #     inputs=[
            #         obs_t_input,
            #         act_t_ph,
            #         rew_t_ph,
            #         obs_tp1_input,
            #         done_mask_ph,
            #         importance_weights_ph
            #     ],
            #     outputs=td_error,
            #     updates=[optimize_expr]
            # )
            update_target = lambda : tf.get_default_session().run([update_target_expr])


            q_values = lambda obs : tf.get_default_session().run([q_t], feed_dict={obs_t_input: obs})

            return act_f, train, update_target, {'q_values': q_values}


    def train(self, lr=5e-4,
          max_timesteps=100000,
          buffer_size=50000,
          exploration_fraction=0.1,
          exploration_final_eps=0.02,
          train_freq=1,
          batch_size=32,
          print_freq=1,
          checkpoint_freq=10000,
          learning_starts=1000,
          gamma=1.0,
          target_network_update_freq=500,
          final_file = None):
        session = tf.Session()
        replay_buffer = rb.ReplayBuffer(buffer_size)
        exploration = ls.LinearScheduler(schedule_timesteps=int(exploration_fraction * max_timesteps),
                                     initial_p=1.0,
                                     final_p=exploration_final_eps)
        with session.as_default() as sess:
            # tf.initialize_all_variables().run()
            act, train, update_target, debug = self.build_train(
                optimizer=tf.train.AdamOptimizer(learning_rate=lr),
                gamma=gamma
            )
            saver = tf.train.Saver()
            sess.run(tf.global_variables_initializer())
            episode_rewards = [0.0]
            saved_mean_reward = None
            obs = self.env.reset()
            td = tempfile.gettempdir()
            # with tempfile.gettempdir() as td:
            # print(td)
            model_saved = False
            model_file = os.path.join(td, "model")
            for t in range(max_timesteps):
                # if callback is not None:
                #     if callback(locals(), globals()):
                #         break
                # Take action and update exploration to the newest value
                action, _ = act(np.array(obs)[None], update_eps=exploration.value(t))
                #[0]
                # print("action{}".format(lenaction))
                action = action[0]
                # print("action{}".format(action.shape))
                new_obs, rew, done, _ = self.env.step(action)
                # Store transition in the replay buffer.
                replay_buffer.add(obs, action, rew, new_obs, float(done))
                obs = new_obs

                episode_rewards[-1] += rew
                if done:
                    obs = self.env.reset()
                    episode_rewards.append(0.0)

                if t > learning_starts and t % train_freq == 0:
                    obses_t, actions, rewards, obses_tp1, dones = replay_buffer.sample(batch_size)

                    weights, batch_idxes = np.ones_like(rewards), None
                    # print("ABCDED{}".format(actions.shape))
                    td_errors = train(obses_t, actions, rewards, obses_tp1, dones, weights)


                if t > learning_starts and t % target_network_update_freq == 0:
                    # Update target network periodically.
                    update_target()

                mean_100ep_reward = round(np.mean(episode_rewards[-101:-1]), 1)
                num_episodes = len(episode_rewards)
                if done and print_freq is not None and len(episode_rewards) % print_freq == 0:
                    print("steps\t{}".format(t))
                    print("episodes\t{}".format(num_episodes))
                    print("mean 100 episode reward\t{}".format(mean_100ep_reward))
                    print("{} time spent exploring".format(int(100 * exploration.value(t))))
                    print("")


                if (checkpoint_freq is not None and t > learning_starts and
                            num_episodes > 100 and t % checkpoint_freq == 0):
                    if saved_mean_reward is None or mean_100ep_reward > saved_mean_reward:
                        if print_freq is not None:
                            print("Saving model due to mean reward increase: {} -> {}".format(
                                saved_mean_reward, mean_100ep_reward))
                        saver.save(sess=sess, save_path=model_file)
                        model_saved = True
                        saved_mean_reward = mean_100ep_reward
            if model_saved:
                if print_freq is not None:
                    print("Restored model with mean reward: {}".format(saved_mean_reward))
                saver.restore(sess=sess, save_path=model_file)
            if final_file:
                saver.save(sess=sess, save_path=final_file)

    def play(self, model_prefix):
        session = tf.Session()
        with session.as_default() as sess:
            act = self.build_act()
            saver = tf.train.Saver()
            saver.restore(sess, save_path=model_prefix)

            obs, done = self.env.reset(), False
            episode_rew = 0
            while not done:
                self.env.render()
                action, _ = act(obs.__array__()[None], 0)
                obs, rew, done, _ = self.env.step(action[0])
                episode_rew += rew
            return episode_rew