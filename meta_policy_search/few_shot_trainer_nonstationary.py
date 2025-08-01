import tensorflow as tf
import numpy as np
import time
from meta_policy_search.utils import logger
from tensorboardX import SummaryWriter
from tensorflow.python.client import device_lib
import pandas as pd
import os
tf.config.set_visible_devices([], "GPU")
tf.compat.v1.disable_eager_execution()# Api for using tf1 at tf2
class Trainer(object):
    """
    Performs steps of meta-policy search.

     Pseudocode::

            for iter in n_iter:
                sample tasks
                for task in tasks:
                    for adapt_step in num_inner_grad_steps
                        sample trajectories with policy
                        perform update/adaptation step
                    sample trajectories with post-update policy
                perform meta-policy gradient step(s)

    Args:
        algo (Algo) :
        env (Env) :
        sampler (Sampler) :
        sample_processor (SampleProcessor) :
        baseline (Baseline) :
        policy (Policy) :
        n_itr (int) : Number of iterations to train for
        start_itr (int) : Number of iterations policy has already trained for, if reloading
        num_inner_grad_steps (int) : Number of inner steps per maml iteration
        sess (tf.Session) : current tf session (if we loaded policy, for example)
    """
    def __init__(
            self,
            algo,
            env,
            sampler_for_learning,
            sampler_for_test,
            sample_processor,
            policy,
            n_itr,
            num_inner_grad_steps,
            sess=None,
            ):
        self.algo = algo
        self.env = env
        self.sampler_for_learning = sampler_for_learning
        self.sampler_for_test = sampler_for_test
        self.sample_processor = sample_processor
        self.policy = policy
        self.n_itr = n_itr
        self.num_inner_grad_steps = num_inner_grad_steps
        if sess is None:
            sess = tf.compat.v1.Session()
        self.sess = sess
        self.reward = 0
        self.reward_by_shot = []
        self.data = None # for validation

    def train(self, model_path):

        with self.sess.as_default() as sess:
            saver = tf.compat.v1.train.Saver()
            uninit_vars = [var for var in tf.compat.v1.global_variables() if not sess.run(tf.compat.v1.is_variable_initialized(var))]
            sess.run(tf.compat.v1.variables_initializer(uninit_vars))
            # Model load
            if model_path != False:
                # 3. 파라미터 복원
                saver.restore(sess, model_path)  # 만들어진 graph에 parameter설정, (폴더명 입력)

            else:
                pass
            # self.sampler.update_tasks()
            self.policy.switch_to_pre_update()  # Switch to pre-update policy
            self.sampler_for_learning.vec_env.set_tasks([self.env.tasks[0]]) # task setting
            self.sampler_for_test.vec_env.set_tasks([self.env.tasks[0]]) # task setting
            # Few-Shot-learning start
            for step in range(0, self.n_itr):
                paths= self.sampler_for_learning.obtain_samples(log=False, log_prefix='Step_%d-' % step)
                samples_data = self.sample_processor.process_samples(paths, log='all', log_prefix='Step_%d-' % step)
                reward, _, self.data = self.env.log_diagnostics(sum(list(paths.values()), []), 0, self.n_itr)
                print("Reward:", reward)
                
                self.algo._adapt(samples_data)

                paths= self.sampler_for_test.obtain_samples(log=False, log_prefix='Step_%d-' % step)
                
                reward, _, self.data = self.env.log_diagnostics(sum(list(paths.values()), []), 0, self.n_itr)
                self.reward += reward
                # Append shot_result
                self.reward_by_shot.append(self.reward)
                # reset
                self.reward = 0

        self.sess.close()      
        return self.reward_by_shot  
