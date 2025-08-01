from meta_policy_search.utils import logger
from meta_policy_search.meta_algos.base import MAMLAlgo
from meta_policy_search.optimizers.maml_first_order_optimizer import MAMLFirstOrderOptimizer

import tensorflow as tf
from collections import OrderedDict

tf.compat.v1.disable_eager_execution()# Api for using tf1 at tf2

class VPGMAML(MAMLAlgo):
    """
    Algorithm for PPO MAML

    Args:
        policy (Policy): policy object
        name (str): tf variable scope
        learning_rate (float): learning rate for the meta-objective
        exploration (bool): use exploration / pre-update sampling term / E-MAML term
        inner_type (str): inner optimization objective - either log_likelihood or likelihood_ratio
        inner_lr (float) : gradient step size used for inner step
        meta_batch_size (int): number of meta-learning tasks
        num_inner_grad_steps (int) : number of gradient updates taken per maml iteration
        trainable_inner_step_size (boolean): whether make the inner step size a trainable variable
    """
    def __init__(
            self,
            *args,
            name="vpg_maml",
            inner_type='likelihood_ratio',
            **kwargs
            ):
        super(VPGMAML, self).__init__(*args, **kwargs)
        assert inner_type in ["log_likelihood", "likelihood_ratio"]


        self.inner_type = inner_type
        self._optimization_keys = ['observations', 'actions', 'advantages', 'agent_infos']
        self.name = name

        self.build_graph()

    def _adapt_objective_sym(self, action_sym, adv_sym, dist_info_old_sym, dist_info_new_sym):
        if self.inner_type == 'likelihood_ratio':
            with tf.compat.v1.variable_scope("likelihood_ratio"):
                likelihood_ratio_adapt = self.policy.distribution.likelihood_ratio_sym(action_sym,
                                                                                       dist_info_old_sym, dist_info_new_sym)
            with tf.compat.v1.variable_scope("surrogate_loss"):
                surr_obj_adapt = -tf.reduce_mean(likelihood_ratio_adapt * adv_sym)

        elif self.inner_type == 'log_likelihood':
            with tf.compat.v1.variable_scope("log_likelihood"):
                log_likelihood_adapt = self.policy.distribution.log_likelihood_sym(action_sym, dist_info_new_sym)
            with tf.compat.v1.variable_scope("surrogate_loss"):
                surr_obj_adapt = -tf.reduce_mean(log_likelihood_adapt * adv_sym)

        else:
            raise NotImplementedError

        return surr_obj_adapt

    def build_graph(self):
        """
        Creates the computation graph
        """

        """ Create Variables """
        with tf.compat.v1.variable_scope(self.name):
            self.step_sizes = self._create_step_size_vars()

            """ --- Build inner update graph for adapting the policy and sampling trajectories --- """
            # this graph is only used for adapting the policy and not computing the meta-updates
            self.adapted_policies_params, self.adapt_input_ph_dict = self._build_inner_adaption()

            """ ----- Build graph for the meta-update ----- """
            self.meta_op_phs_dict = OrderedDict()
            obs_phs, action_phs, adv_phs, dist_info_old_phs, all_phs_dict = self._make_input_placeholders('step0')
            self.meta_op_phs_dict.update(all_phs_dict)

            distribution_info_vars, current_policy_params = [], []
            all_surr_objs = []

        for i in range(self.meta_batch_size):
            dist_info_sym = self.policy.distribution_info_sym(obs_phs[i], params=None)
            distribution_info_vars.append(dist_info_sym)  # step 0
            current_policy_params.append(self.policy.policy_params) # set to real policy_params (tf.Variable)


        with tf.compat.v1.variable_scope(self.name):
            """ Inner updates"""
            for step_id in range(1, self.num_inner_grad_steps+1):
                surr_objs, adapted_policy_params = [], []

                # inner adaptation step for each task
                for i in range(self.meta_batch_size):
                    surr_loss = self._adapt_objective_sym(action_phs[i], adv_phs[i], dist_info_old_phs[i], distribution_info_vars[i])

                    adapted_params_var = self._adapt_sym(surr_loss, current_policy_params[i])

                    adapted_policy_params.append(adapted_params_var)
                    surr_objs.append(surr_loss)

                all_surr_objs.append(surr_objs)
                # Create new placeholders for the next step
                obs_phs, action_phs, adv_phs, dist_info_old_phs, all_phs_dict = self._make_input_placeholders('step%i' % step_id)
                self.meta_op_phs_dict.update(all_phs_dict)

                # dist_info_vars_for_next_step
                distribution_info_vars = [self.policy.distribution_info_sym(obs_phs[i], params=adapted_policy_params[i])
                                          for i in range(self.meta_batch_size)]
                current_policy_params = adapted_policy_params


