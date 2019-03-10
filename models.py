
import tensorflow as tf

class adict(dict):

    def __init__(self, **items):
        super(adict, self).__init__(**items)
    def __getattr__(self, attr):
        return self[attr]
    def __setattr__(self, attr, val):
        self[attr] = val


class LocalNoiseRBM:
    """ RBM which allows for independent, spatially uniform noise processes."""

    def __init__(self, num_visible, num_hidden,
                                    kernel_initializer=tf.initializers.glorot_normal,
                                    bias_initializer=tf.initializers.zeros,
                                        alpha=2):
        """num_visible = number of visible units
            num_hidden = number of hidden units

            """
        self.num_visible = num_visible
        self.num_hidden = num_hidden

        self.dtype=tf.float32
        self.visible_bias = tf.get_variable("visible_bias",
                                            shape=(self.num_visible, 1),
                                            dtype=self.dtype,
                                            initializer=bias_initializer)
        self.hidden_bias = tf.get_variable("hidden_bias",
                                        shape=(self.num_hidden, 1),
                                        dtype=self.dtype,
                                        initializer=bias_initializer)
        self.weights = tf.get_variable("weights",
                                shape= (self.num_visible, self.num_hidden),
                                dtype=self.dtype,
                                initializer=kernel_initializer)

        self.noise_kernel =  tf.get_variable("noise_kernel",

                                    dtype=self.dtype,
                                    initializer=2 * alpha * tf.ones(())
                                    )


        self.noise_bias =   tf.get_variable("noise_bias",

                                            dtype=self.dtype,
                                            initializer= - alpha * tf.ones(())
                                            )



        self.variables = dict(visible_bias=self.visible_bias,
                            hidden_bias = self.hidden_bias,
                            weights = self.weights,
                            noise_kernel = self.noise_kernel,
                            noise_bias = self.noise_bias)

    def compute_noisy_probs(self, visible):
        """ Compute the probability of excitation in the noisy register,
        for a given visible setting """
        return tf.sigmoid(self.noise_kernel * visible + self.noise_bias)

    def compute_visible_probs(self, hidden, noise_condition=None):
        """ Given a hidden tensor, return the excitation probabilities for the
        visible layer.
            noise_condition: if not None, (N, num_visible tensor) of noisy
            register values.
            """
        # standard RBM formula based for the energy, given hiddens
        energy = self.visible_bias + tf.matmul(self.weights, hidden)
        if noise_condition is not None:
            # if we're conditioning on the noisy state, energy needs to be modified.
            energy += (tf.math.softplus(self.noise_bias)
                        - tf.math.softplus(self.noise_bias + self.noise_kernel)
                        + self.noise_kernel * noise_condition)
        return tf.sigmoid(energy)


    def compute_hidden_probs(self, visible, weights, hidden_bias):
        """ Given tensor of visible activations and couplings, return activation
        probabilities for hidden layer.
        weights: (N, num_visible, num_hidden)
        hidden_bias: (N, num_hidden,1)"""
        return tf.sigmoid( hidden_bias + tf.matmul(tf.transpose(weights, perm=[0, 2,1]), visible))

    def build_gibbs_chain(self, visible_init,
                             k,
                             noise_condition=False):
        """ Build a chain for k steps of Gibbs sampling.
            visible_init = (N, num_visible) tensor of initial visible states
            k = int, number of Gibbs sampling steps.
            noise_condition: if true, treat the visible seeds as noisy data,
            update the visible state during gibbs sampling accordingly

            returns: v, ph, ph0
             v, (N, num_visible,1) tensor of visible states.
             ph: (N, num_hidden,1) tensor of hidden probs conditioned on v
             ph0: (N, num_hidden,1) tensor of hidden probs conditioned on visible_init.
            """

        weights = self.weights
        hidden_bias = self.hidden_bias
        visible_bias = self.visible_bias

        noise_setting = visible_init if noise_condition else None
        if k <0:
            raise ValueError("k must be nonnegative int")
        with tf.name_scope("gibbs_sampling_%d"%k):
            v=visible_init
            ph0 = self.compute_hidden_probs(v, weights, hidden_bias)
            ph=ph0
            for step in range(k):
                with tf.name_scope("step_%d"%step):
                    h = tfp.distributions.Bernoulli(probs=ph,dtype=self.dtype).sample()
                    pv = self.compute_visible_probs(h, weights, visible_bias,
                                                    noise_setting=noise_setting)
                    v = tfp.distributions.Bernoulli(probs=pv,dtype=self.dtype).sample()
                    ph = self.compute_hidden_probs(v, weights, hidden_bias)
            return v, ph, ph0

    def free_energy_gradients(self, v, ph):
        """ Compute gradients of the free energy with respect to internal variables.
            v = visible state, shape (N, num_visible, 1)
            ph = excitation probabilities for hidden states, given v
                        shape (N, num_hidden, 1)
                                """
        grads = adict()
        grads.visible_bias = -v
        grads.hidden_bias = -ph
        grads.weights = -v * tf.transpose(ph, perm=(0,2,1))
        return grads

    def log_conditional_prob_gradients(self, visible, noisy):
        """ Compute gradients of the log conditional probs of noisy state values,
        given visibles """
        #prob of noisy activation, given hiddens
        pnoisy = self.compute_noisy_probs(visible)
        grads = adict()
        grads.noise_bias = noisy - pnoisy
        grads.noise_kernel = visible * (noisy-pnoisy)
        return grads

    def log_probability_gradients(self, v_data, ph_data,
                                    v_self, ph_self,
                                    noisy_state):
        """ Compute cost function gradients with respect to internal parameters.
        v_data, ph_data: visible state and corresponding hidden excitation probability
        computed from data.
            (note: if noise_conditioning, these are the outputs of the data-driven
            Gibbs chain; in other words, samples from the visible layer distribution,
            given the data))
        v_self, ph_self: same, but sampled from model distribution.
        noisy_state: the state of the noisy layer (
                v_data will have been obtained from this state by Gibbs sampling)

        all tensors should agree in first dimension.
        returns: adict of gradient values."""

        pos_grads = self.free_energy_gradients(v_data, ph_data)
        neg_grads = self.free_energy_gradients(v_self, ph_self)
        conditional_grads = self.log_conditional_prob_gradients(v_data,
                                                            noisy_data)
        grads = adict()
        for key in pos_grads.keys():
            grads[key] = pos_grads[key] - neg_grads[key]
        for key in conditional_grads.keys():
            grads[key] = conditional_grads[key]
        return grads

    def build_sampler(self, visible_feed, cond_feed, k
                            ):
        """ Return a tensor which produces samples from the visible layer
        using k steps of Gibbs sampling.

        Returns: v:
             visible outputs after k steps of sampling
            """

        #matmul will expect rank-3 tensors (including batch size)
        if len(visible_feed.shape)!=3:
            visible_feed = tf.expand_dims(visible_feed, 2)

        # obtain visible states by Gibbs sampling
        v_data = visible_feed
        v_self, ph_self, ph_data = self.build_gibbs_chain(v_data, k,
                                                        noise_condition=False)
        return v_self

    def estimate_logprob_grads(self, visible_feed, k,
                                    noise_condition=True):
        """ Build a graph for estimating gradients of RBM parameters
            using k steps of Gibbs sampling
            visible_feed: (N, num_visible, 1) tensor of visible samples from data
            cond_feed: (N, ?) tensor of conditioning variables.
            k: int, number of Gibbs sampling steps.
            noise_condition: whether or not to treat visible samples as noisy
            returns: adict of gradients of the log-probability of the data,
            averaged over samples."""

        if len(visible_feed.shape)!=3:
            visible_feed = tf.expand_dims(visible_feed, 2)

        #if conditioning on noise layer, need to run sampling to infer visible
        # states
        if noise_condition:
            v_data, ph_data, __ = self.build_gibbs_chain(visible_feed, k,
                                                noise_condition=True)
            v_self, ph_self, __ = self.build_gibbs_chain(visible_feed, k,
                                                noise_condition=False)
        ### otherwise, visible states are driven directly from data
        else:
            v_data = visible_feed
            v_self, ph_self, ph_data = self.build_gibbs_chain(visible_feed, k,
                                                        noise_condition=False)
        noisy_state = visible_feed

        #compute the internal gradients
        grads = self.log_probability_gradients(v_data, ph_data,
                                                v_self, ph_self, noisy_state)
        # average over examples in batch
        for key in grads.keys():
            grads[key] = tf.reduce_mean(grads[key], axis=0)
        return grads

    def nll_gradients(self, visible_feed, k, noise_condition=True):
        """ Returns list of (variable, gradient) pairs, where
            variables are the trainable params of the rbm, and
            gradients are the gradients of the negative-log-likelihood cost
            function with respect to the paired variables.
            """
        logprob_grads = self.estimate_logprob_grads(visible_feed, k,
                                            noise_condition=noise_condition)
        gradlist = []
        for varname in self.variables.keys():
            gradlist += (self.variables[varname], -logprob_grads[varname])
        return gradlist
