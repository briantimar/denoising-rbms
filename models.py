
class LocalNoiseRBM:
    """ RBM which allows for independent, spatially uniform noise processes."""

    def __init__(self, num_visible, num_hidden,
                                    kernel_initializer='glorot_normal',
                                    bias_initializer='zeroes'
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

        self.noise_kernel = tf.tile( tf.get_variable("noise_kernel",
                                    shape = (1,),
                                    dtype=self.dtype,
                                    initializer=2 * alpha * tf.ones((1,))
                                    ),
                                    [1, self.num_visible])

        self.noise_bias =   tf.tile(   tf.get_variable("noise_bias",
                                            shape = (1,1),
                                            dtype=self.dtype,
                                            initializer= - alpha * tf.ones((1,1))
                                            ),
                                            [1, self.num_visible])


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
        
        return tf.sigmoid( self.visible_bias + tf.matmul(self.weights, hidden))

    def compute_hidden_probs(self, visible, weights, hidden_bias):
        """ Given tensor of visible activations and couplings, return activation
        probabilities for hidden layer.
        weights: (N, num_visible, num_hidden)
        hidden_bias: (N, num_hidden,1)"""
        return tf.sigmoid( hidden_bias + tf.matmul(tf.transpose(weights, perm=[0, 2,1]), visible))

    def build_gibbs_chain(self, visible_init,
                             k,
                             noise_condition=True):
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

    def internal_gradients(self, v_data, ph_data,
                                v_self, ph_self):
        """ Compute cost function gradients with respect to internal parameters.
        v_data, ph_data: visible state and corresponding hidden excitation probability
        computed from data.
        v_self, ph_self: same, but sampled from model distribution.
        all tensors should agree in first dimension.

        returns: adict of gradient values."""

        pos_grads = self.free_energy_gradients(v_data, ph_data)
        neg_grads = self.free_energy_gradients(v_self, ph_self)
        grads = adict()
        for key in pos_grads.keys():
            grads[key] = pos_grads[key] - neg_grads[key]
        return grads

    def get_external_variables(internal_varname):
        """ Return all external trainable variables associated with a
        given internal variable name"""
        return self.internal_variable_models[internal_varname].variables

    def effective_cost_function(self, internal_grads, internal_vars):
        """ Given internal gradients estimated from Gibbs sampling, return
        an effective cost function whose gradients can be used to optimize
        the external variables.
            internal_grads: adict holding estimates of the internal gradients
            internal_vars: adict holding corresponding values of the internal vars"""
        return sum( [ tf.reduce_sum(tf.stop_gradient(internal_grads[k]) * internal_vars[k])
                        for k in internal_grads.keys() ])

    def build_sampler(self, visible_feed, cond_feed, k):
        """ Return a tensor which produces samples from the visible layer
        using k steps of Gibbs sampling.

        Returns: v:
             visible outputs after k steps of sampling
            """

        #matmul will expect rank-3 tensors (including batch size)
        if len(visible_feed.shape)!=3:
            visible_feed = tf.expand_dims(visible_feed, 2)

        internal_vars = adict()
        # compute the values of the internal variables.
        for key in self.internal_var_keys:
            internal_vars[key] = self.internal_variable_models[key](cond_feed)
        #
        # obtain visible states by Gibbs sampling
        v_data = visible_feed
        v_self, ph_self, ph_data = self.build_gibbs_chain(v_data, internal_vars, k)
        return v_self


    def build_training_graph(self, visible_feed, cond_feed, k):
        """ Build a graph for training the RBM using k steps of Gibbs sampling
            visible_feed: (N, num_visible, 1) tensor of visible samples from data
            cond_feed: (N, ?) tensor of conditioning variables.
            k: int, number of Gibbs sampling steps."""

        if len(visible_feed.shape)!=3:
            visible_feed = tf.expand_dims(visible_feed, 2)

        internal_vars = adict()
        # compute the values of the internal variables.
        for key in self.internal_var_keys:
            internal_vars[key] = self.internal_variable_models[key](cond_feed)
        #
        # obtain visible states by Gibbs sampling
        v_data = visible_feed
        v_self, ph_self, ph_data = self.build_gibbs_chain(v_data, internal_vars, k)
        #compute the internal gradients
        internal_grads = self.internal_gradients(v_data, ph_data,
                                                v_self, ph_self)
        #yo, minimize this thing
        J = self.effective_cost_function(internal_grads, internal_vars)
        return J
