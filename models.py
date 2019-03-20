
import tensorflow as tf
import tensorflow_probability as tfp

class adict(dict):

    def __init__(self, **items):
        super(adict, self).__init__(**items)
    def __getattr__(self, attr):
        return self[attr]
    def __setattr__(self, attr, val):
        self[attr] = val

def expand_and_tile(x, batch_size):
    """add a zeroth dimension and tile to match batch size"""
    x = tf.expand_dims(x, 0)
    return tf.tile(x, [batch_size, 1, 1])

class LocalNoiseRBM:
    """ RBM which allows for independent, spatially uniform noise processes."""

    count = 0

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

        while True:

          try:
            self.visible_bias = tf.get_variable("visible_bias%d"%LocalNoiseRBM.count,
                                                shape=(self.num_visible, 1),
                                                dtype=self.dtype,
                                                initializer=bias_initializer)

            self.hidden_bias = tf.get_variable("hidden_bias%d"%LocalNoiseRBM.count,
                                            shape=(self.num_hidden, 1),
                                            dtype=self.dtype,
                                            initializer=bias_initializer)

            self.weights =   tf.get_variable("weights%d"%LocalNoiseRBM.count,
                                    shape= (self.num_visible, self.num_hidden),
                                    dtype=self.dtype,
                                    initializer=kernel_initializer)

            self.noise_kernel =  tf.get_variable("noise_kernel%d"%LocalNoiseRBM.count,
                                        dtype=self.dtype,
                                        initializer=2 * alpha * tf.ones(())
                                        )

            self.noise_bias =   tf.get_variable("noise_bias%d"%LocalNoiseRBM.count,
                                                dtype=self.dtype,
                                                initializer= - alpha * tf.ones(())
                                                )


            self.variables = dict(visible_bias=self.visible_bias,
                                hidden_bias = self.hidden_bias,
                                weights = self.weights,
                                noise_kernel = self.noise_kernel,
                                noise_bias = self.noise_bias)
            break
          except ValueError:
            LocalNoiseRBM.count += 1


    def compute_noisy_probs(self, visible):
        """ Compute the probability of excitation in the noisy register,
        for a given visible setting """
        return tf.sigmoid(self.noise_kernel * visible + self.noise_bias)

    def compute_visible_probs(self, hidden, weights, visible_bias,
                        noise_condition=None):
        """ Given a hidden tensor, return the excitation probabilities for the
        visible layer.
            noise_condition: if not None, (N, num_visible tensor) of noisy
            register values.
            """
        if len(hidden.shape)==3:
            batch_size = tf.shape(hidden)[0]
            weights = expand_and_tile(weights, batch_size)
            visible_bias = expand_and_tile(visible_bias, batch_size)

        # standard RBM formula based for the energy, given hiddens
        energy = visible_bias + tf.matmul(weights, hidden)
        if noise_condition is not None:
            # if we're conditioning on the noisy state, energy needs to be modified.
            energy += (tf.math.softplus(self.noise_bias)
                        - tf.math.softplus(self.noise_bias + self.noise_kernel)
                        + self.noise_kernel * noise_condition)
        return tf.sigmoid(energy)


    def compute_hidden_probs(self, visible, weights, hidden_bias):
        """ Given tensor of visible activations and couplings, return activation
        probabilities for hidden layer.

        """
        if len(visible.shape)==3:
            batch_size = tf.shape(visible)[0]
            weights = expand_and_tile(weights, batch_size)
            hidden_bias = expand_and_tile(hidden_bias, batch_size)
        return tf.sigmoid( hidden_bias +
            tf.matmul(tf.transpose(weights,perm=[0,2,1]), visible))

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


        ### I'm repeating myself, there's a better way to do this...
        batch_size = tf.shape(visible_init)[0]

        weights = expand_and_tile(self.weights, batch_size)
        hidden_bias = expand_and_tile(self.hidden_bias, batch_size)
        visible_bias = expand_and_tile(self.visible_bias, batch_size)

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
                                                    noise_condition=noise_setting)
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
        given visibles
        Inputs should have shape (batch_size, nv, 1) """
        #prob of noisy activation, given hiddens
        pnoisy = self.compute_noisy_probs(visible)
        grads = adict()
        grads.noise_bias = tf.reduce_sum(noisy - pnoisy, axis=1)
        grads.noise_kernel = tf.reduce_sum(visible * (noisy-pnoisy), axis=1)
        return grads

    def log_probability_gradients(self, v_data, ph_data,
                                    v_self, ph_self,
                                    noisy_state):
        """ Compute log prob gradients with respect to internal parameters.
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

        data_grads = self.free_energy_gradients(v_data, ph_data)
        self_grads = self.free_energy_gradients(v_self, ph_self)
        conditional_grads = self.log_conditional_prob_gradients(v_data,
                                                            noisy_state)
        grads = adict()
        for key in data_grads.keys():
            grads[key] = self_grads[key] - data_grads[key]
        for key in conditional_grads.keys():
            grads[key] = conditional_grads[key]
        return grads

    def build_sampler(self, visible_feed,  k
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

    def estimate_logprob_grads(self, data_feed, k,
                                    noise_condition=True,
                                    persistent_state=None):
        """ Build a graph for estimating gradients of log-probabability
        with respect to RBM parameters
            using k steps of Gibbs sampling
            data_feed: (N, num_visible, 1) tensor of visible samples from data
            cond_feed: (N, ?) tensor of conditioning variables.
            k: int, number of Gibbs sampling steps.
            noise_condition: whether or not to treat visible samples as noisy
            returns: adict of gradients of the log-probability of the data,
            averaged over samples.
            persistent_state: if not None, tensor of persistent
            'fantasy' visible states to use for seeding the negative phase

            returns: grads, persistent_state"""

        if len(data_feed.shape)!=3:
            data_feed = tf.expand_dims(data_feed, 2)

        #what to use to seed the gibbs chains for the negative phase
        self_seed = data_feed if persistent_state is None else persistent_state

        #if conditioning on noise layer, need to run sampling to infer visible
        # states
        if noise_condition:
            v_data, ph_data, __ = self.build_gibbs_chain(data_feed, k,
                                                noise_condition=True)
            v_self, ph_self, __ = self.build_gibbs_chain(self_seed, k,
                                                noise_condition=False)
        ### otherwise, visible states are driven directly from data
        else:
            v_data = data_feed
            ## make sure to get the hidden probs conditioned on the data
            ph_data = self.compute_hidden_probs(v_data, self.weights, self.hidden_bias)
            v_self, ph_self, ph_data = self.build_gibbs_chain(self_seed, k,
                                                        noise_condition=False)
        noisy_state = data_feed

        #compute the internal gradients
        grads = self.log_probability_gradients(v_data, ph_data,
                                                v_self, ph_self, noisy_state)
        # average over examples in batch
        for key in grads.keys():
            grads[key] = tf.reduce_mean(grads[key], axis=0)
        grads.noise_kernel = tf.squeeze(grads.noise_kernel)
        grads.noise_bias = tf.squeeze(grads.noise_bias)

        #if applicable, return new persistent state
        return grads, v_self

    def nll_gradients(self, data_feed, k,
                                            weight_decay = 0.0,
                                            noise_condition=True,
                                          persistent_state=None):
        """ Returns list of ( gradient, variable) pairs, where
            variables are the trainable params of the rbm, and
            gradients are the gradients of the negative-log-likelihood cost
            function with respect to the paired variables.

            persisitent_state: if not None, a tensor of visible-states used
            to seed the self-phase chain

            """

        logprob_grads, new_persistent_state = self.estimate_logprob_grads(data_feed, k,
                                            noise_condition=noise_condition,
                                            persistent_state=persistent_state)

        #add weight decay term by hand:
        logprob_grads['weights'] -= 2 * weight_decay * self.weights
        gradlist = []
        for varname in self.variables.keys():
            gradlist += [(-logprob_grads[varname], self.variables[varname])]
        return gradlist, new_persistent_state
