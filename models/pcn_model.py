from ngclearn.utils.model_utils import scanner
from ngcsimlib.compilers import compile_command, wrap_command
from ngcsimlib.context import Context
from ngclearn.utils.io_utils import makedir
from jax import numpy as jnp, random, jit
from ngclearn.components import GaussianErrorCell as ErrorCell, RateCell, HebbianSynapse, StaticSynapse
import ngclearn.utils.weight_distribution as dist
import numpy as np  
import jax

class PCN():
    """
    Structure for constructing the predictive coding network (PCN) in:

    Whittington, James CR, and Rafal Bogacz. "An approximation of the error
    backpropagation algorithm in a predictive coding network with local hebbian
    synaptic plasticity." Neural computation 29.5 (2017): 1229-1262.

    | Node Name Structure:
    | z0 -(W1)-> e1, z1 -(W1)-> e2, z2 -(W3)-> e3;
    | e2 -(E2)-> z1 <- e1, e3 -(E3)-> z2 <- e2
    | Note: W1, W2, W3 -> Hebbian-adapted synapses

    Args:
        dkey: JAX seeding key

        in_dim: input dimensionality

        out_dim: output dimensionality

        hid1_dim: dimensionality of 1st layer of internal neuronal cells

        hid2_dim: dimensionality of 2nd layer of internal neuronal cells
        
        hid3_dim: dimensionality of 3rd layer of internal neuronal cells

        T: number of discrete time steps to simulate neuronal dynamics

        dt: integration time constant

        tau_m: membrane time constant of hidden/internal neuronal layers

        act_fx: activation function to use for internal neuronal layers

        eta: Hebbian learning rate

        exp_dir: experimental directory to save model results

        model_name: unique model name to stamp the output files/dirs with

        save_init: save model at initialization/first configuration time (Default: True)
    """
    def __init__(self, dkey, in_dim=1, out_dim=1, hid1_dim=128, hid2_dim=64, hid3_dim = 32, T=10,
                 dt=1., tau_m=10., act_fx="tanh", eta=0.001, exp_dir="exp",
                 model_name="pc_disc", loadDir=None, **kwargs):
        self.exp_dir = exp_dir
        self.model_name = model_name
        self.nodes = None
        makedir(exp_dir)
        makedir(exp_dir + "/filters")

        dkey, *subkeys = random.split(dkey, 10)

        self.T = T
        self.dt = dt
       
        optim_type = "adam"
        wlb = -0.3
        wub = 0.3

        if loadDir is not None:
            
            self.load_from_disk(loadDir)
        else:
            with Context("Circuit") as self.circuit:
                self.z0 = RateCell("z0", n_units=in_dim, tau_m=0., act_fx="identity")
                self.z1 = RateCell("z1", n_units=hid1_dim, tau_m=tau_m, act_fx=act_fx,
                                   prior=("gaussian", 0.), integration_type="euler")
                self.e1 = ErrorCell("e1", n_units=hid1_dim)
                self.z2 = RateCell("z2", n_units=hid2_dim, tau_m=tau_m, act_fx=act_fx,
                                   prior=("gaussian", 0.), integration_type="euler")
                self.e2 = ErrorCell("e2", n_units=hid2_dim)
                self.z3 = RateCell("z3", n_units=hid3_dim, tau_m=tau_m, act_fx=act_fx,
                                   prior=("gaussian", 0.), integration_type="euler")
                self.e3 = ErrorCell("e3", n_units=hid3_dim)
                self.z4 = RateCell("z4", n_units=out_dim, tau_m=0., act_fx="identity")
                self.e4 = ErrorCell("e4", n_units=out_dim)


               
                self.W1 = HebbianSynapse("W1", shape=(in_dim, hid1_dim), eta=eta,
                                         weight_init=dist.uniform(amin=wlb, amax=wub),
                                         bias_init=dist.constant(value=0.), w_bound=0.,
                                         optim_type=optim_type, sign_value=-1.,
                                         key=subkeys[4])
                self.W2 = HebbianSynapse("W2", shape=(hid1_dim, hid2_dim), eta=eta,
                                         weight_init=dist.uniform(amin=wlb, amax=wub),
                                         bias_init=dist.constant(value=0.), w_bound=0.,
                                         optim_type=optim_type, sign_value=-1.,
                                         key=subkeys[5])
                self.W3 = HebbianSynapse("W3", shape=(hid2_dim, hid3_dim), eta=eta,
                                         weight_init=dist.uniform(amin=wlb, amax=wub),
                                         bias_init=dist.constant(value=0.), w_bound=0.,
                                         optim_type=optim_type, sign_value=-1.,
                                         key=subkeys[6])
                self.W4 = HebbianSynapse("W4", shape=(hid3_dim, out_dim), eta=eta,
                                         weight_init=dist.uniform(amin=wlb, amax=wub),
                                         bias_init=dist.constant(value=0.), w_bound=0.,
                                         optim_type=optim_type, sign_value=-1.,
                                         key=subkeys[7])

                
                self.E2 = StaticSynapse("E2", shape=(hid2_dim, hid1_dim),
                                         weight_init=dist.uniform(amin=wlb, amax=wub),
                                         key=subkeys[4])
                self.E3 = StaticSynapse("E3", shape=(hid3_dim, hid2_dim),
                                        weight_init=dist.uniform(amin=wlb, amax=wub),
                                        key=subkeys[5])
                self.E4 = StaticSynapse("E4", shape=(out_dim, hid3_dim),
                                        weight_init=dist.uniform(amin=wlb, amax=wub),
                                        key=subkeys[6])

                
                self.W1.inputs << self.z0.zF
                self.e1.mu << self.W1.outputs
                self.e1.target << self.z1.z
                self.W2.inputs << self.z1.zF
                self.e2.mu << self.W2.outputs
                self.e2.target << self.z2.z
                self.W3.inputs << self.z2.zF
                self.e3.mu << self.W3.outputs
                self.e3.target << self.z3.z
                self.W4.inputs << self.z3.zF
                self.e4.mu << self.W4.outputs
                self.e4.target << self.z4.z


                
                self.E2.inputs << self.e2.dmu
                self.z1.j << self.E2.outputs
                self.z1.j_td << self.e1.dtarget
                
                self.E3.inputs << self.e3.dmu
                self.z2.j << self.E3.outputs
                self.z2.j_td << self.e2.dtarget
                
                self.E4.inputs << self.e4.dmu
                self.z3.j << self.E4.outputs
                self.z3.j_td << self.e3.dtarget
                


                
                self.W1.pre << self.z0.zF
                self.W1.post << self.e1.dmu
               
                self.W2.pre << self.z1.zF
                self.W2.post << self.e2.dmu
               
                self.W3.pre << self.z2.zF
                self.W3.post << self.e3.dmu
                
                self.W4.pre << self.z3.zF
                self.W4.post << self.e4.dmu


                
                self.q0 = RateCell("q0", n_units=in_dim, tau_m=0., act_fx="identity")
                self.q1 = RateCell("q1", n_units=hid1_dim, tau_m=0., act_fx=act_fx)
                self.q2 = RateCell("q2", n_units=hid2_dim, tau_m=0., act_fx=act_fx)
                self.q3 = RateCell("q3", n_units=hid3_dim, tau_m=0., act_fx=act_fx)
                self.q4 = RateCell("q4", n_units=out_dim, tau_m=0., act_fx="identity")
                self.eq4 = ErrorCell("eq4", n_units=out_dim)
                self.Q1 = StaticSynapse("Q1", shape=(in_dim, hid1_dim),
                                        bias_init=dist.constant(value=0.),
                                        key=subkeys[0])
                self.Q2 = StaticSynapse("Q2", shape=(hid1_dim, hid2_dim),
                                        bias_init=dist.constant(value=0.),
                                        key=subkeys[1])
                self.Q3 = StaticSynapse("Q3", shape=(hid2_dim, hid3_dim),
                                        bias_init=dist.constant(value=0.),
                                        key=subkeys[2])
                self.Q4 = StaticSynapse("Q4", shape=(hid3_dim, out_dim),
                                        bias_init=dist.constant(value=0.),
                                        key=subkeys[3])
               
                self.Q1.inputs << self.q0.zF
                self.q1.j << self.Q1.outputs
                self.Q2.inputs << self.q1.zF
                self.q2.j << self.Q2.outputs
                self.Q3.inputs << self.q2.zF
                self.q3.j << self.Q3.outputs
                self.Q4.inputs << self.q3.zF
                self.q4.j << self.Q4.outputs

                
                self.eq4.target << self.q4.z

                reset_cmd, reset_args = self.circuit.compile_by_key(
                                                self.q0, self.q1, self.q2, self.q3, self.q4, self.eq4,
                                                self.z0, self.z1, self.z2, self.z3, self.z4,
                                                self.e1, self.e2, self.e3, self.e4,
                                            compile_key="reset")
                advance_cmd, advance_args = self.circuit.compile_by_key(
                                                    self.E2, self.E3, self.E4,
                                                    self.z0, self.z1, self.z2, self.z3, self.z4,
                                                    self.W1, self.W2, self.W3, self.W4,
                                                    self.e1, self.e2, self.e3, self.e4,
                                                compile_key="advance_state") 
                evolve_cmd, evolve_args = self.circuit.compile_by_key(
                                                    self.W1, self.W2, self.W3, self.W4,
                                                compile_key="evolve") 
                project_cmd, project_args = self.circuit.compile_by_key(
                                                    self.q0, self.Q1, self.q1, self.Q2,
                                                    self.q2, self.Q3, self.q3, self.Q4, self.q4, self.eq4,
                                                compile_key="advance_state", name="project") 
                self.dynamic()

    def dynamic(self):
        vars = self.circuit.get_components("q0", "q1", "q2", "q3", "q4", "eq4",
                                           "Q1", "Q2", "Q3", "Q4",
                                           "z0", "z1", "z2", "z3", "z4",
                                           "e1", "e2", "e3", "e4",
                                           "W1", "W2", "W3", "W4", "E2", "E3", "E4")
        (self.q0, self.q1, self.q2, self.q3, self.q4, self.eq4, self.Q1, self.Q2, self.Q3, self.Q4,
         self.z0, self.z1, self.z2, self.z3, self.z4, self.e1, self.e2, self.e3, self.e4, self.W1,
         self.W2, self.W3, self.W4, self.E2, self.E3, self.E4) = vars
        self.nodes = vars

        self.circuit.add_command(wrap_command(jit(self.circuit.reset)), name="reset")
        self.circuit.add_command(wrap_command(jit(self.circuit.advance_state)), name="advance")
        self.circuit.add_command(wrap_command(jit(self.circuit.evolve)), name="evolve")
        self.circuit.add_command(wrap_command(jit(self.circuit.project)), name="project")

        @Context.dynamicCommand
        def clamp_input(x):
            self.z0.j.set(x)
            self.q0.j.set(x)

        @Context.dynamicCommand
        def clamp_target(y):
            self.z4.j.set(y)

        @Context.dynamicCommand
        def clamp_infer_target(y):
            self.eq4.target.set(y)

    def save_to_disk(self, params_only=False):
        """
        Saves current model parameter values to disk

        Args:
            params_only: if True, save only param arrays to disk (and not JSON sim/model structure)
        """
        if params_only == True:
            model_dir = "{}/{}/custom".format(self.exp_dir, self.model_name)
            self.W1.save(model_dir)
            self.W2.save(model_dir)
            self.W3.save(model_dir)
            self.W4.save(model_dir)
        else:
            self.circuit.save_to_json(self.exp_dir, self.model_name) 

    def load_from_disk(self, model_directory):
        """
        Loads parameter/config values from disk to this model

        Args:
            model_directory: directory/path to saved model parameter/config values
        """
        print(" > Loading model from ",model_directory)
        with Context("Circuit") as circuit:
            self.circuit = circuit
          
            self.circuit.load_from_dir(model_directory)
          
            self.dynamic()

    def process(self, obs, lab, adapt_synapses=True):
        """
        Runs one pass of inference and learning.

        Args:
            obs: Input observation (data).
            lab: Target label.
            adapt_synapses: Whether to adapt the synapses (learn).
        """
        eps = 0.001
        _lab = jnp.clip(lab, eps, 1. - eps)
        self.circuit.reset()

        
        self.Q1.weights.set(self.W1.weights.value)
        self.Q1.biases.set(self.W1.biases.value)
        self.Q2.weights.set(self.W2.weights.value)
        self.Q2.biases.set(self.W2.biases.value)
        self.Q3.weights.set(self.W3.weights.value)
        self.Q3.biases.set(self.W3.biases.value)
        self.Q4.weights.set(self.W4.weights.value)
        self.Q4.biases.set(self.W4.biases.value)

       
        self.E2.weights.set(jnp.transpose(self.W2.weights.value))
        self.E3.weights.set(jnp.transpose(self.W3.weights.value))
        self.E4.weights.set(jnp.transpose(self.W4.weights.value))

        
        self.circuit.clamp_input(obs)
        self.circuit.clamp_infer_target(_lab)

        self.circuit.project(t=0., dt=1.) 

       
        self.z1.z.set(self.q1.z.value)
        self.z2.z.set(self.q2.z.value)
        self.z3.z.set(self.q3.z.value)
        
        self.e4.dmu.set(self.eq4.dmu.value)
        self.e4.dtarget.set(self.eq4.dtarget.value)
      
        y_mu_inf = self.q4.z.value

        EFE = 0. 
        y_mu = 0.
        if adapt_synapses == True:
           
            for ts in range(0, self.T):
                self.circuit.clamp_input(obs) 
                self.circuit.clamp_target(_lab) 
                self.circuit.advance(t=ts, dt=1.)

            y_mu = self.e4.mu.value
            L1 = self.e1.L.value
            L2 = self.e2.L.value
            L3 = self.e3.L.value
            L4 = self.e4.L.value
            EFE = L4 + L3 + L2 + L1

            
            if adapt_synapses == True:
               
                self.circuit.evolve(t=self.T, dt=1.)
        
        return y_mu_inf, y_mu, EFE

    def predict(self, X):
        """
        Predict labels for input data X.

        Args:
            X: Input data (numpy array).

        Returns:
            Predictions (numpy array).
        """
       
        X = jnp.asarray(X)
        
        dummy_label = jnp.zeros((X.shape[0], self.q4.n_units))  

        
        y_mu_inf, y_mu, EFE = self.process(X, dummy_label, adapt_synapses=False)  


       
        return np.array(y_mu_inf) 

    def evaluate(self, X, y):
        """
        Evaluate the model's accuracy.

        Args:
            X: Input data (numpy array).
            y: Target labels (numpy array, one-hot encoded).

        Returns:
            Accuracy (float).
        """
        
        X = jnp.asarray(X)
        y = jnp.asarray(y)

      
        y_mu_inf, y_mu, EFE = self.process(X, y, adapt_synapses=False)

       
        predicted_classes = np.argmax(np.array(y_mu_inf), axis=1)

       
        true_classes = np.argmax(np.array(y), axis=1)

       
        accuracy = np.mean(predicted_classes == true_classes) * 100
        return accuracy

    def train(self, X, y, epochs=10):
        """
        Train the model.

        Args:
            X: Input data (numpy array).
            y: Target labels (numpy array, one-hot encoded).
            epochs: Number of training epochs.
        """
        for epoch in range(epochs):
            
            X = jnp.asarray(X)
            y = jnp.asarray(y)
            for i in range(X.shape[0]):  
                obs = jnp.asarray(X[i:i+1]) 
                lab = jnp.asarray(y[i:i+1]) 
                y_mu_inf, y_mu, EFE = self.process(obs, lab, adapt_synapses=True) 
            if epoch % 1 == 0:
                accuracy = self.evaluate(X, y)
                print(f"Epoch {epoch}, Accuracy: {accuracy:.2f}%")

    def get_latents(self):
        return self.q3.z.value

    def _get_norm_string(self):
        _W1 = self.W1.weights.value
        _W2 = self.W2.weights.value
        _W3 = self.W3.weights.value
        _b1 = self.W1.biases.value
        _b2 = self.W2.biases.value
        _b3 = self.W3.biases.value
        _norms = "W1: {} W2: {} W3: {}\n b1: {} b2: {} b3: {}".format(jnp.linalg.norm(_W1),
                                                                      jnp.linalg.norm(_W2),
                                                                      jnp.linalg.norm(_W3),
                                                                      jnp.linalg.norm(_b1),
                                                                      jnp.linalg.norm(_b2),
                                                                      jnp.linalg.norm(_b3))
        return _norms

def eval_model(model, X_test, y_test):
    """
    Function to evaluate the performance of the model.

    Args:
        model: The trained model to evaluate.
        X_test: Test data features.
        y_test: Test data labels.
    
    Returns:
        Accuracy of the model.
    """
    accuracy = model.evaluate(X_test, y_test)
    print(f"Test Accuracy: {accuracy:.2f}%")
    return accuracy