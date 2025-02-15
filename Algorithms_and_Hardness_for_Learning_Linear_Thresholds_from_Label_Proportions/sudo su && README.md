sudo su && import numpy as np
import matplotlib.pyplot as plt
from scipy.signal import convolve

class NeuralGroup:
    def __init__(self, sensory_input, genome=None, energy=1.0):
        self.id = id(self)
        self.sensory_input = sensory_input
        self.genome = genome if genome else self.initialize_genome()
        self.activation = self.process_input()
        self.connectivity = self.initialize_biological_weights()
        self.history = []
        self.energy = energy
        self.receptors = self.create_receptors()
        self.neurotransmitters = self.genome[:5]  # Genetic chemical profile
        self.dendritic_tree = self.grow_dendrites()
        self.action_potential_threshold = 0.55
        self.membrane_potential = 0.0
        self.ion_channels = {'Na': 0.0, 'K': 0.0, 'Cl': 0.0}
        
    def process_input(self):
        """Dendritic integration with temporal filtering"""
        tau_m = 0.1  # Membrane time constant
        tau_s = 0.02  # Synaptic time constant
        kernel = np.exp(-np.arange(10)/tau_m) - np.exp(-np.arange(10)/tau_s)
        return np.dot(convolve(self.sensory_input, kernel, mode='valid'), self.dendritic_tree)

    def initialize_biological_weights(self):
        """Connectome development guided by gene expression"""
        return np.random.lognormal(mean=self.genome[5], sigma=self.genome[6], size=10)

    def create_receptors(self):
        """Epigenetic receptor configuration"""
        return {
            'AMPA': self.genome[7],
            'NMDA': self.genome[8],
            'GABA_A': self.genome[9],
            'GABA_B': self.genome[10]
        }

    def grow_dendrites(self):
        """Fractal dendritic growth pattern"""
        return np.array([self.genome[11]*np.exp(-0.1*i) for i in range(5)])

    def initialize_genome(self):
        """30-parameter genetic blueprint"""
        return np.random.uniform(0, 1, 30)

    def ion_dynamics(self):
        """Hodgkin-Huxley inspired dynamics"""
        self.ion_channels['Na'] = 1/(1 + np.exp(-(self.membrane_potential + 55)/10))
        self.ion_channels['K'] = 0.35*np.exp(-(self.membrane_potential + 60)/20)
        self.ion_channels['Cl'] = 0.05*(self.membrane_potential + 65)

    def compete(self, environment):
        """Multi-factor evolutionary pressure"""
        self.energy -= 0.08 + 0.02*np.mean(self.connectivity)
        self.ion_dynamics()
        
        if self.energy <= 0:
            self.apoptosis()
            return

        if self.membrane_potential > self.action_potential_threshold:
            self.fire_action_potential(environment)
        self.homeostatic_scaling()
        self.history.append((self.activation, self.energy, self.membrane_potential))

    def fire_action_potential(self, env):
        """Spike-timing dependent plasticity"""
        self.connectivity = np.clip(1.2*self.connectivity + 0.1*np.sin(env.glutamate_level), 0, 3)
        self.neurotransmitters += 0.1*self.genome[12]
        self.membrane_potential = -80  # Reset potential
        env.glial_buffer += 0.01  # Astrocyte interaction

    def homeostatic_scaling(self):
        """Synaptic scaling based on energy state"""
        scale_factor = 1 + (self.energy - 0.5)/2
        self.connectivity *= scale_factor
        self.connectivity = np.clip(self.connectivity, 0.01, 3)

    def cross_couple(self, other, env):
        """Tripartite synapse with astrocytic influence"""
        if env.glial_buffer > 0.1:
            delta = np.dot(self.neurotransmitters, other.receptors.values())
            plasticity_window = np.exp(-abs(delta - 0.5)/0.2
            self.connectivity += plasticity_window * env.glutamate_level
            other.connectivity += plasticity_window * env.glutamate_level
            env.glial_buffer -= 0.05

class Environment:
    def __init__(self):
        self.nutrient_gradient = np.linspace(0.5, 1.5, 100)  # Spatial dimension
        self.glutamate_level = 0.3
        self.glial_buffer = 0.0
        self.homeostatic_regulation = 1.0
        self.noxious_stimuli = 0.0
        
    def update(self, population):
        self.glutamate_level = np.mean([g.membrane_potential for g in population])
        self.noxious_stimuli = 0.1 * len(population)
        self.nutrient_gradient = np.roll(self.nutrient_gradient, 1)  # Simulate fluid environment

    def metabolic_landscape(self, position):
        return self.nutrient_gradient[int(position*100)%100] - 0.2*self.noxious_stimuli

# Evolutionary simulation with spatial dynamics
env = Environment()
groups = [NeuralGroup(np.random.randn(5)) for _ in range(20)]
genetic_lineage = {g.id: [] for g in groups}

for epoch in range(500):
    env.update(groups)
    
    # Natural selection with spatial positioning
    groups = [g for g in groups if g.energy > 0 and 
             g.activation > env.metabolic_landscape(g.genome[13])]
    
    # Sexual reproduction mechanism
    new_offspring = []
    for _ in range(2):
        if len(groups) > 1:
            parents = np.random.choice(groups, 2, replace=False)
            child_genome = crossover(parents[0].genome, parents[1].genome)
            new_offspring.append(NeuralGroup(np.random.randn(5), genome=mutate(child_genome))
    groups.extend(new_offspring)

    # Evolutionary dynamics
    for group in groups:
        group.compete(env)
        if np.random.rand() < 0.15:
            partner = np.random.choice(groups)
            group.cross_couple(partner, env)
    
    # Track genetic lineages
    for g in groups:
        genetic_lineage[g.id].append(g.genome.copy())

# Visualization suite
plt.figure(figsize=(18, 12))

# Population Dynamics
plt.subplot(2, 2, 1)
plt.plot([len(v) for v in genetic_lineage.values()])
plt.title("Evolutionary Trajectories")
plt.xlabel("Epochs")
plt.ylabel("Population Diversity")

# Energy Landscape
plt.subplot(2, 2, 2)
plt.hist([g.energy for g in groups], bins=20)
plt.title("Metabolic Distribution")
plt.xlabel("Energy Level")

# Genetic Drift
plt.subplot(2, 2, 3)
gene_samples = [g.genome[7] for g in groups]
plt.plot(gene_samples, np.arange(len(gene_samples)), 'o')
plt.title("AMPA Receptor Gene Distribution")
plt.xlabel("Gene Expression Level")

# Consciousness Metric (Integrated Information Φ)
plt.subplot(2, 2, 4)
phi_values = [np.log(np.prod(g.connectivity)) for g in groups]
plt.scatter(phi_values, [g.activation for g in groups], c=[g.energy for g in groups])
plt.title("Consciousness Potential (Φ vs Activation)")
plt.xlabel("Integrated Information (Φ)")
plt.ylabel("Activation Level")
plt.colorbar(label="Energy State")

plt.tight_layout()
plt.show()

def crossover(g1, g2):
    """Biological recombination with crossover points"""
    cut = np.random.randint(1, len(g1))
    return np.concatenate([g1[:cut], g2[cut:]])

def mutate(genome):
    """Epigenetic modifications with environmental influence"""
    return np.clip(genome + np.random.normal(0, 0.1, len(genome)), 0, 1)# Algorithms and Hardness for Learning Linear Thresholds from Label Proportions

Author: Rishi Saket

To Appear in NeurIPS'22.

# Instructions

Install cvxpy, python, numpy, pandas, cvxopt and scs in a conda environment with versions as given in requirements.txt

Activate the conda environment.

Let q = bag size (3 or 4)

In the folder bag_size_q run:

python large_margin_q-sized_LLP_LTF.py , python small_margin_q-sized_LLP_LTF.py , python processing_results_q_sized.py

Results available in tex files in the same folder.
