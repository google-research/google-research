# Installation

Clone the repository and run the following commands from the repository root.

A development environment with all the dependencies can be created using conda as

```
$> conda env create -f dependencies.yaml -n factoring
$> conda activate factoring
```

The library can be installed using

```
$> pip install .
```

# Running the program
The `main.py` binary can be executed with appropriate command line arguments to either factor a known integer `N`, or factor a randomly generated integer of bitsize `b`. 

The usage can be seen as follows:
```bash
$> python factoring_sqif/main.py --help
usage: Integer factoring using Schnorr's algorithm + quantum optimization. [-h] (-N NUMBER_TO_FACTOR | -b BITSIZE) [-l LATTICE_PARAMETER] [-c PRECISION_PARAMETER] [-s SEED] [-m {qaoa,bruteforce}]
                                                                           [-p QAOA_DEPTH] [-NS NUM_SAMPLES]

Implementation of integer factoring algorithm described in https://arxiv.org/abs/2212.12372

options:
  -h, --help            show this help message and exit
  -N NUMBER_TO_FACTOR, --number_to_factor NUMBER_TO_FACTOR
                        Integer to factor.
  -b BITSIZE, --bitsize BITSIZE
                        Bitsize of number to be factored.
  -l LATTICE_PARAMETER, --lattice_parameter LATTICE_PARAMETER
                        Lattice parameter.
  -c PRECISION_PARAMETER, --precision_parameter PRECISION_PARAMETER
                        Precision parameter.
  -s SEED, --seed SEED  Seed for random number generation.
  -m {qaoa,bruteforce}, --method {qaoa,bruteforce}
                        Method to use for finding ground states of the hamiltonian.
  -p QAOA_DEPTH, --qaoa_depth QAOA_DEPTH
                        Depth of qaoa circuit.
  -NS NUM_SAMPLES, --num_samples NUM_SAMPLES
                        Number of low energy states to sample
```

For example, you can run the following to factorize `187` into `(17, 11)` using SQIF + brute force.

```bash
$> python factoring_sqif/main.py -N 187
```

# Configurable parameters and heuristics in this implementation
## Lattice Dimension
One of the primary reasons why Schnorr's algorithm is very widely debated, and why it was used as a starting point for qauntum optimization in the above paper, was that it claims that one can find enough good sr-pairs using a lattice dimension that scales only sublinearly in the bitsize of the number $N$ to be factored. 

From discussions in classical cryptography community, it is not clear that solving the CVP problem on lattices of claimed dimension can even produce enough good SR pairs. See the following references for discussions:
- https://crypto.stackexchange.com/a/88601
- https://twitter.com/inf_0_/status/1367376959055962112


In the proposed quantum algorithm, since we use quantum optimization methods like QAOA to find better solutions for the CVP on a lattice, the dimension of the lattice corresponds to number of qubits required. The lattice dimension is assumed to be sublinear in bitsize and is controlled by a configurable constant **lattice_parameter** as:

$$
n = \mathrm{lattice\textunderscore parameter} * \log_2(N) / \log_2(\log_2(N))
$$

In general, a higher lattice dimension leads to a higher probability of finding enough good SR pairs (thus leading to a successful factorization) at the cost of increasing the time / space runtime complexity of the algorithm. 

## Precision parameter for lattice construction
The precision parameter $c$ controls the basis vectors used for constructing the lattice. 

In the original Schnorr's algorithm, the proposed lattice construction is 

$$
B_{n,c} = \begin{bmatrix} 
    f(1) & 0 & \dots & 0\\
    0 & f(2) & \dots & 0\\
    \vdots & \vdots & \ddots & \vdots\\
    0 & 0 & \dots & f(n) \\
    N^c\ln{p_1} & N^c\ln{p_2} & \dots & N^c\ln{p_n} 
    \end{bmatrix}
$$

where the functions $f(i)$ for $i = 1, ..., n$ are random permutations of $(\sqrt{\ln{p_1}}, \sqrt{\ln{p_2}}, ..., \sqrt{\ln{p_n}})$.

In this implementation of [arxiv:2212.12372](https://arxiv.org/abs/2212.12372), the lattice construction is modified to be:

$$
B_{n,c} = \begin{bmatrix} 
    f(1) & 0 & \dots & 0\\
    0 & f(2) & \dots & 0\\
    \vdots & \vdots & \ddots & \vdots\\
    0 & 0 & \dots & f(n) \\
    \lfloor10^c\ln{p_1}\rceil & \lfloor10^c\ln{p_2}\rceil & \dots & \lfloor10^c\ln{p_n}\rceil 
    \end{bmatrix}
$$

where the functions $f(i)$ for $i = 1, ..., n$ are random permutations of $(\lceil 1/2 \rceil, \lceil 2/2 \rceil, ..., \lceil n/2 \rceil)$.

The determinant of the lattice is positively correlated with precision parameter $c$. It is conjectured that probability of finding vectors close to the target vector should be positively correlated with $c$, but it's not clear to us that why this should be true. 

## Smoothness bound $B_2$ for $|u - v * N|$ where both $u$ and $v$ are $p_{n}$ smooth.
In Schnorr's algorithm, by construction, every point on the lattice can be mapped to a pair of integers $(u, v)$ s.t. both $u$ and $v$ are $p_{n}$-smooth; i.e. have prime factors less than or equal to the $n$'th prime number $p_{n}$. Here $n$ is the dimension of the lattice. 

For the algorithm to succeed, we need to find "enough" (see next paragraph) SR-pairs $(u, v)$ s.t. both $u$ and $v$ are $p_{n}$ smooth and $|u - v * N|$ is **$p_{B_2}$**-smooth, i.e. $|u - v * N|$ have all prime factors less than or equal to $B_2$'th prime factor. Here $B_2$ is another hyper parameter which should typically be at-least $n$. 

It is typically sufficient that the number of such SR-pairs is a few more than smoothness bound $B_2$. In this implementation, by default, we set the smoothness bound $B_2$ and number of SR-pairs to sample as:

$$
B_2 = 2 * n^2  
$$

$$
\mathrm{num \textunderscore sr \textunderscore pairs \textunderscore to \textunderscore sample} = B_2 + 2
$$

where $n$ is the dimension of the lattice. 

In general, hyper parameter $B_2$ affects the algorithms as:
1) Higher the $B_2$ bound, the easier it is to find SR-pairs $(u, v)$ that satisfy the constraint that $|u - v * N|$ is $B_2$-smooth. Thus it increases the probability of finding valid SR-pairs.
2) However, an increase in $B_2$ also leads to an increase in the requirement for number of SR-pairs to sample, which increases the overall runtime of the algorithm. 

## Quantum optimization formulation for exploring the lattice subspace to solve CVP
The proposed reduction to use quantum optimization methods to find low energy states of a Hamiltonian works is done by a two step process as follows:
1) Find an approximate close vector $b_{op}$ to the given target vector $t$ using classical heuristic methods like Babai's algorithm. 
2) Try to find a better solution by exploring the $2^n$ vectors that lie within the $n$-dimensional hypercube of side length $1$ centered at $b_{op}$
    - This corresponds to representing other "good" close vectors as $|b_{op} + \sum_{i}x_{i}d_{i}|$ where each $x_{i}$ is in {0, 1} or {0, -1} and $d_{i}$ represents the LLL reduced basis of the prime lattice.
    - Thus, the problem of finding good close vectors reduces to optimizing the sequence of $x_{i}$'s s.t. $|t-b_{op}-\sum_{i}x_{i}d_{i}|^2$ is minimized. 
    - This optimization problem is further reduced to finding low energy states of an all-to-all connected Ising sping glass Hamiltonian.

It is not clear that this is the best way of exploring the lattice to find other close vectors and one could come up with other reductions for mapping the CVP problem to a quantum optimization problem.  

## Find low energy states of Ising spin glass Hamiltonian with all-to-all connectivity
### Using Brute Force
One way to find low energy states of the given Hamiltonian $H$ is to simply try all the $2^n$ bitrings as input states, where $n$ is the number of qubits / lattice dimension. This doesn't scale well but is implemented to verify rest of the classical reductions assuming a perfect quantum optimizer. One can run the brute force approach by specifying `method=bruteforce`.
```bash
$> python factoring_sqif/main.py -b 50 --method bruteforce
```

### Using QAOA - [arxiv:2212.12372](https://arxiv.org/abs/2212.12372)
QAOA is a heuristic algorithm and the performance of QAOA depends on the following two configurable heuristics:
1) The depth of the QAOA circuit (i.e. parameter $p$). By default, we use $p=2$ but one can experiment with a higher depth circuit by passing the command line argument `--qaoa_depth` while running the program. 
2) Outer classical optimizer to find the optimal parameters of the variational QAOA circuit. The paper claims to use Model gradient descent, for which an implementation exists in [quantumlib/ReCirq](https://github.com/quantumlib/ReCirq/blob/5f2d927ce8b5028a342508c3c2692e6892174876/recirq/optimize/mgd.py#L78). We tried using the out of the box implementation and it didn't converge for us, but the readers are welcome to try optimization methods. 

One can run the qaoa optimization for factoring a 50-bit integer with `qaoa_depth=3` as:
```bash
$> python factoring_sqif/main.py -b 50 --method qaoa -p 3
```


### Using DCQF - [arxiv:2301.11005](https://arxiv.org/abs/2301.11005)
DCQF is another heuristic algorithm that can be used to find low energy states of the Hamiltonian and, unlike QAOA, it doesn't require any outer classical optimization routines. The linked paper talks more about the specific details of DCQF and it is left as a future improvement to incorporate the implementation in this repository. 


# References

**A note on Integer Factorization using Lattices**    
_A. Vera_
https://arxiv.org/abs/1003.5461

**Schnorr’s Approach to Factoring via Lattices**    
_Léo Ducas_
https://projects.cwi.nl/crypto/risc_materials/2021_05_20_ducas_slides.pdf

**SchnorrGate - Testing Schnorr's factoring Claim in SageMath**    
_Léo Ducas_
https://github.com/lducas/SchnorrGate