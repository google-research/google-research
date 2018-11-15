# Time-varying Convex Optimization

(Work in progress)

For background on time-varying convex optimization problems implemented in this package please see the [paper][paper_location].

Contacts:

* bachir009@gmail.com
* sindhwani@google.com

[TOC]

To see the iPython document in Colaboratory from your web browser, please
navigate to [this page](https://colab.research.google.com/github/google-research/google-research/blob/master/tvsdp/tvsdp.ipynb).


## TV-SDP: Time-varying Semi-definite Programs

The TV-SDP framework can be used for imposing constraints in an optimization problem of the form

$$X(t) \succeq 0 \; \forall t \in [0, 1],$$

where $$X(t)$$ is a polynomial symmetric matrix, i.e. a symmetric matrix whose
entries are polynomial functions of time, and $$X(t) \succeq 0$$ means that all
the eigenvalues values of the matrix $$X(t)$$ are non-negative.

The polynomial matrix $$X $$ is represented by a 3D-tensor $$X$$ of shape $$(m,
m, d+1)$$, where $$m \times m$$ is the shape of the matrix $$X$$ and $$d$$ is
the degree of its components. In other words, the polynomial matrix

$$X(t) = \begin{pmatrix}X_{11}(t) & \ldots & X_{m1}(t)\\&\vdots&\\
X_{1m}(t)&\ldots&X_{mm}(t)\end{pmatrix} $$

is represented by the $$m \times m \times d $$ tensor $$X $$ such that
$$X_{ij}(t) = \sum_{k=1}^{d+1} X_{i, j, k} t^{k-1}. $$

This framework is useful for solving optimization programs whose data and or
objective vary with time, i.e programs of the type

$$
\begin{eqnarray*}
\underset{x(t)}{\sup} & \int_0^1 c(t)^Tx(t) \; {\rm d}t & \\
\text{subject to}&  F[x] (t)  + G[\frac{d}{dt}x] (t)  \succeq 0 \; \forall t \in [0, 1]
\end{eqnarray*}
$$

where $$F[x] (t) := A_0(t) + \sum_{i=1}^nx_i(t) A_i(t)$$, $$G[\frac{d}{dt}x]
(t) := B_0(t) + \sum_{i=1}^n\frac{d}{dt} x_i(t) B_i(t)$$.

One can think of the constraint $$F [x] (t) \succeq 0$$ as a describing a
feasible set that changes shapes as time $$t$$ progresses and $$x(t)$$ tries to
stay inside of it. The constraint $$G[\frac{d}{dt}x](t) \succeq 0$$ imposes
additional restrictions on the derivative of $$x(t)$$.

As a numerical example, take $$n=2$$ and data

$$A_0(t) = \begin{pmatrix}(1-\frac85t)^2&0&0&0\\0&0&0&0\\0&0&0&0\\0&0&0&0\end{pmatrix}, A_1(t) = \begin{pmatrix}-1&0&0&0\\0&0&1&0\\0&1&0&0\\0&0&0&0\end{pmatrix}, A_2(t) = \begin{pmatrix}0&0&0&0\\0&0&0&1\\0&0&0&0\\0&1&0&0\end{pmatrix},$$

$$B_0(t) = B_1(t) = B_2(t) = 0 \text{ and } c(t) = \begin{pmatrix}9t^2 - 9t + 1\\23t^3 - 34t^2 + 12t\end{pmatrix}.$$

A visual representation is below. The feasible set $$\{x \in \mathbb R^2 \; |\;
A_0(t) + x_1 A_1(t) + x_2A_2(t) \succeq 0 \}$$ for some sample times $$t$$ is
delimited by blue lines. The objective function $$c(t)$$ is represented by a
black arrow, which also moves in time. The best polynomial solution of degree 20
is plotted with dotted red lines.

![Geometry of TV-SDP](https://storage.googleapis.com/bachirelkhadir.com/time-varying-semidefinite-programs/example_tvsdp.png){width="600" height="300"}

*Figure: The feasible set, objective and solution of a TV-SDP.

## Time-varying Linear Programming and Time-varying Convex QPs
A general time-varying convex quadratic program corresponds to an optimization
problem of the form:

$$
\begin{eqnarray*}
\underset{x(t)}{\sup} & \int_0^1 c(t)^Tx(t) \; {\rm d}t & \\
\text{subject to} &\quad  y(t)^TQ_i(t)y(t) + a_i(t)^Ty(t) +  b_i(t) \le 0\; i = 1, \ldots, m, \; \forall t \in [0, 1],
\end{eqnarray*}
$$

where $$y(t) = \begin{pmatrix}x(t)\\ \frac{d}{dt}x(t)\end{pmatrix}$$, the
$$Q_i(t)$$ are some convex polynomial matrices, the $$a_i(t)$$ are vectors of
polynomials, and the $$b_i(t)$$ are scalar-valued polynomials. The case where
all the $$Q_i(t)$$ are equal to 0 corresponds to time-varying linear programs.

Since every $$Q_i(t)$$ is convex, it can be decomposed as $$V_i(t)^TV_i(t)$$. By
[Schur complement][schur], the constraint

$$y(t)^TQ_i(t)y(t) + a_i(t)^Ty(t) +  b_i(t) \le 0$$

can be reformulated as

$$\begin{pmatrix}-a_i(t)^Ty(t)-b_i(t) & (V_i(t)y(t))^T \\ V_i(t)y(t) & I\end{pmatrix} \succeq 0,$$

which is exactly the type of constraints that Time-Varying-SDPs can handle.

Applications of TV-SDPs to a "time-varying maxflow" and a "wireless coverage problem" pictured
below are explained in the [paper][paper_location].

![Time-varyig maxflow](https://storage.googleapis.com/bachirelkhadir.com/time-varying-semidefinite-programs/tv-maxflow.png){width="350" height="350"}
![Wireless coverage problem](https://storage.googleapis.com/bachirelkhadir.com/time-varying-semidefinite-programs/tv-wireless.png){width="500" height="300"}




# How to use the library?

The main function of the library is
`make_poly_matrix_psd_on_0_1(X)` , which takes an $$(m, m,
d+1)\text{-tensor}$$ $$X$$ representing a symmetric polynomial matrix $$A(t)$$,
and returns a list of constraints that can be directly fed to CVXPY to impose
$$A(t) \succeq 0\; \forall t\in[0, 1]$$.

## Basic API

```python

import cvxpy

# makes a 3x3 matrix of polynomials of degree 10
m = 3
d = 10
X = cvxpy.Variable( (m, m, d+1), name='X')

# make it PSD on [0, 1]
constraints = []
constraints += make_poly_matrix_psd_on_0_1(X)

# add other constraints, possibly involving other variables
constraints += ...

# Minimize X_12(0)
objective = cvxpy.Minimize(X[1, 2, 0])

# Solve the problem using the solver SCS
prob = cvxpy.Problem(objective, constraints)
prob.solve(solver=cvxpy.SCS, verbose=True)

```

[paper_location]: https://arxiv.org/pdf/1808.03994.pdf
[schur]: https://en.wikipedia.org/wiki/Schur_complement
