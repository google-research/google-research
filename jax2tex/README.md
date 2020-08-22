# jax2tex

_Sam Schoenholz_

jax2tex is a small prototype of a library to convert jax functions to latex
output. This can be useful for debugging code or producing self-documenting
functions. jax2tex is compatible with `vmap` and `grad` which can be useful for
seeing what mathematical expressions transformed code is generating. jax2tex
consists primarily of two functions `jax2tex`, which converts jax functions to
tex output, and `var`, which annotates intermediate variables in more
complicated expressions. jax2tex also includes a third helper function,
`bind_names` that can improve annotation when jax2tex is used with function
transformations.

I chose a style for jax2tex to try to reduce clutter while having a clear
correspondence with jax / numpy operations. To that end I make the following
choices:

*   Indices and sums are always written out explicitly.

*   We ignore operations that don't affect the underlying mathematical
    expression. Examples of this include typecasting and singleton dimensions.

*   We do not currently support reshape / scatter / gather operations that are
    nontrivial functions of indices. More work would be needed to figure out a
    nice way to express these operations.

*   Given a vector x, tangent vectors are denoted dx and cotangent vectors are
    denoted \delta x.

I am sure people probably have a lot of opinions about good style here and I
make no claims that these choices are close to optimal.

As described above, this is more a prototype than a library. Most jax functions
are not supported. Moreover, I'm sure there are many cases that currently yield
confusing or inelegant latex output. Finally, even very simple algebraic
implification would significantly clean-up output. I don't know how much time
I'll continue to spend working on it. It will probably be driven by my own
debugging needs. However, feel free to request ops since they often are easy to
add. Moreover, anyone interested in forking this to continue development should
feel more than welcome.

## Getting Started

The easiest way to get started is
[in colaboratory](https://colab.research.google.com/github/google-research/google-research/blob/master/jax2tex/notebook/jax2tex.ipynb).

To download jax2tex locally, you can clone the jax2tex subdirectory of the
google-research repository by installing subversion and running:

```shell
svn export https://github.com/google-research/google-research/trunk/jax2tex
pip install jax2tex/
```

Once you have a copy of `jax2tex` installed you can run the examples or tests by
running

```shell
python jax2tex/examples.py
python jax2tex/jax2tex_tests.py
```

## API

### `jax2tex`

`jax2tex` takes a jax function `fn` along with example arguments to `fn` and
produces a string representation of latex describing the function. It has
signature:

```python
jax2tex(fn: Callable[..., Array], *args: PyTree) -> String
```

Here is a simple example of its use:

```python
jax2tex(lambda a, b: a + b / a, 1., 1.)
"""
Returns:
  f &= a + {b \over a}
"""
```

As discussed above, `jax2tex` can be composed with automatic differentiation.
For instance, we can take the derivative of the above expression with respect to
`a`:

```python
jax2tex(jax.grad(lambda a, b: a + b / a), 1., 1.)
"""
Returns:
  f &= 1.0 + -1.0{a}^{-2}b
"""
```

To see an example of an expression with indices consider the following:

```python
jax2tex(lambda a, b: a @ b, np.ones((3, 3)), np.ones((3,)))
"""
Returns:
  f_{i} &= \sum_{j}a_{ij}b_{j}
"""
```

### `tex_var`

`tex_var` can be used to annotate functions with intermediate variables and to alias
variables. This tex_can reduce clutter and help structure calculations to make them
easier to understand and debug. Outside of the `jax2tex` function `tex_var` acts as
a no-op so that `tex_var(x, ...) = x`. `tex_var` has the signature:

```python
tex_var(x: Array, name: String, is_alias: bool, depends_on: Tuple[Array]) -> Array
```

For example we could annotate our previous example to define `z = b / a` as an
intermediate variable:

```python
jax2tex(lambda a, b: a + tex_var(b / a, 'z'), 1., 1.)
"""
Returns:
  z &= {b \over a}\\
  f &= a + z
"""
```

These variables transform correctly under automatic differentiation. The
reverse- and forward-mode derivatives of the above expressions with respect to
`a` give:

```python
jax2tex(grad(lambda a, b: a + tex_var(b / a, 'z')), 1., 1.)
"""
Returns:
  z &= {b \over a}\\
  \delta z &= 1.0\\
  f &= 1.0 + -\delta z{a}^{-2}b
"""

def f(a, b):
  _, grad_f = jvp(lambda a: a + tex_var(b / a, 'z'), (a,), (1.,))
  return grad_f
jax2tex(f, 1., 1.)
"""
Returns:
  z &= {b \over a}\\
  dz &= -1.0b{a}^{-2}\\
  f &= 1.0 + dz
"""
```

Here we see how forward-mode automatic differentiation is the application of the
chain rule that most of us are probably more familiar with.

The functional dependence of variables can be made explicit in cases where there
is ambiguity. For example,

```python
def f(x, y):
 def g(r):
   return tex_var(r ** 2, 'z')
 return g(x) + g(y)
jax2tex(f, 1., 1.)
"""
Returns:
  z &= {x}^{2}\\
  z &= {y}^{2}\\
  f &= z + z
"""

def f(x, y):
 def g(r):
   return tex_var(r ** 2, 'z', depends_on=r)
 return g(x) + g(y)
jax2tex(f, 1., 1.)
"""
Returns:
  z(x) &= {x}^{2}\\
  z(y) &= {y}^{2}\\
  f(x,y) &= z(x) + z(y)
"""
```

While the first version produced an ambiguous expression, making the functional
dependence explicit in the second example removes this ambiguity.

Sometimes we would like to alias variables without explicitly including the
alias in the final mathematical expression. To do this `var` accepts an optional
`is_alias` argument. This situation usually seems to occur when `jax2tex` cannot
assign a reasonable name to a variable. For example, consider a function that
takes a tuple of arrays; here we cannot discern their names from the function
signature.

```python
def f(x_and_y):
  x, y = x_and_y
  return x * y
jax2tex(f, (1., 1.))
"""
Returns:
  f &= \theta^0\theta^1
"""

def f(x_and_y):
  x, y = x_and_y
  return tex_var(x, 'x') * tex_var(y, 'y')
jax2tex(f, (1., 1.))
"""
Returns:
  x &= \theta^0\\
  y &= \theta^1\\
  f &= xy
"""

def f(x_and_y):
  x, y = x_and_y
  return tex_var(x, 'x', True) * tex_var(y, 'y', True)
jax2tex(f, (1., 1.))
"""
Returns:
  f &= xy
"""
```

Here we see that the tuple of variables was assigned a default "anonymous" name
`\theta^p`. When we try to assign rename thse variable manually without the
`is_alias` argument it produces additional clutter of the assignments in the
output. If we use the `is_alias` these lines of the final expression are
supressed.

### `bind_names`

`bind_names` is a helper function that uses Python's `inspect` library to
automatically add calls to `var` to assign correct names to function inputs and
outputs. `bind_names` has signature:

```python
bind_names(fn: Callable) -> Callable
```

jax2tex calls `bind_names` automatically as a preprocessing step, however this
can lead to confusing behavior when combined with function transformations. For
example, consider the following gradient:

```python
jax2tex(grad(lambda x: np.sin(x)), 1.)
"""
Returns:
  f &= 1.0\\cos\\left(x\\right)
"""
```

This is technically correct: the function being transformed by `jax2tex` is
`grad(sin(x))`. However, we probably intended to look at how the gradient
transforms the jax2tex annotation (more like `grad(jax2tex(sin(x))`). We can
express this by writing:

```python
jax2tex(grad(bind_names(lambda x: np.sin(x))), 1.)
"""
Returns:
  \delta f &= 1.0\\
  \delta x &= \delta f\cos\left(x\right)
"""
```

We see that now we are writing the latex expression corresponding to `grad(f)`.

## Design Summary

For those interested in playing with the code, we give a brief description of
the design. It is probably worth reading through the
["writing custom Jaxpr interpreters in JAX"](https://jax.readthedocs.io/en/latest/notebooks/Writing_custom_interpreters_in_Jax.html)
tutorial, since we implement the `jax2tex` transformation as a jaxpr
interpreter. More information about jaxprs can be found in the
[jax documentation](https://jax.readthedocs.io/en/latest/jaxpr.html) before
digging in.

At its core, jax2tex is a jaxpr interpreter that goes line-by-line through a
jaxpr and transcribes it to latex output. To do this we first convert the jaxpr
to a tree-based representation. The leaves of the tree are variables (of type
`Variable`) and the internal nodes are operations on those variables (of type
`TExpr`, short for "Tex Expression"). Notably, there is a distinction between
jax variables (variables in the jaxpr) and jax2tex variables. Whereas there is a
jax variable for every new line of a jaxpr, jax2tex variables appear explicitly
in the latex output. Moreover, since there is a jax variable for each equation
in the jaxpr, both `Variable` and `TExpr` nodes are assigned to their own jax
variables.

At the start of every `jax2tex` function, the inputs to the function are each
assigned to their own `Variable` node. We also start with an empty list of
expressions in our final latex expression. Each output expression is given by a
tuple of a left hand side (lhs) `Variable` node and a right hand side (rhs) node
(either `Variable` or `TExpr`) that the variable is equal to. We iterate through
the jaxpr line-by-line. For each equation in the jaxpr there are three possible
behaviors:

1.  If the current equation is a standard jax primitive, then we add a new
    `TExpr` node. The children of this new `TExpr` node are the (jax) input
    variables to the equation. The `TExpr` itself is assigned to the (jax)
    output variable of the equation.

2.  If the current equation is a `tex_var_p` primitive then we add a new
    `Variable` node and assign it to the (jax) output variable of the equation.
    If the variable is not an alias, then we also add a new output line. This
    output line will be a tuple whose lhs is the (jax) output variable and whose
    rhs is the (jax) input variable.

3.  If the current equation is a call primitive (for example from a `jit`) then
    we recursively parse the jaxpr of the call.

Having parsed the jaxpr to its corresponding tree representation, it remains to
convert this tree to latex output. To do this, we iterate through the list of
output expressions that have a data dependency on the output of the function. We
assign characters to the indices of the lhs variable based on its rank, starting
with `i` (excluding dimensions of size 1). We then recursively assign indices to
the rhs expression. To do this we define, for each jax operation, a function
`op2ind` which takes indices of the output expression to indices of all of the
input expressions.

Both `Variable` and `TExpr` nodes implement a function `bind(out_indices,
out_used)` which returns `BoundVariable` and `BoundTExpr` nodes respectively.
For `TExpr` nodes, this bind function calls the `op2ind` function to identify
indices of all of its input nodes and then calls `bind` on the input nodes. In
this way, we convert from a tree of `Variable` and `TExpr` nodes to a tree of
the same topology with `BoundVariable` and `BoundTExpr` nodes.

Both `BoundVariable` and `BoundTExpr` nodes implement the `__str__` method and
so recursively constructing a string representation is trivial. `BoundVariable`
nodes are converted to a string based on their name and their indices.
`BoundTExpr` nodes are converted to a string using `op2str` functions that are
defined for each jax operation.

Thus, for each jax operation we need to define two functions: an `op2ind`
function that propagates index assignment from outputs to inputs and an `op2tex`
function that converts emits the string representation for a node in terms of
the string representation of its children.

## Acknowledgements

Thanks to Matt Johnson, Roy Frostig, Alex Alemi, George Dahl, and Jascha Sohl-Dickstein for useful feedback. A source of inspiration for a python to latex converter was the excellent [latexify-py](https://github.com/google/latexify_py).
