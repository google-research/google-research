This is the code covering two papers:

(A) "Boosting with Tempered Exponential Measures"
    Richard Nock, Ehsan Amid and Manfred K. Warmuth
    Advances in Neural Information Processing Systems 2023

(B) "Tempered Calculus for ML: Application to Hyperbolic Model Embedding"
     Richard Nock, Ehsan Amid, Frank Nielsen, Manfred K. Warmuth and Alexander Soen
     ArXiv 2024 (TBD)

Code provided without any warranty, use it at your own risks. See Licence.

====================================================================================
Brief description: 

this code allows to learn boosted combinations of decision trees
to optimize tempered losses (A) and the log-/logistic-loss (B). In the case of (B),
a simple GUI embeds the resulting models in Poincar\'e model of hyperbolic geometry.
In all cases, the program allows to simulate noise in the training data and if 
several algorithms are run simultaneously, automatically compares the outputs of
algorithms 2-by-2 using a Student paired t-test. **NOTE: our code for this functionality 
was not open-source, so please replace it by your code if you want to use this 
functionality (See Algorithm.java for what to compare). This does not prevent
all algorithms to operate.

Training is done using a X-folds stratified CV (X = tunable, 10 by default).

Tempered boosting includes the option to train clamped models (A).

The program can also save the models learned and a number of useful statistics.

Poincar\'e embedding is run on the last sequence of model training using (B) -- if
several runs are made, the last one is displayed only.

MDTs are computed for trees learned with @LogLoss, the last one(s) are plotted
(all trees are plottable, not just in the model but also among splits in CV)

====================================================================================
Technical bits: 

* the losses optimized for DT induction and combination are dual of each other 
(ex: log-/logistic-)

====================================================================================
Java bits: 

* most variables that could be interesting to change (e.g. to speed-up processing,
polish graphical output, etc.) are in Misc.java (interface Debuggable)

====================================================================================
HowTo:

compile e.g. via ./compile.sh

run via e.g.: java -Xmx10000m Experiments -R resource_ionosphere.txt 

(example resource file on UCI ionosphere in /Datasets)

resource_ionosphere.txt contains all parameters to be used (see example file)
the file must contain lines like:

@ALGORITHM,@TemperedLoss,10,5,0.0,NOT_CLAMPED
@ALGORITHM,@LogLoss,20,20

@ALGORITHM = general tag, keep it
{@TemperedLoss, @LogLoss} = loss
X,Y = #trees, max size of trees
If @TemperedLoss, the last two parameters specify t and [clamped] (if t = -1.0, uses
a simple adaptive t scheme, not in the paper)

At display time, a list of key appears in the shell to manipulate the display;
the details of the MDT displayed in Poincar\'e disk are displayed in the shell

====================================================================================
contact: richard.m.nock@gmail.com
