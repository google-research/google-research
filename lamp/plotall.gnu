# set terminal pdf dashed
set terminal pdf

set style line 1 lt 1 lw 2 lc rgbcolor "red"
set style line 2 lt 1 lw 2 lc rgbcolor "green"
set style line 3 lt 1 lw 2 lc rgbcolor "blue"
set style line 4 lt 1 lw 2 lc rgbcolor "brown"
set style line 5 lt 1 lw 2 lc rgbcolor "orange"

set style line 6 lt 3 lw 2 lc rgbcolor "red"
set style line 7 lt 3 lw 2 lc rgbcolor "green"
set style line 8 lt 3 lw 2 lc rgbcolor "blue"
set style line 9 lt 3 lw 2 lc rgbcolor "brown"
set style line 10 lt 3 lw 2 lc rgbcolor "orange"

set output '../figs/TAG-train-Order-vs-Perplexity.pdf'
set xlabel "Order"
set ylabel "Train Perplexity"
set key top right
plot 'TAG-ngram-train.tsv' using "Order":"Perplexity" title 'Naive N-Gram' with linespoints ls 6,\
     'TAG-kn-train.tsv' using "Order":"Perplexity" title 'Kneyser-Ney N-Gram' with linespoints ls 7,\
     'TAG-lamp-train.tsv' using "Order":"Perplexity" title 'LAMP' with linespoints ls 8,\
     'TAG-lamp-weights-only-train.tsv' using "Order":"Perplexity" title 'Weight Only LAMP' with linespoints ls 9,\
     'TAG-initial-train.tsv' using "Order":"Perplexity" title 'Initial Weights' with linespoints ls 10

set output '../figs/TAG-test-Order-vs-Perplexity.pdf'
set ylabel "Test Perplexity"
set key top left
set yrange [*:*<MAXNGRAMPERPLEX]
plot 'TAG-ngram-test.tsv' using "Order":"Perplexity" notitle with lines ls 1,\
     'TAG-ngram-test.tsv' using "Order":"Perplexity":"PerplexityStDev" title 'Naive N-Gram' with errorbars ls 1,\
     'TAG-kn-test.tsv' using "Order":"Perplexity" notitle with lines ls 2,\
     'TAG-kn-test.tsv' using "Order":"Perplexity":"PerplexityStDev" title 'Kneyser-Ney N-Gram' with errorbars ls 2,\
     'TAG-lamp-test.tsv' using "Order":"Perplexity" notitle with lines ls 3,\
     'TAG-lamp-test.tsv' using "Order":"Perplexity":"PerplexityStDev" title 'LAMP' with errorbars ls 3,\
     'TAG-lamp-weights-only-test.tsv' using "Order":"Perplexity" notitle with lines ls 4,\
     'TAG-lamp-weights-only-test.tsv' using "Order":"Perplexity":"PerplexityStDev" title 'Weight Only LAMP' with errorbars ls 4,\
     'TAG-initial-test.tsv' using "Order":"Perplexity" notitle with lines ls 5,\
     'TAG-initial-test.tsv' using "Order":"Perplexity":"PerplexityStDev" title 'Initial Weights' with errorbars ls 5
set key default
set yrange [*:*]

set output '../figs/TAG-train-Order-vs-Accuracy.pdf'
set xlabel "Order"
set ylabel "Train Accuracy"
set key top right
plot 'TAG-lamp-train.tsv' using "Order":"Accuracy" title 'LAMP' with linespoints ls 8,\
     'TAG-lamp-weights-only-train.tsv' using "Order":"Accuracy" title 'Weight Only LAMP' with linespoints ls 9,\
     'TAG-initial-train.tsv' using "Order":"Accuracy" title 'Initial Weights' with linespoints ls 10

set output '../figs/TAG-test-Order-vs-Accuracy.pdf'
set ylabel "Test Accuracy"
plot 'TAG-lamp-test.tsv' using "Order":"Accuracy" notitle with lines ls 3,\
     'TAG-lamp-test.tsv' using "Order":"Accuracy":"AccuracyStDev" title 'LAMP' with errorbars ls 3,\
     'TAG-lamp-weights-only-test.tsv' using "Order":"Accuracy" notitle with lines ls 4,\
     'TAG-lamp-weights-only-test.tsv' using "Order":"Accuracy":"AccuracyStDev" title 'Weight Only LAMP' with errorbars ls 4,\
     'TAG-initial-test.tsv' using "Order":"Accuracy" notitle with lines ls 5,\
     'TAG-initial-test.tsv' using "Order":"Accuracy":"AccuracyStDev" title 'Initial Weights' with errorbars ls 5
set key default

set output '../figs/TAG-Order-vs-NumParams.pdf'
set xlabel "Order"
set ylabel "Number of Parameters"
set key top left
set logscale y
plot 'TAG-ngram-train.tsv' using "Order":"NumParams" title 'Naive N-Gram' with linespoints ls 1,\
     'TAG-kn-train.tsv' using "Order":"NumParams" title 'Kneyser-Ney N-Gram' with linespoints ls 2,\
     'TAG-lamp-train.tsv' using "Order":"NumParams" title 'LAMP' with linespoints ls 3,\
     'TAG-lamp-weights-only-train.tsv' using "Order":"NumParams" title 'Weight Only LAMP' with linespoints ls 4
set key default
unset logscale y

set output '../figs/TAG-train-NumParams-vs-Perplexity.pdf'
set xlabel "Number of Parameters"
set ylabel "Train Perplexity"
set logscale x
plot 'TAG-ngram-train.tsv' using "NumParams":"Perplexity" title 'Naive N-Gram' with linespoints ls 6,\
     'TAG-kn-train.tsv' using "NumParams":"Perplexity" title 'Kneyser-Ney N-Gram' with linespoints ls 7,\
     'TAG-lamp-train.tsv' using "NumParams":"Perplexity" title 'LAMP' with linespoints ls 8,\
     'TAG-lamp-weights-only-train.tsv' using "NumParams":"Perplexity" title 'Weight only LAMP' with linespoints ls 9

set output '../figs/TAG-test-NumParams-vs-Perplexity.pdf'
set ylabel "Test Perplexity"
set yrange [*:*<MAXNGRAMPERPLEX]
plot 'TAG-ngram-test.tsv' using "NumParams":"Perplexity" title 'Naive N-Gram' with linespoints ls 1,\
     'TAG-kn-test.tsv' using "NumParams":"Perplexity" title 'Kneyser-Ney N-Gram' with linespoints ls 2,\
     'TAG-lamp-test.tsv' using "NumParams":"Perplexity" title 'LAMP' with linespoints ls 3,\
     'TAG-lamp-weights-only-test.tsv' using "NumParams":"Perplexity" title 'Weight Only LAMP' with linespoints ls 4
unset logscale x
set yrange [*:*]

set output '../figs/TAG-train-Iterations-vs-Perplexity.pdf'
set xlabel "Number of Iterations"
set ylabel "Train Perplexity"
set key top right
plot 'TAG-lamp-order-2-iter-train.tsv' using "Iter":"Perplexity" title 'LAMP Order 2' with linespoints ls 6,\
     'TAG-lamp-order-3-iter-train.tsv' using "Iter":"Perplexity" title 'LAMP Order 3' with linespoints ls 7,\
     'TAG-lamp-order-4-iter-train.tsv' using "Iter":"Perplexity" title 'LAMP Order 4' with linespoints ls 8,\
     'TAG-lamp-order-5-iter-train.tsv' using "Iter":"Perplexity" title 'LAMP Order 5' with linespoints ls 9,\
     'TAG-lamp-order-6-iter-train.tsv' using "Iter":"Perplexity" title 'LAMP Order 6' with linespoints ls 10

set output '../figs/TAG-test-Iterations-vs-Perplexity.pdf'
set ylabel "Test Perplexity"
plot 'TAG-lamp-order-2-iter-test.tsv' using "Iter":"Perplexity" notitle with lines ls 1,\
     'TAG-lamp-order-2-iter-test.tsv' using "Iter":"Perplexity":"PerplexityStDev" title 'LAMP Order 2' with errorbars ls 1,\
     'TAG-lamp-order-3-iter-test.tsv' using "Iter":"Perplexity" notitle with lines ls 2,\
     'TAG-lamp-order-3-iter-test.tsv' using "Iter":"Perplexity":"PerplexityStDev" title 'LAMP Order 3' with errorbars ls 2,\
     'TAG-lamp-order-4-iter-test.tsv' using "Iter":"Perplexity" notitle with lines ls 3,\
     'TAG-lamp-order-4-iter-test.tsv' using "Iter":"Perplexity":"PerplexityStDev" title 'LAMP Order 4' with errorbars ls 3,\
     'TAG-lamp-order-5-iter-test.tsv' using "Iter":"Perplexity" notitle with lines ls 4,\
     'TAG-lamp-order-5-iter-test.tsv' using "Iter":"Perplexity":"PerplexityStDev" title 'LAMP Order 5' with errorbars ls 4,\
     'TAG-lamp-order-6-iter-test.tsv' using "Iter":"Perplexity" notitle with lines ls 5,\
     'TAG-lamp-order-6-iter-test.tsv' using "Iter":"Perplexity":"PerplexityStDev" title 'LAMP Order 6' with errorbars ls 5
set key default

set output '../figs/TAG-train-Iterations-vs-Accuracy.pdf'
set xlabel "Number of Iterations"
set ylabel "Train Accuracy"
set key top right
plot 'TAG-lamp-order-2-iter-train.tsv' using "Iter":"Accuracy" title 'LAMP Order 2' with linespoints ls 6,\
     'TAG-lamp-order-3-iter-train.tsv' using "Iter":"Accuracy" title 'LAMP Order 3' with linespoints ls 7,\
     'TAG-lamp-order-4-iter-train.tsv' using "Iter":"Accuracy" title 'LAMP Order 4' with linespoints ls 8,\
     'TAG-lamp-order-5-iter-train.tsv' using "Iter":"Accuracy" title 'LAMP Order 5' with linespoints ls 9,\
     'TAG-lamp-order-5-iter-train.tsv' using "Iter":"Accuracy" title 'LAMP Order 6' with linespoints ls 10

set output '../figs/TAG-test-Iterations-vs-Accuracy.pdf'
set ylabel "Test Accuracy"
plot 'TAG-lamp-order-2-iter-test.tsv' using "Iter":"Accuracy" notitle with lines ls 1,\
     'TAG-lamp-order-2-iter-test.tsv' using "Iter":"Accuracy":"AccuracyStDev" title 'LAMP Order 2' with errorbars ls 1,\
     'TAG-lamp-order-3-iter-test.tsv' using "Iter":"Accuracy" notitle with lines ls 2,\
     'TAG-lamp-order-3-iter-test.tsv' using "Iter":"Accuracy":"AccuracyStDev" title 'LAMP Order 3' with errorbars ls 2,\
     'TAG-lamp-order-4-iter-test.tsv' using "Iter":"Accuracy" notitle with lines ls 3,\
     'TAG-lamp-order-4-iter-test.tsv' using "Iter":"Accuracy":"AccuracyStDev" title 'LAMP Order 4' with errorbars ls 3,\
     'TAG-lamp-order-5-iter-test.tsv' using "Iter":"Accuracy" notitle with lines ls 4,\
     'TAG-lamp-order-5-iter-test.tsv' using "Iter":"Accuracy":"AccuracyStDev" title 'LAMP Order 5' with errorbars ls 4,\
     'TAG-lamp-order-6-iter-test.tsv' using "Iter":"Accuracy" notitle with lines ls 5,\
     'TAG-lamp-order-6-iter-test.tsv' using "Iter":"Accuracy":"AccuracyStDev" title 'LAMP Order 6' with errorbars ls 5
set key default

set output '../figs/TAG-lamp-Weights.pdf'
set ylabel "Weight value"
set xlabel "Index"
plot 'TAG-lamp-order-2-weights.tsv' using (1+$0):"Weight" notitle with lines ls 1,\
     'TAG-lamp-order-2-weights.tsv' using (1+$0):"Weight":"WeightStDev" title 'LAMP Order 2' with errorbars ls 1,\
     'TAG-lamp-order-3-weights.tsv' using (1+$0):"Weight" notitle with lines ls 2,\
     'TAG-lamp-order-3-weights.tsv' using (1+$0):"Weight":"WeightStDev" title 'LAMP Order 3' with errorbars ls 2,\
     'TAG-lamp-order-4-weights.tsv' using (1+$0):"Weight" notitle with lines ls 3,\
     'TAG-lamp-order-4-weights.tsv' using (1+$0):"Weight":"WeightStDev" title 'LAMP Order 4' with errorbars ls 3,\
     'TAG-lamp-order-5-weights.tsv' using (1+$0):"Weight" notitle with lines ls 4,\
     'TAG-lamp-order-5-weights.tsv' using (1+$0):"Weight":"WeightStDev" title 'LAMP Order 5' with errorbars ls 4,\
     'TAG-lamp-order-6-weights.tsv' using (1+$0):"Weight" notitle with lines ls 5,\
     'TAG-lamp-order-6-weights.tsv' using (1+$0):"Weight":"WeightStDev" title 'LAMP Order 6' with errorbars ls 5

set output '../figs/TAG-lamp-weights-only-Weights.pdf'
plot 'TAG-lamp-weights-only-order-2-weights.tsv' using (1+$0):"Weight" notitle with lines ls 1,\
     'TAG-lamp-weights-only-order-2-weights.tsv' using (1+$0):"Weight":"WeightStDev" title 'Weight Only LAMP Order 2' with errorbars ls 1,\
     'TAG-lamp-weights-only-order-3-weights.tsv' using (1+$0):"Weight" notitle with lines ls 2,\
     'TAG-lamp-weights-only-order-3-weights.tsv' using (1+$0):"Weight":"WeightStDev" title 'Weight Only LAMP Order 3' with errorbars ls 2,\
     'TAG-lamp-weights-only-order-4-weights.tsv' using (1+$0):"Weight" notitle with lines ls 3,\
     'TAG-lamp-weights-only-order-4-weights.tsv' using (1+$0):"Weight":"WeightStDev" title 'Weight Only LAMP Order 4' with errorbars ls 3,\
     'TAG-lamp-weights-only-order-5-weights.tsv' using (1+$0):"Weight" notitle with lines ls 4,\
     'TAG-lamp-weights-only-order-5-weights.tsv' using (1+$0):"Weight":"WeightStDev" title 'Weight Only LAMP Order 5' with errorbars ls 4,\
     'TAG-lamp-weights-only-order-6-weights.tsv' using (1+$0):"Weight" notitle with lines ls 5,\
     'TAG-lamp-weights-only-order-6-weights.tsv' using (1+$0):"Weight":"WeightStDev" title 'Weight Only LAMP Order 6' with errorbars ls 5
set key default
