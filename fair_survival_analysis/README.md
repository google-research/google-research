#Fair Survival Analysis
Code for evaluation and learning of fair survival analysis models. 

#Usage

To run baselines:
```
import fair_survival_analysis
fair_survival_analysis.baseline_experiment(data=<DATASET>, quantiles=<QUANTILES>, prot_att=<PROTECTED_ATTRIBUTE>, groups=<GROUPS>, model=<MODEL>, cv_folds=<CV_FOLDS>)
```

To run Coupled Cox Models
```
fair_survival_analysis.experiment(data=<DATASET>,quantiles=<QUANTILES>, prot_att=<PROTECTED_ATTRIBUTE>, groups=<GROUPS>,model=<MODEL>, cv_folds=<CV_FOLDS>)
```

