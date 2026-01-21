# This program is free software: you can redistribute it and/or modify
# it under the terms of the GNU General Public License as published by
# the Free Software Foundation, either version 3 of the License, or
# (at your option) any later version.
#
# This program is distributed in the hope that it will be useful,
# but WITHOUT ANY WARRANTY; without even the implied warranty of
# MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
# GNU General Public License for more details.
#
# You should have received a copy of the GNU General Public License
# along with this program.  If not, see <http://www.gnu.org/licenses/>.

# This module was adapted from Scikit-Survival python package to
# extract the adjusted TPR and FPR rates for all classification
# thresholds for the censoring adjusted ROC curve. It depends on
# Scikit-Survival. For the full package please check: 
# https://github.com/sebp/scikit-survival

import numpy
from scipy.integrate import trapz
from sklearn.utils import check_consistent_length, check_array

from sksurv.nonparametric import CensoringDistributionEstimator, SurvivalFunctionEstimator
from sksurv.util import check_y_survival

__all__ = [
    'brier_score',
    'concordance_index_censored',
    'concordance_index_ipcw',
    'cumulative_dynamic_auc',
    'integrated_brier_score',
]


def _check_estimate(estimate, test_time):
    estimate = check_array(estimate, ensure_2d=False)
    if estimate.ndim != 1:
        raise ValueError(
            'Expected 1D array, got {:d}D array instead:\narray={}.\n'.format(
                estimate.ndim, estimate))
    check_consistent_length(test_time, estimate)
    return estimate


def _check_inputs(event_indicator, event_time, estimate):
    check_consistent_length(event_indicator, event_time, estimate)
    event_indicator = check_array(event_indicator, ensure_2d=False)
    event_time = check_array(event_time, ensure_2d=False)
    estimate = _check_estimate(estimate, event_time)

    if not numpy.issubdtype(event_indicator.dtype, numpy.bool_):
        raise ValueError(
            'only boolean arrays are supported as class labels for survival analysis, got {0}'.format(
                event_indicator.dtype))

    if len(event_time) < 2:
        raise ValueError("Need a minimum of two samples")

    if not event_indicator.any():
        raise ValueError("All samples are censored")

    return event_indicator, event_time, estimate


def _check_times(test_time, times):
    times = check_array(numpy.atleast_1d(times), ensure_2d=False, dtype=test_time.dtype)
    times = numpy.unique(times)

    if times.max() >= test_time.max() or times.min() < test_time.min():
        raise ValueError(
            'all times must be within follow-up time of test data: [{}; {}['.format(
                test_time.min(), test_time.max()))

    return times


def _get_comparable(event_indicator, event_time, order):
    n_samples = len(event_time)
    tied_time = 0
    comparable = {}
    i = 0
    while i < n_samples - 1:
        time_i = event_time[order[i]]
        start = i + 1
        end = start
        while end < n_samples and event_time[order[end]] == time_i:
            end += 1

        # check for tied event times
        event_at_same_time = event_indicator[order[i:end]]
        censored_at_same_time = ~event_at_same_time
        for j in range(i, end):
            if event_indicator[order[j]]:
                mask = numpy.zeros(n_samples, dtype=bool)
                mask[end:] = True
                # an event is comparable to censored samples at same time point
                mask[i:end] = censored_at_same_time
                comparable[j] = mask
                tied_time += censored_at_same_time.sum()
        i = end

    return comparable, tied_time


def _estimate_concordance_index(event_indicator, event_time, estimate, weights, tied_tol=1e-8):
    order = numpy.argsort(event_time)

    comparable, tied_time = _get_comparable(event_indicator, event_time, order)

    concordant = 0
    discordant = 0
    tied_risk = 0
    numerator = 0.0
    denominator = 0.0
    for ind, mask in comparable.items():
        est_i = estimate[order[ind]]
        event_i = event_indicator[order[ind]]
        w_i = weights[order[ind]]

        est = estimate[order[mask]]

        assert event_i, 'got censored sample at index %d, but expected uncensored' % order[ind]

        ties = numpy.absolute(est - est_i) <= tied_tol
        n_ties = ties.sum()
        # an event should have a higher score
        con = est < est_i
        n_con = con[~ties].sum()

        numerator += w_i * n_con + 0.5 * w_i * n_ties
        denominator += w_i * mask.sum()

        tied_risk += n_ties
        concordant += n_con
        discordant += est.size - n_con - n_ties

    cindex = numerator / denominator
    return cindex, concordant, discordant, tied_risk, tied_time


def concordance_index_censored(event_indicator, event_time, estimate, tied_tol=1e-8):
    """Concordance index for right-censored data

    The concordance index is defined as the proportion of all comparable pairs
    in which the predictions and outcomes are concordant.

    Two samples are comparable if (i) both of them experienced an event (at different times),
    or (ii) the one with a shorter observed survival time experienced an event, in which case
    the event-free subject "outlived" the other. A pair is not comparable if they experienced
    events at the same time.

    Concordance intuitively means that two samples were ordered correctly by the model.
    More specifically, two samples are concordant, if the one with a higher estimated
    risk score has a shorter actual survival time.
    When predicted risks are identical for a pair, 0.5 rather than 1 is added to the count
    of concordant pairs.

    See [1]_ for further description.

    Parameters
    ----------
    event_indicator : array-like, shape = (n_samples,)
        Boolean array denotes whether an event occurred

    event_time : array-like, shape = (n_samples,)
        Array containing the time of an event or time of censoring

    estimate : array-like, shape = (n_samples,)
        Estimated risk of experiencing an event

    tied_tol : float, optional, default: 1e-8
        The tolerance value for considering ties.
        If the absolute difference between risk scores is smaller
        or equal than `tied_tol`, risk scores are considered tied.

    Returns
    -------
    cindex : float
        Concordance index

    concordant : int
        Number of concordant pairs

    discordant : int
        Number of discordant pairs

    tied_risk : int
        Number of pairs having tied estimated risks

    tied_time : int
        Number of comparable pairs sharing the same time

    References
    ----------
    .. [1] Harrell, F.E., Califf, R.M., Pryor, D.B., Lee, K.L., Rosati, R.A,
           "Multivariable prognostic models: issues in developing models,
           evaluating assumptions and adequacy, and measuring and reducing errors",
           Statistics in Medicine, 15(4), 361-87, 1996.
    """
    event_indicator, event_time, estimate = _check_inputs(
        event_indicator, event_time, estimate)

    w = numpy.ones_like(estimate)

    return _estimate_concordance_index(event_indicator, event_time, estimate, w, tied_tol)


def concordance_index_ipcw(survival_train, survival_test, estimate, tau=None, tied_tol=1e-8):
    """Concordance index for right-censored data based on inverse probability of censoring weights.

    This is an alternative to the estimator in :func:`concordance_index_censored`
    that does not depend on the distribution of censoring times in the test data.
    Therefore, the estimate is unbiased and consistent for a population concordance
    measure that is free of censoring.

    It is based on inverse probability of censoring weights, thus requires
    access to survival times from the training data to estimate the censoring
    distribution. Note that this requires that survival times `survival_test`
    lie within the range of survival times `survival_train`. This can be
    achieved by specifying the truncation time `tau`.
    The resulting `cindex` tells how well the given prediction model works in
    predicting events that occur in the time range from 0 to `tau`.

    The estimator uses the Kaplan-Meier estimator to estimate the
    censoring survivor function. Therefore, it is restricted to
    situations where the random censoring assumption holds and
    censoring is independent of the features.

    See [1]_ for further description.

    Parameters
    ----------
    survival_train : structured array, shape = (n_train_samples,)
        Survival times for training data to estimate the censoring
        distribution from.
        A structured array containing the binary event indicator
        as first field, and time of event or time of censoring as
        second field.

    survival_test : structured array, shape = (n_samples,)
        Survival times of test data.
        A structured array containing the binary event indicator
        as first field, and time of event or time of censoring as
        second field.

    estimate : array-like, shape = (n_samples,)
        Estimated risk of experiencing an event of test data.

    tau : float, optional
        Truncation time. The survival function for the underlying
        censoring time distribution :math:`D` needs to be positive
        at `tau`, i.e., `tau` should be chosen such that the
        probability of being censored after time `tau` is non-zero:
        :math:`P(D > \\tau) > 0`. If `None`, no truncation is performed.

    tied_tol : float, optional, default: 1e-8
        The tolerance value for considering ties.
        If the absolute difference between risk scores is smaller
        or equal than `tied_tol`, risk scores are considered tied.

    Returns
    -------
    cindex : float
        Concordance index

    concordant : int
        Number of concordant pairs

    discordant : int
        Number of discordant pairs

    tied_risk : int
        Number of pairs having tied estimated risks

    tied_time : int
        Number of comparable pairs sharing the same time

    References
    ----------
    .. [1] Uno, H., Cai, T., Pencina, M. J., D’Agostino, R. B., & Wei, L. J. (2011).
           "On the C-statistics for evaluating overall adequacy of risk prediction
           procedures with censored survival data".
           Statistics in Medicine, 30(10), 1105–1117.
    """
    test_event, test_time = check_y_survival(survival_test)

    if tau is not None:
        mask = test_time < tau
        survival_test = survival_test[mask]

    estimate = _check_estimate(estimate, test_time)

    cens = CensoringDistributionEstimator()
    cens.fit(survival_train)
    ipcw_test = cens.predict_ipcw(survival_test)
    if tau is None:
        ipcw = ipcw_test
    else:
        ipcw = numpy.empty(estimate.shape[0], dtype=ipcw_test.dtype)
        ipcw[mask] = ipcw_test
        ipcw[~mask] = 0

    w = numpy.square(ipcw)

    return _estimate_concordance_index(test_event, test_time, estimate, w, tied_tol)


def cumulative_dynamic_auc(survival_train, survival_test, estimate, times, tied_tol=1e-8):
    """Estimator of cumulative/dynamic AUC for right-censored time-to-event data.

    The receiver operating characteristic (ROC) curve and the area under the
    ROC curve (AUC) can be extended to survival data by defining
    sensitivity (true positive rate) and specificity (true negative rate)
    as time-dependent measures. *Cumulative cases* are all individuals that
    experienced an event prior to or at time :math:`t` (:math:`t_i \\leq t`),
    whereas *dynamic controls* are those with :math:`t_i > t`.
    The associated cumulative/dynamic AUC quantifies how well a model can
    distinguish subjects who fail by a given time (:math:`t_i \\leq t`) from
    subjects who fail after this time (:math:`t_i > t`).

    Given an estimator of the :math:`i`-th individual's risk score
    :math:`\\hat{f}(\\mathbf{x}_i)`, the cumulative/dynamic AUC at time
    :math:`t` is defined as

    .. math::

        \\widehat{\\mathrm{AUC}}(t) =
        \\frac{\\sum_{i=1}^n \\sum_{j=1}^n I(y_j > t) I(y_i \\leq t) \\omega_i
        I(\\hat{f}(\\mathbf{x}_j) \\leq \\hat{f}(\\mathbf{x}_i))}
        {(\\sum_{i=1}^n I(y_i > t)) (\\sum_{i=1}^n I(y_i \\leq t) \\omega_i)}

    where :math:`\\omega_i` are inverse probability of censoring weights (IPCW).

    To estimate IPCW, access to survival times from the training data is required
    to estimate the censoring distribution. Note that this requires that survival
    times `survival_test` lie within the range of survival times `survival_train`.
    This can be achieved by specifying `times` accordingly, e.g. by setting
    `times[-1]` slightly below the maximum expected follow-up time.
    IPCW are computed using the Kaplan-Meier estimator, which is
    restricted to situations where the random censoring assumption holds and
    censoring is independent of the features.

    The function also provides a single summary measure that refers to the mean
    of the :math:`\\mathrm{AUC}(t)` over the time range :math:`(\\tau_1, \\tau_2)`.

    .. math::

        \\overline{\\mathrm{AUC}}(\\tau_1, \\tau_2) =
        \\frac{1}{\\hat{S}(\\tau_1) - \\hat{S}(\\tau_2)}
        \\int_{\\tau_1}^{\\tau_2} \\widehat{\\mathrm{AUC}}(t)\\,d \\hat{S}(t)

    where :math:`\\hat{S}(t)` is the Kaplan–Meier estimator of the survival function.

    See [1]_, [2]_, [3]_ for further description.

    Parameters
    ----------
    survival_train : structured array, shape = (n_train_samples,)
        Survival times for training data to estimate the censoring
        distribution from.
        A structured array containing the binary event indicator
        as first field, and time of event or time of censoring as
        second field.

    survival_test : structured array, shape = (n_samples,)
        Survival times of test data.
        A structured array containing the binary event indicator
        as first field, and time of event or time of censoring as
        second field.

    estimate : array-like, shape = (n_samples,)
        Estimated risk of experiencing an event of test data.

    times : array-like, shape = (n_times,)
        The time points for which the area under the
        time-dependent ROC curve is computed. Values must be
        within the range of follow-up times of the test data
        `survival_test`.

    tied_tol : float, optional, default: 1e-8
        The tolerance value for considering ties.
        If the absolute difference between risk scores is smaller
        or equal than `tied_tol`, risk scores are considered tied.

    Returns
    -------
    auc : array, shape = (n_times,)
        The cumulative/dynamic AUC estimates (evaluated at `times`).
    mean_auc : float
        Summary measure referring to the mean cumulative/dynamic AUC
        over the specified time range `(times[0], times[-1])`.

    References
    ----------
    .. [1] H. Uno, T. Cai, L. Tian, and L. J. Wei,
           "Evaluating prediction rules for t-year survivors with censored regression models,"
           Journal of the American Statistical Association, vol. 102, pp. 527–537, 2007.
    .. [2] H. Hung and C. T. Chiang,
           "Estimation methods for time-dependent AUC models with survival data,"
           Canadian Journal of Statistics, vol. 38, no. 1, pp. 8–26, 2010.
    .. [3] J. Lambert and S. Chevret,
           "Summary measure of discrimination in survival models based on cumulative/dynamic time-dependent ROC curves,"
           Statistical Methods in Medical Research, 2014.
    """
        
    test_event, test_time = check_y_survival(survival_test)
    
    estimate = _check_estimate(estimate, test_time)

    times = _check_times(test_time, times)

    # sort by risk score (descending)
    o = numpy.argsort(-estimate)
    test_time = test_time[o]
    test_event = test_event[o]
    estimate = estimate[o]
    survival_test = survival_test[o]

    cens = CensoringDistributionEstimator()
    cens.fit(survival_train)
    ipcw = cens.predict_ipcw(survival_test)

    n_samples = test_time.shape[0]
    scores = numpy.empty(times.shape[0], dtype=float)
    rocs = []
    
    
    for k, t in enumerate(times):
        is_case = (test_time <= t) & test_event
        is_control = test_time > t
        n_controls = is_control.sum()

        true_pos = []
        false_pos = []
        tp_value = 0.0
        fp_value = 0.0
        est_prev = numpy.infty

        for i in range(n_samples):
            est = estimate[i]
            if numpy.absolute(est - est_prev) > tied_tol:
                true_pos.append(tp_value)
                false_pos.append(fp_value)
                est_prev = est
            if is_case[i]:
                tp_value += ipcw[i]
            elif is_control[i]:
                fp_value += 1
        true_pos.append(tp_value)
        false_pos.append(fp_value)

        sens = numpy.array(true_pos) / ipcw[is_case].sum()
        fpr = numpy.array(false_pos) / n_controls
        scores[k] = trapz(sens, fpr)
        
        rocs.append((sens, fpr))
        
    if times.shape[0] == 1:
        mean_auc = scores[0]
    else:
        surv = SurvivalFunctionEstimator()
        surv.fit(survival_test)
        s_times = surv.predict_proba(times)
        # compute integral of AUC over survival function
        d = -numpy.diff(numpy.concatenate(([1.0], s_times)))
        integral = (scores * d).sum()
        mean_auc = integral / (1.0 - s_times[-1])

    return rocs, scores, mean_auc


def brier_score(survival_train, survival_test, estimate, times):
    """Estimate the time-dependent Brier score for right censored data.

    The time-dependent Brier score is the mean squared error at time point :math:`t`:

    .. math::

        \\mathrm{BS}^c(t) = \\frac{1}{n} \\sum_{i=1}^n I(y_i \\leq t \\land \\delta_i = 1)
        \\frac{(0 - \\hat{\\pi}(t | \\mathbf{x}_i))^2}{\\hat{G}(y_i)} + I(y_i > t)
        \\frac{(1 - \\hat{\\pi}(t | \\mathbf{x}_i))^2}{\\hat{G}(t)} ,

    where :math:`\\hat{\\pi}(t | \\mathbf{x})` is the predicted probability of
    remaining event-free up to time point :math:`t` for a feature vector :math:`\\mathbf{x}`,
    and :math:`1/\\hat{G}(t)` is a inverse probability of censoring weight, estimated by
    the Kaplan-Meier estimator.

    See [1]_ for details.

    Parameters
    ----------
    survival_train : structured array, shape = (n_train_samples,)
        Survival times for training data to estimate the censoring
        distribution from.
        A structured array containing the binary event indicator
        as first field, and time of event or time of censoring as
        second field.

    survival_test : structured array, shape = (n_samples,)
        Survival times of test data.
        A structured array containing the binary event indicator
        as first field, and time of event or time of censoring as
        second field.

    estimate : array-like, shape = (n_samples, n_times)
        Estimated risk of experiencing an event for test data at `times`.
        The i-th column must contain the estimated probability of
        remaining event-free up to the i-th time point.

    times : array-like, shape = (n_times,)
        The time points for which to estimate the Brier score.
        Values must be within the range of follow-up times of
        the test data `survival_test`.

    Returns
    -------
    times : array, shape = (n_times,)
        Unique time points at which the brier scores was estimated.

    brier_scores : array , shape = (n_times,)
        Values of the brier score.

    Examples
    --------
    >>> from sksurv.datasets import load_gbsg2
    >>> from sksurv.linear_model import CoxPHSurvivalAnalysis
    >>> from sksurv.metrics import brier_score
    >>> from sksurv.preprocessing import OneHotEncoder

    Load and prepare data.

    >>> X, y = load_gbsg2()
    >>> X.loc[:, "tgrade"] = X.loc[:, "tgrade"].map(len).astype(int)
    >>> Xt = OneHotEncoder().fit_transform(X)

    Fit a Cox model.

    >>> est = CoxPHSurvivalAnalysis(ties="efron").fit(Xt, y)

    Retrieve individual survival functions and get probability
    of remaining event free up to 5 years (=1825 days).

    >>> survs = est.predict_survival_function(Xt)
    >>> preds = [fn(1825) for fn in survs]

    Compute the Brier score at 5 years.

    >>> times, score = brier_score(y, y, preds, 1825)
    >>> print(score)
    [0.20881843]

    See also
    --------
    integrated_brier_score

    References
    ----------
    .. [1] E. Graf, C. Schmoor, W. Sauerbrei, and M. Schumacher,
           "Assessment and comparison of prognostic classification schemes for survival data,"
           Statistics in Medicine, vol. 18, no. 17-18, pp. 2529–2545, 1999.
    """
    test_event, test_time = check_y_survival(survival_test)
    times = _check_times(test_time, times)

    estimate = check_array(estimate, ensure_2d=False)
    if estimate.ndim == 1 and times.shape[0] == 1:
        estimate = estimate.reshape(-1, 1)

    if estimate.shape[0] != test_time.shape[0]:
        raise ValueError("expected estimate with {} samples, but got {}".format(
            test_time.shape[0], estimate.shape[0]
        ))

    if estimate.shape[1] != times.shape[0]:
        raise ValueError("expected estimate with {} columns, but got {}".format(
            times.shape[0], estimate.shape[1]))

    # fit IPCW estimator
    cens = CensoringDistributionEstimator().fit(survival_train)
    # calculate inverse probability of censoring weight at current time point t.
    prob_cens_t = cens.predict_proba(times)
    prob_cens_t[prob_cens_t == 0] = numpy.inf
    # calculate inverse probability of censoring weights at observed time point
    prob_cens_y = cens.predict_proba(test_time)
    prob_cens_y[prob_cens_y == 0] = numpy.inf

    # Calculating the brier scores at each time point
    brier_scores = numpy.empty(times.shape[0], dtype=float)
    for i, t in enumerate(times):
        est = estimate[:, i]
        is_case = (test_time <= t) & test_event
        is_control = test_time > t

        brier_scores[i] = numpy.mean(numpy.square(est) * is_case.astype(int) / prob_cens_y
                                     + numpy.square(1.0 - est) * is_control.astype(int) / prob_cens_t[i])

    return times, brier_scores


def integrated_brier_score(survival_train, survival_test, estimate, times):
    """The Integrated Brier Score (IBS) provides an overall calculation of
    the model performance at all available times :math:`t_1 \\leq t \\leq t_\\text{max}`.

    The integrated time-dependent Brier score over the interval
    :math:`[t_1; t_\\text{max}]` is defined as

    .. math::

        \\mathrm{IBS} = \\int_{t_1}^{t_\\text{max}} \\mathrm{BS}^c(t) d w(t)

    where the weighting function is :math:`w(t) = t / t_\\text{max}`.
    The integral is estimated via the trapezoidal rule.

    See [1]_ for further details.

    Parameters
    ----------
    survival_train : structured array, shape = (n_train_samples,)
        Survival times for training data to estimate the censoring
        distribution from.
        A structured array containing the binary event indicator
        as first field, and time of event or time of censoring as
        second field.

    survival_test : structured array, shape = (n_samples,)
        Survival times of test data.
        A structured array containing the binary event indicator
        as first field, and time of event or time of censoring as
        second field.

    estimate : array-like, shape = (n_samples, n_times)
        Estimated risk of experiencing an event for test data at `times`.
        The i-th column must contain the estimated probability of
        remaining event-free up to the i-th time point.

    times : array-like, shape = (n_times,)
        The time points for which to estimate the Brier score.
        Values must be within the range of follow-up times of
        the test data `survival_test`.

    Returns
    -------
    ibs : float
        The integrated Brier score.

    Examples
    --------
    >>> import numpy
    >>> from sksurv.datasets import load_gbsg2
    >>> from sksurv.linear_model import CoxPHSurvivalAnalysis
    >>> from sksurv.metrics import integrated_brier_score
    >>> from sksurv.preprocessing import OneHotEncoder

    Load and prepare data.

    >>> X, y = load_gbsg2()
    >>> X.loc[:, "tgrade"] = X.loc[:, "tgrade"].map(len).astype(int)
    >>> Xt = OneHotEncoder().fit_transform(X)

    Fit a Cox model.

    >>> est = CoxPHSurvivalAnalysis(ties="efron").fit(Xt, y)

    Retrieve individual survival functions and get probability
    of remaining event free from 1 year to 5 years (=1825 days).

    >>> survs = est.predict_survival_function(Xt)
    >>> times = numpy.arange(365, 1826)
    >>> preds = numpy.asarray([[fn(t) for t in times for fn in survs]])

    Compute the integrated Brier score from 1 to 5 years.

    >>> score = integrated_brier_score(y, y, preds, times)
    >>> print(score)
    0.1815853064627424

    See also
    --------
    brier_score

    References
    ----------
    .. [1] E. Graf, C. Schmoor, W. Sauerbrei, and M. Schumacher,
           "Assessment and comparison of prognostic classification schemes for survival data,"
           Statistics in Medicine, vol. 18, no. 17-18, pp. 2529–2545, 1999.
    """
    # Computing the brier scores
    times, brier_scores = brier_score(survival_train, survival_test, estimate, times)

    if times.shape[0] < 2:
        raise ValueError("At least two time points must be given")

    # Computing the IBS
    ibs_value = trapz(brier_scores, times) / (times[-1] - times[0])

    return ibs_value
