"""Functions for data manipulation and fitting."""
import numpy as np
from scipy.optimize import curve_fit

from nagin.tools.stats import lorentzian


def fit_data(x_data, y_data, func, p0=None):
    """fit data to a given function and manage exception.

    Parameters
    ----------
    x_data: Array.
        Independent variable data of the function to fit.

    y_data: Array.
        Dependent variable data of the function to fit.

    fit_func: Callable.
        Function to use in the fit in the data.

    p0: Array.
        Initial values of the parameters of the function to be fitted.

    """
    # Try the fit.
    try:

        popt, *_ = curve_fit(func, x_data, y_data, p0=p0)

        # If any value is nan, the fit was not successful.
        if (np.isnan(popt).any()):
            raise RuntimeError()

    except RuntimeError:
        return None

    return popt


def center_derivative(vals):
    """Perform central value derivative.

    Parameters
    ----------
    vals: Array.
        Values of the signal.

    """
    vals = vals.copy()
    vals_left = vals[:-2]
    vals_right = vals[2:]
    return vals_right - vals_left


def second_derivative(vals):
    """Perform second derivative.

    Parameters
    ----------
    vals: Array.
        Values of the signal.

    """
    vals = vals.copy()
    vals_left = vals[:-2]
    vals_central = vals[1:-1]
    vals_right = vals[2:]
    return vals_right - 2*vals_central + vals_left


def derivative_peak_clean(vals, threshold=2.2e-15, repeat=1, second_der=False):
    """Remove peaks of the data using a reference function to subtract.

    Parameters
    ----------
    vals: Array.
        Values of the signal.

    threshold: Float.
        Value of the cut for the peaks in the derivative.

    """
    mask = np.ones(len(vals), dtype=bool)
    for _ in range(repeat):
        d_vals = center_derivative(vals.copy())

        if len(d_vals < threshold):
            d_vals[d_vals < threshold] = 0
            mask[1:-1][d_vals >= threshold] = False

        if second_der:
            dd_vals = second_derivative(vals.copy())
            if len(dd_vals < threshold):
                dd_vals[dd_vals < threshold] = 0
                mask[1:-1][dd_vals >= threshold] = False

        vals[1:-1] -= d_vals

        if second_der:
            vals[1:-1] -= dd_vals

    return mask


def PSD_fit(fr, PSD_data, PSD_0=None, **config):
    """Fit PSD data to a given distribution, normally a Lorentzian.

    Parameters
    ----------
    fr: Array.
        Frequency data array.

    PSD_0: Array.
        Shot noise data.

    PSD_0: Array | Float. Optional.
        Background or 0 V_bias data to subtract to the shot noise data.

    Other Parameters
    ----------------
    Additional configuration parameters passed to `derivative_peak_clean` and
    `fit_data`.

    """
    if PSD_0 is None:
        PSD_0 = 0

    PSD_data -= PSD_0

    mask_clean, d_PSD = derivative_peak_clean(
        PSD_data, threshold=config.pop("threshold", 2.2e-15),
        repeat=config.pop("repeat", 1),
        second_der=config.pop("second_der", False)
    )

    fr_clean = fr[mask_clean]
    PSD_clean = PSD_data[mask_clean]

    popt = fit_data(
        fr_clean, PSD_clean, config.pop("func", lorentzian),
        config.pop("p0", None)
    )

    if config.pop("full_output", False):
        return fr_clean, PSD_clean, d_PSD, popt

    return mask_clean, popt
