"""Statistic and distribution functions."""
import numpy as np


def lorentzian(x, x0, gamma, a):
    """Lorentzian function for curves fitting .

    Parameters
    ----------
    x: Float | Array of floats.
        Points in which the function will be evaluated.

    x0: Float.
        Mean value of the distribution.

    gamma: Float.
        With of the distribution.

    a: Float.
        Amplitud of the curve.

    """
    return a * gamma**2 / (gamma**2 + (x-x0)**2)


def bkg_lorentzian(x, x0, gamma, a, *b, bgk_func=lambda x, b, c: x*b + c):
    """Lorentzian + background function for curves fitting.

    Parameters
    ----------
    x: Float | Array of floats.
        Points in which the function will be evaluated.

    x0: Float.
        Mean value of the distribution.

    gamma: Float.
        With of the distribution.

    a: Float.
        Amplitud of the curve.

    b: Float.
        Parameter(s) of the background.

    bkg_func: Callable.
        Background function to be added to the Lorentzian. If nothing is given,
        a linear background if used.

    """
    return lorentzian(x, x0, gamma, a) + bgk_func(x, *b)


def expbkg_lorentzian(x, x0, gamma, a, b, c):
    """Lorentzian + exponential background for curves fitting.

    Parameters
    ----------
    x: Float | Array of floats.
        Points in which the function will be evaluated.

    x0: Float.
        Mean value of the distribution.

    gamma: Float.
        With of the distribution.

    a: Float.
        Amplitud of the curve.

    b: Float.
        b value in a exp(x*b + c) background.

    c: Float.
        c value in a exp(x*b + c) background.

    """
    return bkg_lorentzian(
        x, x0, gamma, a, b, c, bgk_func=lambda x, b, c: np.exp(x*b + c)
    )


def potbkg_lorentzian(x, x0, gamma, a, b, c):
    """Lorentzian + potential background for curves fitting.

    Parameters
    ----------
    x: Float | Array of floats.
        Points in which the function will be evaluated.

    x0: Float.
        Mean value of the distribution.

    gamma: Float.
        With of the distribution.

    a: Float.
        Amplitud of the curve.

    b: Float.
        b value in a c * x^b background.

    c: Float.
        c value in a c * x^b background.

    """
    return bkg_lorentzian(
        x, x0, gamma, a, b, c, bgk_func=lambda x, b, c: c*x**b
    )
