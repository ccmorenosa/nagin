"""Statistic and distribution functions."""
import numpy as np
from scipy.constants import k


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


def noise_lorentzian(x, x0, gamma, a, b, c):
    """Lorentzian + 1/f noise background for curves fitting.

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
        b value in a b * x^-1 + c background.

    c: Float.
        c value in a b * x^-1 + c background.

    """
    return bkg_lorentzian(
        x, x0, gamma, a, b, c, bgk_func=lambda x, b, c: b*x**-1 + c
    )


def S(f, T, G, S_A, S_B, S_C, alpha, Z, *zargs):
    r"""General Noise curve.

    The curve describe a noise of the form:

    \(S_{v^2} = (4k_BT + S_A) * Z_{eq} * G^2 + S_B/f^{\alpha} + S_C\)

    where `Z_{eq} = Re{Z(f, *zargs)}`

    Parameters
    ----------
    f: float.
        Frequency at which the value is evaluated.

    T: float.
        Temperature of the system.

    G: float.
        Gain of the system from the cold chamber to the output.

    S_A: float.
        Scaling factor for the T-independent contribution of the equivalent
        impedance.

    S_B: float.
        Constant of the potential contribution of the noise 1/f^alpha.

    S_C: float.
        Noise offset.

    alpha: float.
        Exponent of the potential contribution of the noise 1/f^alpha.

    Z: float.
        Equivalent impedance of the system.

    zargs: float.
        Arguments of Z.

    """
    Z_eq = np.real(Z(f, *zargs))
    return (4 * k * T + S_A) * Z_eq * G**2 + S_B/f**alpha + S_C


def Z_QPC_LC(f, Rqpc, L=2.2e-6, C=400e-12, Rp=2):
    """Z_eq of the circuit with the resistance of the QPC and a LC resonator.

    Parameters
    ----------
    f: float.
        Frequency at which the value is evaluated.

    Rqpc:
        Resistance of the QPC.

    L:
        Inductance of the LC system.

    C:
        Capacitance of the LC system.

    Rp:
        Parasite resistance of the LC system.

    """
    w = 2*np.pi*f

    Z_L = 1j*w*L + Rp

    Y_QPC_LC = 0j * np.zeros_like(w)
    Y_QPC_LC += 1/Rqpc
    Y_QPC_LC += 1/Z_L
    Y_QPC_LC += 1j*w*C

    return 1/Y_QPC_LC


def S_QPC_LC(f, T, G, S_A, S_B, S_C, alpha, Rqpc, L, C, Rp):
    """General Noise curve for a RL circuit connected to the QPC.

    Parameters
    ----------
    f: float.
        Frequency at which the value is evaluated.

    T: float.
        Temperature of the system.

    G: float.
        Gain of the system from the cold chamber to the output.

    S_A: float.
        Scaling factor for the T-independent contribution of the equivalent
        impedance.

    S_B: float.
        Constant of the potential contribution of the noise 1/f^alpha.

    S_C: float.
        Noise offset.

    alpha: float.
        Exponent of the potential contribution of the noise 1/f^alpha.

    Z: float.
        Equivalent impedance of the system.

    Rqpc:
        Resistance of the QPC.

    L:
        Inductance of the LC system.

    C:
        Capacitance of the LC system.

    Rp:
        Parasite resistance of the LC system.

    """
    return S(f, T, G, S_A, S_B, S_C, alpha, Z_QPC_LC, Rqpc, L, C, Rp)
