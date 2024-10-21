import numpy as np
from scipy.fft import rfft, irfft
import cmath as cm
import os


# Set the number of threads to 1 for OpenMP and BLAS to avoid conflicts with multiprocessing
os.environ["OMP_NUM_THREADS"] = "1"
os.environ['OPENBLAS_NUM_THREADS'] = '1'
os.environ['MKL_NUM_THREADS'] = '1'


def predicted_current_pulse(voltage_array_pulse: np.ndarray,
                            Y0, Rs, Rf, nY, frequency_acquisition) -> np.ndarray:
    """
    For a given voltage pulse, returns the current pulse predicted by the equivalent circuit model defined by
    parameters Y0, Rs, Rf, and nY.

    Parameters
    ----------
    voltage_array_pulse: np.ndarray
        Voltage pulse, in volts.
    Y0: float
        Variable Y0 of CPE. Typical range: 21.19e-9 - 75.44e-9 EIS to 329e-6 - 879e-6 Helmholtz or Stern.
    Rs: float
        Solution resistance in Ohms. Typical range: 65 - 105.
    Rf: float
        Charge transfer resistance in Ohms. Typical range: 260 - 310.
    nY: float
        Exponent of the CPE.
    frequency_acquisition: int
        Frequency of data acquisition, in samples per second.

    Returns
    -------
    np.ndarray
        Predicted current pulse, in amperes. The length of the array is the same as the input voltage pulse.
        The time between points is the same as the input voltage pulse.

    """
    Vfft = rfft(voltage_array_pulse)
    time_step = 1 / frequency_acquisition
    n = voltage_array_pulse.size
    freq = np.fft.rfftfreq(n, d=time_step)

    ## Impedance calculation
    # CPE impedance
    CPEimpedance_array = (1 / (((2j * cm.pi * (
                freq + 1e-12)) ** nY) * Y0))  # epsilon frequencies (1e-12) added to avoid issue with division by zero
    # Total impedance of the system
    Total_impedance_array = Rs + 1 / ((1 / CPEimpedance_array) + (1 / (Rf)))

    ## Predicted current in the Fourier space
    I_fft = Vfft / Total_impedance_array

    ## Inverse FFT
    pulse_predicted_current = irfft(I_fft)

    return pulse_predicted_current


def predicted_current_pulse_precomp_rfft(Vfft, n, Y0, Rs, Rf, nY, frequency_acquisition) -> np.ndarray:
    """
    For a given voltage pulse, returns the current pulse predicted by the equivalent circuit defined by
    parameters Y0, Rs, Rf, and nY.
    This function is optimized for repeated calls with the same voltage pulse, and so it is using
    the Fourier transform of the voltage pulse as an input, instead of the pulse itself.
    Parameters
    ----------
    Vfft: np.ndarray
        Fourier transform of the voltage pulse.
    n: int
        Number of points in the voltage pulse.
    Y0: float
        Variable Y0 of CPE. Typical range: 21.19e-9 - 75.44e-9 EIS to 329e-6 - 879e-6 Helmholtz or Stern.
    Rs: float
        Solution resistance in Ohms. Typical range: 65 - 105.
    Rf: float
        Charge transfer resistance in Ohms. Typical range: 260 - 310.
    nY: float
        Exponent of the CPE.
    frequency_acquisition: int
        Frequency of data acquisition, in samples per second

    Returns
    -------
    np.ndarray
        Predicted current pulse, in amperes. The length of the array is the same as the input voltage pulse.
        The time between points is the same as the input voltage pulse.

    """
    time_step = 1 / frequency_acquisition
    freq = np.fft.rfftfreq(n, d=time_step)

    ##Impedance calculation
    # CPE impedance
    CPEimpedance_array = (1 / (((2j * cm.pi * (
            freq + 1e-12)) ** nY) * Y0))  # epsilon frequences added to avoid issue with division by zero
    # Total impedance of the system
    Total_impedance_array = Rs + 1 / ((1 / CPEimpedance_array) + (1 / (Rf)))

    ##Predicted current
    I_fft = Vfft / Total_impedance_array

    ##iFFT
    pulse_predicted_current = irfft(I_fft)

    return pulse_predicted_current


def predicted_faradaic_current_pulse(voltage_array_pulse: np.ndarray,
                                     Y0, Rs, Rf, nY, frequency_acquisition) -> np.ndarray:
    """
    For a given voltage pulse, returns the Faradaic current pulse predicted by the equivalent circuit model defined by
    parameters Y0, Rs, Rf, and nY.

    Parameters
    ----------
    voltage_array_pulse: np.ndarray
        Voltage pulse, in volts.
    Y0: float
        Variable Y0 of CPE. Typical range: 21.19e-9 - 75.44e-9 EIS to 329e-6 - 879e-6 Helmholtz or Stern.
    Rs: float
        Solution resistance in Ohms. Typical range: 65 - 105.
    Rf: float
        Charge transfer resistance in Ohms. Typical range: 260 - 310.
    nY: float
        Exponent of the CPE.
    frequency_acquisition: int
        Frequency of data acquisition, in samples per second.

    Returns
    -------
    np.ndarray
        Predicted Faradaic current pulse, in amperes. The length of the array is the same as the input voltage pulse.
        The time between points is the same as the input voltage pulse.

    """
    Vfft = rfft(voltage_array_pulse)
    time_step = 1 / frequency_acquisition
    n = voltage_array_pulse.size
    freq = np.fft.rfftfreq(n, d=time_step)

    ##Impedance calculation
    # CPE impedance
    # epsilon frequences (1e-12) added to avoid issue with division by zero
    CPEimpedance_array = (1 / (((2j * cm.pi * (freq + 1e-12)) ** nY) * Y0))
    # Total impedance of the system
    Total_impedance_array = Rs + 1 / ((1 / CPEimpedance_array) + (1 / (Rf)))

    ##Predicted current
    I_faradaic_fft = (Vfft - ((Vfft / Total_impedance_array) * Rs)) / (Rf)

    ##iFFT
    pulse_faradaic_current = irfft(I_faradaic_fft)

    return pulse_faradaic_current


if __name__ == '__main__':
    pass
