from typing import Any, Tuple
import numpy as np
import scipy.signal as sps
import scipy.fft as sfft
from matplotlib import axes
import matplotlib.pyplot as plt

# =====================================================================
class Diffuser:
    """
    Parent class for a diffuser, which represents either an actual
    diffuser or a rough surface.

    Do not instantiate this directly; only use the sub-classes.
    """
    def __init__(
        self,
        dims: tuple = (1024, 1024),
        dtype: Any = np.float32
    ):
        self._dims = dims
        self._dtype = dtype

        # Call the (sub-)class generators
        self._phase = self._generatePhases()
        self._kernel, self._phase = self._correlatePhases()
        self._field = self._makeField()

    def _generatePhases(self) -> np.ndarray:
        raise NotImplementedError("Subclass Diffuser and implement this.")

    def _correlatePhases(self) -> Tuple[np.ndarray | None, np.ndarray]:
        """
        Default to no-op.
        """
        return None, self._phase

    def _makeField(self) -> np.ndarray:
        """
        Defaults to unity amplitude.
        """
        return np.exp(1j*self._phase)

    def plotFieldPowerSpectrum(self, ax: axes._axes.Axes, **kwargs):
        ax.imshow(
            self.spectrum(),
            **kwargs
        )
        ax.set_title("Field Power Spectrum")

    def plotFieldIntensity(self, ax: axes._axes.Axes, **kwargs):
        ax.imshow(self.intensity(), **kwargs)
        ax.set_title("Field Intensity")

    @property
    def phase(self) -> np.ndarray:
        """
        The phase property.
        """
        return self._phase

    @property
    def field(self) -> np.ndarray:
        """The field property."""
        return self._field

    def spectrum(self) -> np.ndarray:
        """
        Calculates the power spectrum of the field
        i.e. magnitude squared of 2D FFT of field.

        Returns
        -------
        spectrum : np.ndarray
            Spectrum matrix.
        """
        return np.abs(sfft.fftshift(
            sfft.fft2(self.field)
        ))**2

    def intensity(self) -> np.ndarray:
        """
        Calculates the intensity i.e.
        magnitude squared of the field.

        Returns
        -------
        intensity : np.ndarray
            Intensity matrix.
        """
        return np.abs(self._field)**2


# =====================================================================
class NormalDiffuser(Diffuser):
    """
    A basic diffuser with normally distributed random phases.
    """
    def __init__(
        self,
        mean_phase: float = 0.0,
        std_phase: float = 2*np.pi,
        dims: tuple = (1024, 1024),
        dtype: Any = np.float32
    ):
        # Store normal distribution parameters before generating phases
        self._mean_phase = mean_phase
        self._std_phase = std_phase

        # Generate the phase
        super().__init__(dims, dtype)

    def _generatePhases(self):
        return np.random.normal(self._mean_phase, self._std_phase, self._dims).astype(self._dtype)

# =====================================================================
class CircularCorrelatedNormalDiffuser(NormalDiffuser):
    """
    A diffuser with normally distributed random phases and
    a correlation radius, implemented via convolution with
    a circular window.
    """
    def __init__(
        self,
        mean_phase: float = 0.0,
        std_phase: float = 2*np.pi,
        corrRadius: int = 20,
        dims: tuple = (1024, 1024),
        dtype: Any = np.float32
    ):
        # Store correlation window parameters before generating phases
        self._corrRadius = corrRadius

        # Generate the phase
        super().__init__(mean_phase, std_phase, dims, dtype)

    def _correlatePhases(self) -> Tuple[np.ndarray, np.ndarray]:
        kernel = np.zeros(self._dims, self._dtype)
        x, y = np.meshgrid(np.arange(self._dims[0]), np.arange(self._dims[1]))
        radius = np.sqrt((x - self._dims[0]/2)**2 + (y - self._dims[1]/2)**2)
        # Set to 1/sqrt(pi r^2)
        kernel[radius < self._corrRadius] = 1.0/(np.pi**0.5 * self._corrRadius)

        corrphase = sps.fftconvolve(self._phase, kernel, mode='same')

        return kernel, corrphase


# =================== Simple standalone testing
if __name__ == "__main__":
    plt.close('all')

    ndiffuser = NormalDiffuser()
    fig, ax = plt.subplots(1,2,num="normal diffuser")
    ndiffuser.plotFieldPowerSpectrum(ax[0], cmap='gray')
    ndiffuser.plotFieldIntensity(ax[1], cmap='gray')

    ccndiffuser = CircularCorrelatedNormalDiffuser()
    cfig, cax = plt.subplots(1,2,num="circular correlated diffuser")
    ccndiffuser.plotFieldPowerSpectrum(cax[0], cmap='gray')
    ccndiffuser.plotFieldIntensity(cax[1], cmap='gray')

    plt.show()
    print("Done")

