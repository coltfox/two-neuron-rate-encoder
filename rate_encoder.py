from spiking_layer import SpikingLayer
import numpy as np
from numpy.typing import ArrayLike
import matplotlib.pyplot as plt
from matplotlib.colors import LinearSegmentedColormap


class RateEncoder(SpikingLayer):
    """Encodes an input as spikes using rate encoding"""

    def __init__(self, n_neurons: int, encoders: ArrayLike | int = 1, v_thr: float = 1.0, gain: float = 1.0, bias: float = 0.0) -> None:
        """
        Args:
            n_neurons (int): Number of neurons in the layer
            batch_size (int): Size of batches. Default 500.
            v_thr (float): Voltage threshold. Default 1.0
            device (torch.device | str): Device to allocate tensors. Default 'cpu'
            gain (float): Neuron gain multiplier. Default 1.0
            bias (float): Neuron bias. Default 0.0
        """
        super().__init__(n_neurons, v_thr)
        self._c = np.zeros(n_neurons)
        self.gain = gain
        self.bias = bias
        self.encoders = encoders

    def spike(self) -> ArrayLike:
        v_diff = self._v - self._v_thr
        spikes = np.zeros_like(self._v)
        spikes[v_diff >= 0] = 1

        return spikes

    def encode(self, x_t: ArrayLike) -> ArrayLike:
        self._c = self.encoders * self.gain * x_t + self.bias
        self.calculate_voltage(self._c)

        return self.spike_and_reset_voltage()


# from nengo
cm_gray_r_a = LinearSegmentedColormap.from_list(
    "gray_r_a", [(0.0, 0.0, 0.0, 0.0), (0.0, 0.0, 0.0, 1.0)]
)


def plot_spikes(t, spikes, contrast_scale=1.0, ax=None, **kwargs):
    """Plots a spike raster.

    Will use an alpha channel by default which allows to plot colored regions
    below the spike raster to add highlights. You can set the *cmap* keyword
    argument to a different color map like *matplotlib.cm.gray_r* to get an
    opaque spike raster.

    Utilizes Matplotlib's *imshow*.

    Parameters
    ----------
    t : (n,) array
        Time indices of *spike* matrix. The indices are assumed to be
        equidistant.
    spikes : (n, m) array
        Spike data for *m* neurons at *n* time points.
    contrast_scale : float, optional
        Scales the contrst of the spike raster. This value multiplied with the
        maximum value in *spikes* will determine the minimum *spike* value to
        appear black (or the corresponding color of the chosen colormap).
    ax : matplotlib.axes.Axes, optional
        Axes to plot onto. Uses the current axes by default.
    kwargs : dict
        Additional keyword arguments will be passed on to *imshow*.

    Returns
    -------
    matplotlib.image.AxesImage
        The spikeraster.
    """

    t = np.asarray(t)
    spikes = np.asarray(spikes)
    if ax is None:
        ax = plt.gca()

    kwargs.setdefault("aspect", "auto")
    kwargs.setdefault("cmap", cm_gray_r_a)
    kwargs.setdefault("interpolation", "nearest")
    kwargs.setdefault("extent", (t[0], t[-1], 0.0, spikes.shape[1]))

    spikeraster = ax.imshow(spikes.T, **kwargs)
    spikeraster.set_clim(0.0, np.max(spikes) * contrast_scale)
    return spikeraster
