from spiking_layer import SpikingLayer
import numpy as np
from numpy.typing import ArrayLike


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
