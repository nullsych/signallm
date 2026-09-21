import numpy as np, matplotlib.pyplot as plt
from scipy import signal

raw = np.fromfile("cap.bin", dtype=np.int16)
iq = (raw[0::2] + 1j*raw[1::2]).astype(np.complex64) / 2048.0

f, p = signal.welch(iq, fs=20e6, nperseg=4096, return_onesided=False)
f = np.fft.fftshift(f) + 2437e6
plt.plot(f/1e6, 10*np.log10(np.fft.fftshift(p)))
plt.show()