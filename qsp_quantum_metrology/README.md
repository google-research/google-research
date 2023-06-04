# QSPCalibration

This is the support code for the Paper:
Beyond Heisenberg Limit Quantum Metrology through Quantum Signal Processing,
https://arxiv.org/abs/2209.11207


Requirements:

- python >= 3.9
- numpy >= 1.23.5
- scipy >= 1.10.1
- cirq >= 1.1.0

`circuit_generation.py` deals with the circuit generation part. If the circuit
is executed on a real device, some functions may be changed for compatibility.

`fft_analysis.py` deals with classical data postprocessing. No quantum computing
backend is included. The code is independent with the circuit generation.

main_fft_calibrate.py is a demo code for using APIs.

Sample outputs:

```
$ python -m main_fft_calibrate.py 
theta = 0.0003217402947098892
varphi = 0.43733127381301434
alpha = 1.000451531074018

$ python main_fft_calibrate.py
theta = 0.00029413466896938736
varphi = 0.323017603981168
alpha = 0.9995551634620906

$ python main_fft_calibrate.py
theta = 0.00025526304952497856
varphi = 1.484152824413771
alpha = 0.9994942598083861
```

Note that `varphi` is not well defined in a perfect `CZ` because `theta = 0` and
any `varphi` can be inferred due to the translational invariance of FFT.
`theta` is inferred consistently and ~1e-4 error is intrinsic by statistics.
