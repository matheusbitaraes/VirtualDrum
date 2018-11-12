from HearBeat import HearBeat
import numpy as np

hb = HearBeat()
volume = 0.1     # range [0.0, 1.0]
fs = 44100       # sampling rate, Hz, must be integer
duration = 0.3   # in seconds, may be float
f = 750.0        # sine frequency, Hz, may be float
ind = 0
samples = (np.sin(2*np.pi*np.arange(fs*duration)*f/fs)).astype(np.float32)
print(ind)
hb.setSamples(samples)
hb.start()
while ind != 500000:
    print(ind)
    ind = ind + 1

hb.stop()
