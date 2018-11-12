import time
import pyaudio
import numpy as np

p = pyaudio.PyAudio()

volume = 0.5     # range [0.0, 1.0]
fs = 44100       # sampling rate, Hz, must be integer
duration = 1   # in seconds, may be float
f = 750.0        # sine frequency, Hz, may be float

t1 = []
t2 = []
t3 = []

# generate samples, note conversion to float32 array
samples = (np.sin(2*np.pi*np.arange(fs*duration)*f/fs)).astype(np.float32)

# for paFloat32 sample values must be in range [-1.0, 1.0]
stream = p.open(format=pyaudio.paFloat32,
                channels=1,
                rate=fs,
                output=True)

# play. May repeat with different volume values (if done interactively)
for i in range(10):
    print("signal sent")
    t1.append(time.time())
    stream.write(volume*samples)
    t2.append(time.time())

stream.stop_stream()
stream.close()

p.terminate()

print ("test with thread (creating one thread each loop):")
for i in range(len(t1)):
    t3.append(1000*(t2[i] - t1[i]))
print("maximum time for printing (ms): " + str(max(t3)))
print("average time for printing (ms): " + str(sum(t3)/len(t3)))