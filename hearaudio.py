import pyaudio
import numpy as np
import math
import struct
import simpleaudio as sa
from imutils.video import FPS


#CHUNK = 1024
CHUNK = 1024
FORMAT = pyaudio.paInt16
CHANNELS = 2
RATE = 44100
RECORD_SECONDS = 0.03
print(int(RATE / CHUNK * RECORD_SECONDS))

volume = 0.1     # range [0.0, 1.0]
fs = 44100       # sampling rate, Hz, must be integer
duration = 0.3   # in seconds, may be float
f = 750.0        # sine frequency, Hz, may be float

fps = FPS().start()

def rms(data):
    if data != []:
        count = len(data)/2
        format = "%dh"%(count)
        shorts = struct.unpack(format, data)
        sum_squares = 0.0
        for sample in shorts:
            n = sample * (1.0/32768)
            sum_squares += n*n
        return math.sqrt(sum_squares / count)
    return 0


p = pyaudio.PyAudio()
p2 = pyaudio.PyAudio()

stream = p.open(format=FORMAT,
                channels=CHANNELS,
                rate=RATE,
                input=True,
                frames_per_buffer=CHUNK)

# for paFloat32 sample values must be in range [-1.0, 1.0]
streamout = p2.open(format=pyaudio.paFloat32,
                   channels=1,
                   rate=fs,
                   output=True)

# generate samples, note conversion to float32 array
samples = (np.sin(2*np.pi*np.arange(fs*duration)*f/fs)).astype(np.float32)

wave_obj = sa.WaveObject.from_wave_file("bumbo.wav")
ind = 0
count = 0
while ind != 500:
    frames = []
    energy = []
    for _ in range(0, int(RATE / CHUNK * RECORD_SECONDS)):
        try:
            data = stream.read(CHUNK, exception_on_overflow=False)
        except IOError:
            #data = '\x00' * CHUNK
            data = []
        energy.append(rms(data))
    if np.mean(energy) > 0.01 and count > 10:
        #wave_obj.play()
        streamout.get_write_available()
        #streamout.write(samples * np.mean(energy)/0.2)
        streamout.write(samples * (volume + np.mean(energy) * 5))
        print(np.mean(energy))
        # energy = 0.02 - volume = 0.1
        # energy = 0.2 - volume = 1
        count = 0
    else:
        count = count + 1

    ind = ind + 1
    fps.update()

# stop the timer and display FPS information
fps.stop()
print("[INFO] elapsed time: {:.2f}".format(fps.elapsed()))
print("[INFO] approx. FPS: {:.2f}".format(fps.fps()))

#energy = np.mean(energy)
#print("Average energy in an execution of " + str(RECORD_SECONDS*ind) + " Seconds: " + str(energy))
stream.stop_stream()
stream.close()
streamout.stop_stream()
streamout.close()
p.terminate()
p2.terminate()

