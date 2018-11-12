# ref: https://www.pyimagesearch.com/2015/12/21/increasing-webcam-fps-with-python-and-opencv/
# import the necessary packages
from threading import Thread
import pyaudio
import math
import struct


class HearBeat:
    def __init__(self, CHUNK = 1024, FORMAT = pyaudio.paInt16, CHANNELS = 2, RATE = 44100, fs = 44100):

        # initialize the variable used to indicate if the thread should
        # be stopped
        self.paused = True
        self.stopped = False
        self.CHUNK = CHUNK
        self.samples = 0
        # Audio stuff
        self.p = pyaudio.PyAudio()

        self.streamIn = self.p.open(format=FORMAT,channels=CHANNELS,rate=RATE,input=True,frames_per_buffer=CHUNK)

        # for paFloat32 sample values must be in range [-1.0, 1.0]
        self.streamOut = self.p.open(format=pyaudio.paFloat32,channels=1,rate=fs,output=True)

    def start(self):
            # start the thread to listen to audio
            Thread(target=self.listen, args=(), daemon=True).start()
            return self

    def listen(self):
        # keep looping infinitely until the thread is stopped
        cnt = 0
        while True:
            # if the thread indicator variable is set, stop the thread
            if self.paused:
                pass
            else:
                # otherwise, keep listening and playing sounds
                data = self.streamIn.read(self.CHUNK, exception_on_overflow=False)

                # RMS value calculation
                count = len(data) / 2
                format = "%dh" % (count)
                shorts = struct.unpack(format, data)
                sum_squares = 0.0
                for sample in shorts:
                    n = sample * (1.0 / 32768)
                    sum_squares += n * n
                energy = math.sqrt(sum_squares / count)
                # check energy and play sound
                if energy > 0.01 and cnt > 5:
                    self.streamOut.get_write_available()
                    self.streamOut.write(self.samples * (0.1 + energy * 4.5))
                    cnt = 0
                else:
                    cnt = cnt + 1
            if self.stopped:
                return

    def setSamples(self, samples):
        self.samples = samples

    def stop(self):
        # indicate that the thread should be stopped
        self.stopped = True

    def pause(self):
        # indicate that the thread should be stopped
        self.paused = True
    
    def retake(self):
        # indicate that the thread should be stopped
        self.paused = False

    def terminate(self):
        self.streamIn.stop_stream()
        self.streamOut.stop_stream()
        self.streamIn.stop_stream()
        self.streamOut.stop_stream()
        self.p.terminate()

