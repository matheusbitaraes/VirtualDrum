import threading
import time
import pyaudio
import numpy as np

#thread that will be created in each loop 
class thr_1 (threading.Thread):
    def __init__(self, sound, intensity):
        threading.Thread.__init__(self)
        self.sound = sound
        self.intensity = intensity
    def run(self):
        global t2
        t2.append(time.time())
        stream.write(volume * samples)
        # colocar função de execução de som
        t2.append(time.time())
    
#thread with event
class thr_executesound (threading.Thread):
    def __init__(self, sound, intensity):
        threading.Thread.__init__(self)
        self.sound = sound
        self.intensity = intensity
        self.daemon = True #this thread will not interrup the end of the execution - it will die after exit()
    def run(self):
        for a in range(0,20):
            time.sleep(0.1)
            print("eu")
            t200.append(time.time())
        

t1 = []
t2 = []
t3 = []
t10 = []
t20 = []
t30 = []
t100 = []
t200 = []
t300 = []
loops = 50

p = pyaudio.PyAudio()

volume = 0.5     # range [0.0, 1.0]
fs = 44100       # sampling rate, Hz, must be integer
duration = 1   # in seconds, may be float
f = 750.0        # sine frequency, Hz, may be float

# generate samples, note conversion to float32 array
samples = (np.sin(2*np.pi*np.arange(fs*duration)*f/fs)).astype(np.float32)

# for paFloat32 sample values must be in range [-1.0, 1.0]
stream = p.open(format=pyaudio.paFloat32,
                channels=1,
                rate=fs,
                output=True)

for i in range(0,loops): 
    if i % 5 == 0:
        t10.append(time.time())
        print("som " + str(i))
        t20.append(time.time())


for i in range(0,loops): 
    if i % 5 == 0:
        t1.append(time.time())
        thr_1("som " + str(i), 2).start()
        

thr_executesound ("show", 2).start()
for i in range(0,loops): 
    if i % 5 == 0:
        t100.append(time.time())



time.sleep(3)
print ("test without thread:")
for i in range(len(t10)):
    t30.append(1000*(t20[i] - t10[i]))
print("maximum time for printing (ms): " + str(max(t30)))
print("average time for printing (ms): " + str(sum(t30)/len(t30)))



print ("test with thread (creating one thread each loop):")
for i in range(len(t1)):
    t3.append(1000*(t2[i] - t1[i]))
print("maximum time for printing (ms): " + str(max(t3)))
print("average time for printing (ms): " + str(sum(t3)/len(t3)))



print ("test with thread event (just one thread for the sound generation):")
#for i in range(len(t200)):
    #t300.append(1000*(t200[i] - t100[i]))
#print("maximum time for printing (ms): " + str(max(t300)))
#print("average time for printing (ms): " + str(sum(t300)/len(t300)))

#exit() #- used to kill daemon threads