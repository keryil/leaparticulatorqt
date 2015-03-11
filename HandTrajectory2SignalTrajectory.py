# -*- coding: utf-8 -*-
# <nbformat>3.0</nbformat>

# <codecell>

# files = !ls logs/*.responses.csv

# <codecell>

import pandas as pd
import numpy as np
import csv
from LeapTheremin import palmToAmpAndFreq, palmToFreq, freqToMel
from StreamlinedDataAnalysis import id_to_log

# <codecell>

# taken from ClientUI.py
# end of steal

def calculate_amp_and_freq(f, delimiter="|"):
    import Constants
    default_volume = Constants.default_amplitude
    default_pitch = Constants.default_pitch
    freqs = []
    amps = []
    mels = []
    images = []
    cond = f.split(".")[1]
    print "File: %s, Condition: %s" % (f, cond)
    data = pd.read_csv(f, delimiter=delimiter, na_values=["NaN"])
    new_file = ".".join(f.split(".")[:-1]) + ".freq_and_amp.csv"
    
    
    series = lambda x: pd.Series(x, index=data.index)
    normalize = lambda x: (x - np.average(x)) / np.std(x)
    norm_series = lambda x: series(normalize(x))
    doublequote = lambda x: "\"%s\"" % x
    
    print new_file
    oldx,oldy = -1,-1
    for row in data[['x','y', 'phase', 'image']].iterrows():
        x,y = row[1][0], row[1][1]
        phase = int(row[1][2])
        image = row[1][3]
        if cond in ('1r', '2r'):
            x, y = y, x
        amp, freq = palmToAmpAndFreq((x,y,0))
        
        
        if cond[-1] == 'r' and cond[-2] != 'e':
            if (cond in ('1', 'master') and phase == 1) or \
                ('2' in cond and phase == 2) or \
                phase ==0:
                    freq = default_pitch
        else: 
            if (cond in ('1', 'master') and phase == 1) or \
                ('2' in cond and phase == 2) or \
                phase ==0:
                    amp = default_volume
        mel = freqToMel(freq)
#         print phase, amp, freq, mel
        freqs.append(freq)
        amps.append(amp)
        mels.append(mel)
        images.append(doublequote(image))
    data["frequency"] = series(freqs)
    data["amplitude"] = series(amps)
    data["mel"] = series(mels)
    data["frequency_n"] = norm_series(freqs)
    data["amplitude_n"] = norm_series(amps)
    data["mel_n"] = norm_series(mels)
    data["image"] = series(images)
    data.to_csv(new_file, sep="|", na_rep="NaN", quoting=csv.QUOTE_NONE)
    
def doit():
#     files = []
    for f in files:
        calculate_amp_and_freq(f)
    
#         print amp, freq
#         print phase
            #             print palmToAmpAndFreq((x,y,z))

# <codecell>

# doit()

# <codecell>


