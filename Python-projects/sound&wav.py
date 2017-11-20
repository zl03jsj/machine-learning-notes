#!/usr/bin/env python
# -*- coding: utf-8 -*-
# code:myhaspl@qq.com
# 3-13.py

import wave
import pylab as pl
import matplotlib.pyplot as plt
import numpy as np
import utils


def draw_wav(wavfile):
    utils.print_line('begin draw wav')
    params = wavfile.getparams()
    # (声道数, 采样精度, 采样率, 帧数，......
    print params
    nchannels, sampwidth, framerate, nframes = params[:4]
    str_data = wavfile.readframes(nframes)
    wave_data = np.fromstring(str_data, dtype=np.short)
    wave_data.shape = -1, nchannels
    wave_data = wave_data.T

    time = np.arange(0, nframes) / (1.0 / framerate)

    # # 绘制波形
    # pl.subplot(211)
    # pl.plot(time, wave_data[0])
    # pl.subplot(212)
    # pl.plot(time, wave_data[1], c="g")
    # pl.xlabel("time (seconds)")
    # pl.show()

    plt.subplot(2, 1, 1)
    plt.plot(time, wave_data[0])
    plt.subplot(2, 1, 2)
    plt.plot(time, wave_data[1], c="g")
    plt.xlabel("time (seconds)")
    plt.show()
    utils.print_line('end draw wav')

framerate = 44100
nchannels = 2
sampwidth = 2
n_cycle = 4
base_amplitude = 200
max_amplitude = 128 * base_amplitude

interval = 1

def wav_hide_information(message):
    mdata = map(ord, message)
    mdata = np.array(mdata)

    wav_data = gen_noise_wav(message)
    nframe = wav_data.__len__()
    global interval

    interval = nframe / message.__len__()

    count = 0
    rand = np.random.rand(nframe)

    print "rand is:", rand

    utils.print_line('encrypt chars begin')
    for curpos in xrange(0, nframe):
        if curpos%interval == 0 and count<mdata.__len__():
            c = mdata[count] * base_amplitude - 64 * base_amplitude
            count += 1
        elif curpos%60 == 0:
            c = int(rand[curpos] * max_amplitude - max_amplitude / 2)
        else:
            c = 0
        wav_data[curpos] = c

    print wav_data

    utils.print_line('encrypt chars end')
    return wav_data

def write_wav(wav, filepath):
    wav_outfile = wave.open(filepath, "wb")
    stringdata = wav.tostring()
    wav_outfile.setnchannels(nchannels)
    wav_outfile.setframerate(framerate)
    wav_outfile.setsampwidth(sampwidth)
    wav_outfile.setnframes(framerate * n_cycle)
    wav_outfile.writeframes(stringdata)
    wav_outfile.close()


def wav_get_hiden_message(wav_data, lmsg):
    len = wav_data.__len__()
    n = 0
    mesage = []

    utils.print_line('uncrypt chars begin')
    print interval
    for i in xrange(0, len):
        if i%interval == 0:
            mesage.append(int((wav_data[i] + 64*base_amplitude)/base_amplitude))
            n += 1
            if n==(lmsg-1):
                break
    utils.print_line('uncrypt chars end')
    mesage = "".join(map(chr, mesage))
    return mesage



def gen_noise_wav(message):
    cframes = framerate * n_cycle
    wav_data = np.zeros(cframes, dtype=np.short)
    return wav_data


# wavfilename = "./res/back.wav"
# f = wave.open(wavfilename, "rb")
# draw_wav(f)



wavfilename = "./res/noiseWave.wav"

hide_string = "i love you!!!!!"
wav = wav_hide_information(hide_string)
write_wav(wav, wavfilename)

f = wave.open(wavfilename, "rb")
draw_wav(f)

uncrypt_message = wav_get_hiden_message(wav, hide_string.__len__())

print "the hide information is:", uncrypt_message

