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
    plt.subplot(212)
    plt.plot(time, wave_data[1], c="g")
    plt.xlabel("time (seconds)")
    plt.show()
    utils.print_line('end draw wav')


def wav_hide_information(wavfile, message):
    wdata = map(ord, message)
    wdata = np.array(wdata)


def gen_noise_wav(message):
    framerate = 44100
    cchannels = 2
    samplewidth = 2
    cframes = framerate * 4
    base_amplitude = 200
    max_amplitude = 128 * base_amplitude
    interval = (cframes - 10) / message.__len__()

    wav_data = np.zeros(cframes, dtype=np.short)


wavfilename = "./res/back.wav"
f = wave.open(wavfilename, "rb")

hide_string = "hide string in wav"
wav_hide_information(f, "hiden messages")
# draw_wav(f)

