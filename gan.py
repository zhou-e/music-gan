'''

DCGAN on MNIST using Keras
Authors: Edward Zhou, Sammy Zhang, Colin Chen, David Chong
Project: https://github.com/ColinAChen/gan_experiments
Dependencies: tensorflow 1.0 and keras 2.0
Adapted from: https://github.com/roatienza/Deep-Learning-Experiments

'''
import math
import cv2
import pretty_midi
import os
import numpy as np
import time

import keras
from keras.models import Sequential
from keras.layers import Dense, Activation, Flatten, Reshape
from keras.layers import Conv2D, Conv2DTranspose, UpSampling2D
from keras.layers import LeakyReLU, Dropout
from keras.layers import BatchNormalization
from keras.optimizers import Adam

import matplotlib.pyplot as plt

class ElapsedTimer(object):

    def __init__(self):
        self.start_time = time.time()

    def elapsed(self,sec):
        if sec < 60:
            return str(sec) + " sec"
        elif sec < (60 * 60):
            return str(sec / 60) + " min"
        else:
            return str(sec / (60 * 60)) + " hr"

    def elapsed_time(self):
        print("Elapsed: %s " % self.elapsed(time.time() - self.start_time) )


class DCGAN(object):

    def __init__(self, img_rows=28, img_cols=28, channel=1):
        self.img_rows = img_rows
        self.img_cols = img_cols
        self.channel = channel
        self.D = None   # discriminator
        self.G = None   # generator
        self.AM = None  # adversarial model
        self.DM = None  # discriminator model

    # (W−F+2P)/S+1

    def discriminator(self):

        if self.D:
            return self.D

        self.D = Sequential()
        depth = 64
        dropout = 0.4
        # In: 28 x 28 x 1, depth = 1
        # Out: 14 x 14 x 1, depth=64
        input_shape = (self.img_rows, self.img_cols, self.channel)
        self.D.add(Conv2D(depth*1, 5, strides=2, input_shape=input_shape,\
            padding='same'))
        self.D.add(LeakyReLU(alpha=0.2))
        self.D.add(Dropout(dropout))

        self.D.add(Conv2D(depth*2, 5, strides=2, padding='same'))
        self.D.add(LeakyReLU(alpha=0.2))
        self.D.add(Dropout(dropout))

        self.D.add(Conv2D(depth*4, 5, strides=2, padding='same'))
        self.D.add(LeakyReLU(alpha=0.2))
        self.D.add(Dropout(dropout))

        self.D.add(Conv2D(depth*8, 5, strides=1, padding='same'))
        self.D.add(LeakyReLU(alpha=0.2))
        self.D.add(Dropout(dropout))

        self.D.add(Conv2D(depth*16, 5, strides=1, padding='same'))
        self.D.add(LeakyReLU(alpha=0.2))
        self.D.add(Dropout(dropout))

        # Out: 1-dim probability
        self.D.add(Flatten())
        self.D.add(Dense(1))
        self.D.add(Activation('sigmoid'))
        self.D.summary()

        return self.D



    def generator(self):

        if self.G:
            return self.G
        self.G = Sequential()
        dropout = 0.4
        depth = 64+64+64+64
        dim = 7
        # In: 100
        # Out: dim x dim x depth
        self.G.add(Dense(dim*dim*depth, input_dim=100))
        self.G.add(BatchNormalization(momentum=0.9))
        self.G.add(LeakyReLU(alpha=0.2))
        self.G.add(Reshape((dim, dim, depth)))
        self.G.add(Dropout(dropout))

        # In: dim x dim x depth
        # Out: 2*dim x 2*dim x depth/2

        self.G.add(UpSampling2D())
        self.G.add(Conv2DTranspose(int(depth/2), 5, padding='same'))
        self.G.add(BatchNormalization(momentum=0.9))
        self.G.add(LeakyReLU(alpha=0.2))
        self.G.add(UpSampling2D())

        self.G.add(Conv2DTranspose(int(depth/4), 5, padding='same'))
        self.G.add(BatchNormalization(momentum=0.9))
        self.G.add(LeakyReLU(alpha=0.2))

        self.G.add(Conv2DTranspose(int(depth/8), 5, padding='same'))
        self.G.add(BatchNormalization(momentum=0.9))
        self.G.add(LeakyReLU(alpha=0.2))

        self.G.add(Conv2DTranspose(int(depth/16), 5, padding='same'))
        self.G.add(BatchNormalization(momentum=0.9))
        #self.G.add(Activation('relu'))
        self.G.add(LeakyReLU(alpha=0.2))

        # Out: 28 x 28 x 1 grayscale image [0.0,1.0] per pix

        self.G.add(Conv2DTranspose(1, 5, padding='same'))
        self.G.add(Activation('tanh'))
        self.G.summary()
        return self.G

    def discriminator_model(self):
        if self.DM:
            return self.DM
        optimizer = Adam(lr=0.0002, beta_1=0.5)
        self.DM = Sequential()
        self.DM.add(self.discriminator())
        self.DM.compile(loss='binary_crossentropy', optimizer=optimizer,\
            metrics=['accuracy'])
        return self.DM

    def adversarial_model(self):
        if self.AM:
            return self.AM
        optimizer = Adam(lr=0.0001, beta_1=0.25)
        self.AM = Sequential()
        self.AM.add(self.generator())
        self.AM.add(self.discriminator())
        self.AM.compile(loss='binary_crossentropy', optimizer=optimizer,\
            metrics=['accuracy'])
        return self.AM



class SONG_DCGAN(object):

    def __init__(self):
        self.img_rows = 28
        self.img_cols = 28
        self.channel = 1

        exist = input('Was the data created before (Y/N)? ')
        if exist.lower() == 'n':
            name = input('Folder name of mid/midi files: ')
            self.x_train = createAll(name)
            print(self.x_train)
            np.savetxt('songs.csv', self.x_train, delimiter=',')
            self.x_train = self.x_train.reshape(-1, self.img_rows,\
                self.img_cols, 1).astype(np.float32)
            self.x_train /= 255
        else:
            self.x_train = np.genfromtxt('songs.csv', delimiter=',')
            self.x_train = self.x_train.reshape(-1, self.img_rows,\
                self.img_cols, 1).astype(np.float32)
            self.x_train /= 255
        self.DCGAN = DCGAN()
        self.discriminator =  self.DCGAN.discriminator_model()
        self.adversarial = self.DCGAN.adversarial_model()
        self.generator = self.DCGAN.generator()

    def train(self, train_steps=2000, batch_size=256, save_interval=0):
        noise_input = None
        if save_interval>0:
            noise_input = np.random.uniform(-1.0, 1.0, size=[16, 100])
        for i in range(train_steps):
            images_train = self.x_train[np.random.randint(0,
                self.x_train.shape[0], size=batch_size), :, :, :]
            noise = np.random.normal(0, np.random.uniform(0, 1), size=[batch_size, 100])
            images_fake = self.generator.predict(noise)
            x = np.concatenate((images_train, images_fake))
            y = np.ones([2*batch_size, 1])
            y[batch_size:, :] = 0
            d_loss = self.discriminator.train_on_batch(x, y)
            y = np.ones([batch_size, 1])
            noise = np.random.uniform(-1.0, 1.0, size=[batch_size, 100])
            a_loss = self.adversarial.train_on_batch(noise, y)

            if d_loss[1] < a_loss[1]:
                y = np.ones([batch_size, 1])
                self.discriminator.train_on_batch(images_train, y)
            log_mesg = "%d: [D loss: %f, acc: %f]" % (i, d_loss[0], d_loss[1])
            log_mesg = "%s  [A loss: %f, acc: %f]" % (log_mesg, a_loss[0], a_loss[1])
            print(log_mesg)

            if save_interval>0 and i > 500:
                if (i+1)%save_interval==0:
                    self.plot_images(save2file=True, samples=noise_input.shape[0],\
                        noise=noise_input, step=(i+1))



    def plot_images(self, save2file=False, fake=True, samples=16, noise=None, step=0):

        filename = 'mnist.png'
        if fake:
            if noise is None:
                noise = np.random.uniform(-1.0, 1.0, size=[samples, 100])
            else:
                filename = "mnist_%d.png" % step
            images = self.generator.predict(noise)
        else:
            i = np.random.randint(0, self.x_train.shape[0], samples)
            images = self.x_train[i, :, :, :]
        
        #added
        for image in enumerate(images):
            filename = "mnist_%d_%d.png"%(step, image[0])
            imoge = np.reshape(image[1], [self.img_rows, self.img_cols])
            imoge += 1
            imoge *= 255
            imoge = imoge.astype(np.uint8)
            with np.nditer(imoge, op_flags=['readwrite']) as it:
                for x in it:
                    if x > 220:
                        x[...] = 255
            cv2.imwrite(filename, imoge)

def createAll(midiPath):
    '''
    Read a folder of midi files. Encode each midi file to an image of 28x28 pixels. 
    Write each image to folder in the same directory as the midi folder
    '''
    imageFolder = str(midiPath).strip('/')+'Images'
    imagePath = os.path.join(os.path.abspath(midiPath), '..',imageFolder)
    print('saving images to ',imagePath)
    train = np.zeros((28,28),np.uint8)
    skipped = 0
    try:
        os.mkdir(imagePath)
    except:
        print(imagePath, 'could not be created')
    for midiFile in os.listdir(midiPath):
        iMade = createImage(str(os.path.join(midiPath,midiFile)),\
                          str(os.path.join(imagePath,midiFile.split('.')[0] + '.png')))
        if not iMade is None:
            train = np.append(train, iMade)
        else:
            skipped += 1
            print(skipped)
    return train

def process_midi(filename): 
    notes = []  
    times = []
    prev_note = None    
    index = 0

    midi_data = pretty_midi.PrettyMIDI(filename)
    instruments = []
    for instr in midi_data.instruments:
        if(not instr.is_drum):
            instruments.append(instr)
            break

    instrument_notes = []
    for instrument in instruments:
        instrument_notes.append([])

    # obtain notes for each instrument
    for index in range(len(instruments)):
        instrument_notes[index] = instruments[index].notes  
    
    # sort list
    total_notes = []
    for n in instrument_notes:
        total_notes += n
    
    total_notes.sort(key=lambda Note: Note.start)

    #still skipping a lot try to fix
    if len(total_notes) == 0 or total_notes[-1].end < 10:
        return

    for note in total_notes:
        if note.pitch > 0:
            if(prev_note == None):
                notes.append(note)
                prev_note = note
            elif(note.start == prev_note.start):
                if(note.pitch > prev_note.pitch):
                    notes[len(notes)-1] = note
                    prev_note = note
            elif(note.start >= prev_note.end):
                notes.append(note)
                prev_note = note

    first_note = notes[0]
    start_time = first_note.start
    for note in notes:
        note.start -= start_time
        note.end -= start_time

    # separate pitches
    pitches = []
    for note in notes:
        pitches.append(note.pitch)

    # separate times
    time_step = 100000
    #prev_note = None
    for note in notes:
        diff = note.end-note.start
        if(diff < time_step and diff > .1):
            time_step = diff

    final_note = notes[len(notes)-1]
    total_time_steps = final_note.end / time_step

    times = []
    prev_note = None
    index = 0
    for note in notes:
        if(prev_note != None):
            diff = int((note.start-prev_note.end) // time_step)
            if(diff > 0):
                times.append((diff, 255))
        diff = int((note.end-note.start) // time_step)
        if diff > 0:
            times.append((diff, note.pitch*2))
        
        prev_note = note
    
    return times

def createImage(midiName, imageName):
    '''
    Create an image from a midi file by determing assigning a pixel at a given time interval based on the highest pitch at that time.
    Arguments:
        imageName (str): the name of the image to save
    '''
    
    
    
    #midiImage = np.zeros((28,28,1), np.uint8)
    midiImage = np.zeros(784, np.uint8)

    times = process_midi(midiName)
    if times is None:
        return

    index = 0
    while(index < 784):
        for time in times:
            if len(times) <= 1:
                return
            if (index+time[0] <= 784):
                midiImage[index:index+time[0]] = time[1]
                index+=time[0]
            else:
                midiImage[index:784] = time[1]
                index = 784
    cv2.imwrite(imageName, np.reshape(midiImage,(28,28)))
    return np.reshape(midiImage,(28,28))

if __name__ == '__main__':

    song_dcgan = SONG_DCGAN()
    timer = ElapsedTimer()
    song_dcgan.train(train_steps=10000, batch_size=256, save_interval=250)
    timer.elapsed_time()
    song_dcgan.plot_images(fake=True)
    song_dcgan.plot_images(fake=False, save2file=True)
