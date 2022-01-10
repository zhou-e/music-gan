import pretty_midi
import numpy as np
import cv2

def decode(fileName, outputName):
    img = cv2.imread(fileName, cv2.IMREAD_GRAYSCALE)

    # image to duration and pitch array
    times = []
    prev_pixel = None
    curr_pixel = None
    pixel_count = 1
    for row in img:
        for pixel in row:
            if(prev_pixel == None):
                prev_pixel = pixel
            elif(pixel == prev_pixel):
                pixel_count += 1
            else:
                times.append((pixel_count, prev_pixel))
                pixel_count = 1
                prev_pixel = pixel
    times.append((pixel_count, prev_pixel)) 

    #print(times)

    output_music = pretty_midi.PrettyMIDI()
    music_program = pretty_midi.instrument_name_to_program('Acoustic Grand Piano')
    instr = pretty_midi.Instrument(program=music_program)

    # convert to midi notes
    time_step = 0.2
    current_time = 0
    delta = 0.01
    for time in times:
        if(time[1] == 255):
            current_time += time_step*time[0]
        else:
            note = pretty_midi.Note(velocity=64, pitch=int(time[1]/2), start=current_time+delta, end = current_time+delta+time_step*time[0])
            instr.notes.append(note)
            current_time += delta+time_step*time[0]
    
    output_music.instruments.append(instr)
    output_music.write(outputName)
    
    #for instrument in output_music.instruments:
        #for note in instrument.notes:
            #print(note)

def set_to_scale(filename):
    # presets
    c_maj = ['C', 'D', 'E', 'F', 'G', 'A', 'B']
    c_maj_pentatonic = ['C', 'D', 'E', 'G', 'A']
    c_min_blues = ['C', 'Eb', 'F', 'F#', 'G', 'Bb']
    c_min_harmonic = ['C', 'D', 'Eb', 'F', 'G', 'Ab', 'B']

    # set scale
    char_scale = c_min_blues

    # convert note names to numbers
    for i in range(len(char_scale)):
        char_scale[i] += '-1'
    
    int_scale = []
    for note_name in char_scale:
        int_scale.append(pretty_midi.note_name_to_number(note_name))
    
    full_scale = []
    NUM_OCTAVES = 11
    MAX_NOTE = 127
    for i in range(NUM_OCTAVES):    
        for note in int_scale:
            new_note = note + 12*i
            if(new_note <= MAX_NOTE):
                full_scale.append(note + 12*i)

    midi_data = pretty_midi.PrettyMIDI(filename)
    notes = midi_data.instruments[0].notes
    for note in notes:
        if(note.pitch < full_scale[0]):
            note.pitch += 12
        while(note.pitch not in full_scale):
            note.pitch -= 1

    output = pretty_midi.PrettyMIDI()
    instr_program = pretty_midi.instrument_name_to_program("Acoustic Grand Piano")
    instr = pretty_midi.Instrument(program=instr_program)
    instr.notes = notes
    output.instruments.append(instr)
    output.write(filename[:-4] + "_scale.mid")

#x = 'x.png'
#y = 'x.mid'
#decode(x, y)
start = 750
steps = 10000
saveInt = 250
for x in range(int(start/saveInt), int(steps/saveInt)+1):
        current = x*saveInt
        for y in range(16):
                decode('mnist_%d_%d.png'%(current, y), \
                       'music_%d_%d.mid'%(current, y))
                set_to_scale('music_%d_%d.mid'%(current, y))
