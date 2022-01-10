import pretty_midi

# input: x.mid
# output: x_scale.mid
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

if __name__ == '__main__':
    filename = input("filename: ")
    set_to_scale(filename)
        