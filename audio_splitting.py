from pydub import AudioSegment
from pydub.utils import make_chunks

myaudio = AudioSegment.from_file("/Users/adees/Code/neural_granular_synthesis/datasets/ESC-50_SeaWaves/audio/samples/5secs/small_train/1-39901-A-11.wav" , "wav") 
chunk_length_ms = 1000 # pydub calculates in millisec
chunks = make_chunks(myaudio, chunk_length_ms) #Make chunks of one sec

import os

directory = "/Users/adees/Code/neural_granular_synthesis/datasets/ESC-50_SeaWaves/audio/samples/5secs/small_train/"
    
for file in os.listdir(directory):
    filename = os.fsdecode(file)
    if filename.endswith(".wav"): 
        # print(filename)
        for i, chunk in enumerate(chunks):
            chunk_name = f"/Users/adees/Code/neural_granular_synthesis/datasets/ESC-50_SeaWaves/audio/samples/5secs/small_split/{str(filename)[:-4]}_chunk{i}.wav"
            chunk.export(chunk_name, format="wav")
        print("Chunked: ", os.path.join(directory, str(filename)))
        continue
    else:
        continue