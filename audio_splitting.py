from pydub import AudioSegment
from pydub.utils import make_chunks
import os

directory = "/home/CAMPUS/d22127229/code/github/neuralGranularSynthesis/data/EpidemicSound/water_wave_eval_train/"
    
for file in os.listdir(directory):
    filename = os.fsdecode(file)
    if filename.endswith(".wav"):
        myaudio = AudioSegment.from_file(directory+filename , "wav") 
        chunk_length_ms = 1000 # pydub calculates in millisec
        chunks = make_chunks(myaudio, chunk_length_ms) #Make chunks of one sec
        # print(filename)
        for i, chunk in enumerate(chunks):
            chunk_name = f"/home/CAMPUS/d22127229/code/github/neuralGranularSynthesis/data/EpidemicSound/water_wave_chunked_eval_train/{str(filename)[:-4]}_chunk{i}.wav"
            chunk.export(chunk_name, format="wav")
        print("Chunked: ", os.path.join(directory, str(filename)))
        continue
    else:
        continue