from pydub import AudioSegment
from pydub.utils import make_chunks
import os

directory = "/home/CAMPUS/d22127229/code/github/neuralGranularSynthesis/data/EpidemicSound/water_wave_train/"

j = 0 
for file in os.listdir(directory):
    filename = os.fsdecode(file)
    if filename.endswith(".wav"):
        myaudio = AudioSegment.from_file(directory+filename , "wav") 
        chunk_length_ms = 5000 # pydub calculates in millisec
        chunks = make_chunks(myaudio, chunk_length_ms) #Make chunks of one sec
        # print(filename)
        for i, chunk in enumerate(chunks):
            chunk_name = f"/home/CAMPUS/d22127229/code/github/neuralGranularSynthesis/data/FreeSound/sea_waves/5_sec_chunked_train/{str(filename)[3:-21]}_chunk{i}.wav"
            chunk.export(chunk_name, format="wav")
            j+=1
        print("Chunked: ", os.path.join(directory, str(filename)))
        continue
    if(j>1800):
        print("Reached max number of files, stopping")
        print(img)
    else:
        continue
