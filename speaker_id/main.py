from pyannote.audio import Pipeline
import numpy as np
import soundfile as sf
import librosa





# # dump the diarization output to disk using RTTM format
# with open("audio.rttm", "w") as rttm:
#     diarization.write_rttm(rttm)
    
    
def collect_tstamps(diarization):
   timestamps = []
   for turn,_, speaker in diarization.itertracks(yield_label=True):
      timestamps = (f"speaker_{speaker}",turn.start, turn.end) 
   return timestamps
   # with open("audio.rttm") as file:
      
      # speaker_tstamp = {}
      # info = file.read()
      # info = info.split('\n')
      # for i,line in enumerate(info):
      #    line_spl = line.split(' ')
      #    speaker_n = line_spl[-3]
      #    t_stamp = line_spl[3]
         
      #    if i==len(info)-2:
      #       speaker_tstamp[i] = (speaker_n,t_stamp, librosa.get_duration(filename="audio.wav"))
      #       return speaker_tstamp
      #    speaker_tstamp[i] = (speaker_n,t_stamp, info[i+1].split(' ')[3])


def produce_audio_for_sep_speakers(timestamps):
   
   # timestamps = collect_tstamps('audio.rttm')
   # Create the two speakers from the array
   y, sr = librosa.load('audio.wav', sr=None)

   speaker1_speech = np.array([])
   speaker2_speech = np.array([])
   for i in range(len(timestamps)):
      start = int(float(timestamps[1]) * sr)
      end = int(float(timestamps[2]) * sr)
      if timestamps[0] == 'speaker_0':
         speaker1_speech = np.concatenate((speaker1_speech, y[start:end]), axis=None)
      else:
         speaker2_speech = np.concatenate((speaker2_speech, y[start:end]), axis=None)

   sf.write('results/speaker1.wav', speaker1_speech, sr)
   sf.write('results/speaker2.wav', speaker2_speech, sr)

def run(input_path):
   pipeline = Pipeline.from_pretrained("pyannote/speaker-diarization@2.1",
                                    use_auth_token="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")
   diarization = pipeline("audio.wav", num_speakers=2, max_speakers=2)
   timestamps = collect_tstamps(diarization)
   produce_audio_for_sep_speakers(timestamps)
   
   
if __name__ == '__main__':
   run('audio.wav')