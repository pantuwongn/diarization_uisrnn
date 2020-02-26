import pyaudio
import wave

class AudioPlayer:
    '''
    This class is implemented as a player for audio wav file
    '''

    def __init__(self, wav):
        ''' initial for audio player object
        '''
        # pyaudio object
        self.p = pyaudio.PyAudio()

        # frame position
        self.pos = 0

        # audio stream data
        self.stream = None

        # call a method to open wav file
        self._open(wav)

    def callback(self, frame_count, time_info, status):
        ''' This is a callback function for pyaudio to 
                open wav
        '''
        data = self.wf.readframes(frame_count)
        self.pos += frame_count
        return (data, pyaudio.paContinue)

    def _open(self, wav):
        ''' A method to open wav file
        '''
        # open wav file
        self.wf = wave.open(wav, 'rb')
        # use pyaudio object to read wav file to get audio stream
        self.stream = self.p.open(format=self.p.get_format_from_width(self.wf.getsampwidth()),
                channels = self.wf.getnchannels(),
                rate = self.wf.getframerate(),
                output=True,
                stream_callback=self.callback)

        # pause (not play steam immediately after open)
        self.pause()

    def play(self):
        ''' A method to play file from current pos
        '''
        self.stream.start_stream()

    def pause(self):
        ''' A method to pause the playing file
        '''
        self.stream.stop_stream()

    def seek(self, seconds = 0.0):
        ''' A method to move play head to specific second
        '''
        sec = seconds * self.wf.getframerate()
        self.pos = int(sec)
        self.wf.setpos(int(sec))

    def time(self):
        ''' A method to get time of the current position of play head
        '''
        return float(self.pos)/self.wf.getframerate()

    def playing(self):
        ''' A mthod that return a flag to state that the player is playing
        '''
        return self.stream.is_active()

    def close(self):
        ''' A method to close stream and file
        '''
        self.stream.close()
        self.wf.close()
        self.p.terminate()

