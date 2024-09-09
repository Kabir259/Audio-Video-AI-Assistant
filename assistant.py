import base64
from threading import Lock, Thread

import cv2
import requests
import openai
from cv2 import VideoCapture, imencode
from dotenv import load_dotenv
from langchain.prompts import ChatPromptTemplate, MessagesPlaceholder
from langchain.schema.messages import SystemMessage
from langchain_community.chat_message_histories import ChatMessageHistory
from langchain_core.output_parsers import StrOutputParser
from langchain_core.runnables.history import RunnableWithMessageHistory
from langchain_openai import ChatOpenAI
from langchain_google_genai import ChatGoogleGenerativeAI
from pyaudio import PyAudio, paInt16
from speech_recognition import Microphone, Recognizer, UnknownValueError # speech_recognition is a module that provides speech recognition capabilities

load_dotenv() # Load environment variables from the .env file

########################################################################################################
########################################################################################################
########################################################################################################

# Define a class for the webcam stream
class WebcamStream:
    def __init__(self):
        self.stream = VideoCapture(index=0)  # Initialize the video capture
        _, self.frame = self.stream.read()  # Read the first frame (returns (success, frame))
        self.running = False  # Flag to indicate if the stream is running
        self.lock = Lock()  # Lock to synchronize access to the frame
        
        ''' if multiple threads or processes need to access a critical section of code or a shared variable, 
        they can acquire the lock using self.lock.acquire() before accessing the resource, and release it using 
        self.lock.release() when they are done. This ensures that only one thread can execute the critical section at a time, 
        preventing race conditions and maintaining data consistency.

        It's important to note that the Lock object is specific to the Python threading module, which provides 
        high-level threading support. If you are working with multiprocessing or asynchronous programming, there are other 
        synchronization primitives available, such as multiprocessing.Lock'''

    def start(self) -> 'WebcamStream':
        """
        Start the webcam stream if it is not already running.
        Returns:
            WebcamStream: The WebcamStream instance.
        """
        if self.running:
            return self

    def update(self): # not using this
        while self.running:
            _, frame = self.stream.read()  # Read the current frame
            self.lock.acquire()  # Acquire the lock to update the frame
            '''The line self.lock.acquire() is calling the acquire() method on the self.lock object. 
            This method is used to acquire a lock, which is a synchronization mechanism that 
            allows only one thread to access a shared resource at a time'''
            
            self.frame = frame
            self.lock.release()  # Release the lock
            

    def read(self, encode=False):
        self.lock.acquire()  # Acquire the lock to read the frame
        frame = self.frame.copy()
        self.lock.release()  # Release the lock

        if encode:
            _, buffer = imencode(".jpeg", frame)  # Encode the frame as JPEG
            return base64.b64encode(buffer)  # Return the base64 encoded frame
        

        return frame

    def stop(self):
        self.running = False
        if self.thread.is_alive():
            self.thread.join()

    def __exit__(self, exc_type, exc_value, exc_traceback):
        self.stream.release()  # Release the video capture resources


########################################################################################################
########################################################################################################
########################################################################################################

# Define a class for the assistant
class Assistant:
    def __init__(self, model):
        """
        Initializes the Assistant object.

        Parameters:
        - model (str): The filepath of the model to be used for inference.

        Returns:
        None
        """
        self.chain = self._create_inference_chain(model) # defined below 
        

    def answer(self, prompt, image):
        if not prompt:
            return

        print("Prompt:", prompt)

        # Invoke the inference chain with the prompt and image
        response = self.chain.invoke(
            {"prompt": prompt, "image_base64": image.decode()},
            config={"configurable": {"session_id": "unused"}},
        ).strip()

        print("Response:", response)

        if response:
            self._tts(response)

    def _tts(self, response):
        '''method or attribute is intended for internal use within a class. 
        It is a way to signal to other developers that the method or attribute is not intended 
        to be accessed directly from outside the class.'''
        
        # Create an instance of the PyAudio class and open the audio stream
        player = PyAudio().open(format=paInt16, channels=1, rate=24000, output=True) # the player object is used to play the audio stream

        # Create a streaming response for text-to-speech conversion
        with openai.audio.speech.with_streaming_response.create(
            model="tts-1",
            voice="alloy",
            response_format="pcm",
            input=response,
        ) as stream:
            # Iterate over the response stream and write audio chunks to the player
            for chunk in stream.iter_bytes(chunk_size=1024):
                '''The purpose of iterating over a stream in chunks of 1024 bytes is to 
                process the audio data in smaller, manageable pieces. By reading and processing 
                the audio data in chunks, it allows for more efficient handling of the data and 
                reduces the memory usage.'''
                
                player.write(chunk)
                
                ''' The write() method writes audio data to the output stream.'''

    def _create_inference_chain(self, model): # used in constructor
        SYSTEM_PROMPT = """
        You are a witty assistant that will use the chat history and the image 
        provided by the user to answer its questions.

        Use few words on your answers. Go straight to the point. Do not use any
        emoticons or emojis. Do not ask the user any questions.

        Be friendly and helpful. Show some personality. Do not be too formal.
        """

        prompt_template = ChatPromptTemplate.from_messages(
            [
                SystemMessage(content=SYSTEM_PROMPT),
                MessagesPlaceholder(variable_name="chat_history"),
                (
                    "human",
                    [
                        {"type": "text", "text": "{prompt}"},
                        {
                            "type": "image_url",
                            "image_url": "data:image/jpeg;base64,{image_base64}",
                        },
                    ],
                ),
            ]
        )

        chain = prompt_template | model | StrOutputParser() # The | operator is used to chain the components together

        chat_message_history = ChatMessageHistory()   # Create an instance of the ChatMessageHistory class
        # the ChatMessageHistory class is used to store and manage the chat history and is defined as a subclass of MessageHistory
        '''in langchain, Message HIstory is different from ChatMessageHistory. MessageHistory is a base class for 
        storing and managing messages, while ChatMessageHistory is a subclass of MessageHistory that is 
        specifically designed for chat messages. other classes that inherit from MessageHistory include 
        EmailMessageHistory, SMSMessageHistory, and SocialMediaMessageHistory'''
        
        '''email, chat and smsm message histories are different from each other in terms of the type of messages they store and the
        methods they provide for managing those messages. For example, an email message history would store email messages, while a chat
        message history would store chat messages. Each type of message history would provide methods for adding, removing, and retrieving
        messages specific to that type of message.'''
        
        return RunnableWithMessageHistory(
            chain, # The chain is the inference chain that combines the model with the prompt template
            lambda _: chat_message_history, # The lambda function returns the chat message history. the use of _ is to indicate that the argument is not used
            input_messages_key="prompt",
            history_messages_key="chat_history",
        )
        
        '''returnable with message history is a class that combines a runnable with a message history.
        It is used to create a runnable that can access and modify a message history during execution.
        The runnable can read and write messages to the history, and the history can be accessed by other runnables or components.'''
        
########################################################################################################
########################################################################################################
########################################################################################################

# Create an instance of the WebcamStream class and start the stream
webcam_stream = WebcamStream().start() 

# Create an instance of the Assistant class with the specified model
model = ChatGoogleGenerativeAI(model="gemini-1.5-flash-latest")

# You can use OpenAI's GPT-4o model instead of Gemini Flash
# by uncommenting the following line:
# model = ChatOpenAI(model="gpt-4o")

assistant = Assistant(model)

# Define a function to call OpenAI's Whisper API
def recognize_whisper(audio_file_path, model="base", language="english"):
    url = "https://api.openai.com/v1/audio/transcriptions"
    headers = {
        "Authorization": f"Bearer YOUR_API_KEY",  # Replace YOUR_API_KEY with your actual OpenAI API key
        "Content-Type": "application/json"
    }
    data = {
        "model": model,
        "language": language
    }
    files = {
        "file": open(audio_file_path, "rb")
    }
    response = requests.post(url, headers=headers, data=data, files=files)
    return response.json().get("text", "")


def audio_callback(recognizer, audio):
    '''
    The audio_callback function is called when the recognizer listens to audio in the background.'''
    try:
        prompt = recognizer.recognize_whisper(audio, model="base", language="english")
        '''The recognize_whisper method is used to recognize speech from the audio input.'''
        assistant.answer(prompt, webcam_stream.read(encode=True))

    except UnknownValueError:
        print("There was an error processing the audio.")


# Create instances of the Recognizer and Microphone classes
recognizer = Recognizer() # The Recognizer class is used to recognize speech from audio input. it is defined under the speech_recognition module
microphone = Microphone() # The Microphone class is used to capture audio input from the microphone. it is defined under the speech_recognition module

# Adjust for ambient noise using the microphone
with microphone as source:
    recognizer.adjust_for_ambient_noise(source) 

# Start listening for audio in the background and call the audio_callback function
stop_listening = recognizer.listen_in_background(microphone, audio_callback) 
'''stop_listening stops reeiveing input from microphone when audio_callback is playing'''

# Continuously display the webcam stream until the user presses 'q' or 'Esc'
while True:
    cv2.imshow("webcam", webcam_stream.read())
    if cv2.waitKey(1) in [27, ord("q")]:
        break

# Stop the webcam stream and close the OpenCV windows
webcam_stream.stop()
cv2.destroyAllWindows()

# Stop listening for audio
stop_listening(wait_for_stop=False)
