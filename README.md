# Audio-Video-AI-Assistant

You need an `OPENAI_API_KEY` and a `GOOGLE_API_KEY` to run this code _(credit card too hehe)_. Store them in a `.env` file in the root directory of the project, or set them as environment variables.


If you are running the code on Apple M1/2/3, run the following command in the terminal:

```
$ brew install portaudio
```

Create a virtual environment, update pip, and install the required packages:

```
$ conda create -n "YOURENV" python=3.9
$ conda activate "YOURENV"
$ pip install -r requirements.txt
```

Run the assistant:

```
$ python assistant.py
```
