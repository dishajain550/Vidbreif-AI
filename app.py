from flask import Flask, render_template, request
import whisper
from transformers import pipeline
from moviepy.editor import VideoFileClip
import imageio_ffmpeg
import os

# -------------------------------
# Flask App Setup
# -------------------------------
app = Flask(__name__)
UPLOAD_FOLDER = "uploads"
app.config["UPLOAD_FOLDER"] = UPLOAD_FOLDER

# -------------------------------
# Force MoviePy to use local ffmpeg
# -------------------------------
os.environ["IMAGEIO_FFMPEG_EXE"] = os.path.join(os.getcwd(), "ffmpeg.exe")

# -------------------------------
# Load Deep Learning Models (once)
# -------------------------------
print("Loading Whisper model...")
whisper_model = whisper.load_model("base")

print("Loading summarization model...")
summarizer = pipeline(
    "summarization",
    model="facebook/bart-large-cnn"
)

# -------------------------------
# Routes
# -------------------------------
@app.route("/", methods=["GET", "POST"])
def index():
    transcript = ""
    summary = ""

    if request.method == "POST":
        video = request.files["video"]

        if video:
            video_path = os.path.join(app.config["UPLOAD_FOLDER"], video.filename)
            video.save(video_path)

            # -------------------------------
            # Extract audio from video
            # -------------------------------
            audio_path = video_path.rsplit(".", 1)[0] + ".wav"

            clip = VideoFileClip(video_path)
            clip.audio.write_audiofile(audio_path, logger=None)
            clip.close()

            # -------------------------------
            # Speech to Text (Whisper)
            # -------------------------------
            result = whisper_model.transcribe(audio_path)
            transcript = result["text"]

            # -------------------------------
            # Text Summarization
            # -------------------------------
            summarized = summarizer(
                transcript,
                max_length=150,
                min_length=50,
                do_sample=False
            )

            summary = summarized[0]["summary_text"]

    return render_template(
        "index.html",
        transcript=transcript,
        summary=summary
    )

# -------------------------------
# Run App
# -------------------------------
if __name__ == "__main__":
    os.makedirs(UPLOAD_FOLDER, exist_ok=True)
    app.run(debug=True)
