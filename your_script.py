import subprocess

subprocess.run(["ffmpeg", "-i", "output.mp4", "-c:v", "libx264", "-pix_fmt", "yuv420p", "output_fixed.mp4"])
from IPython.display import Video
Video("output_fixed.mp4", embed=True)