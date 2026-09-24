import gdown

# 1. Download the fine-tuned weights FIRST — before importing the package
url = 'https://drive.google.com/uc?id=1-8TyT7MkAS7LLsRTbO03tDsuuoBM6q1D'
gdown.download(url, 'model_ft.pth', quiet=False)

# 2. Only import the package after the weights file exists
from car_crash_detection import CrashUtils

# 3. Run detection on your video
inputPath = "crash1.mp4"
outputPath = "crash1_output.mp4"
seq = 16
skip = 1

CrashUtils.crashDetection(inputPath, seq, skip, outputPath, showInfo=False, thresholding=0.85)

print("Done. Check crash1_output.mp4 in this folder.")