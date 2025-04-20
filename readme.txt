## To login using ssh
ssh pi@pi.local
## then enter pw :  1234

## virtual keyboard
wvkbd-mobintl

## check camera 
rpicam-still -o ~/Desktop/image.jpg


pip install --pre --extra-index-url https://snapshots.linaro.org/ldcg/python-cache/tensorflow/ tensorflow-aarch64

sudo apt update
sudo apt install -y python3-libcamera python3-libcamera-apps python3-opencv python3-picamera2 python3-pyqt5 python3-prctl libatlas-base-dev libopenjp2-7 libtiff5

pip install -r requirements.txt
