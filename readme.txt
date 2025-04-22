## To login using ssh
ssh user@raspberrypi.local
## then enter pw :  user

## virtual keyboard
wvkbd-mobintl

## check camera 
rpicam-still -o ~/Desktop/image.jpg

# fix steps

### Remove existing installations
sudo pip3 uninstall tensorflow tensorflow-aarch64 -y
sudo apt remove python3-numpy -y

sudo apt update

### Install compatible versions (critical for Pi 4B)
python3 -m pip install --upgrade --no-cache-dir numpy==1.23.5 tensorflow-aarch64==2.15.0 --extra-index-url https://snapshots.linaro.org/ldcg/python-cache/

pip install -r requirements.txt

picamera2
Pillow>=9.4.0
RPi.GPIO>=0.7.1