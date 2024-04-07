docker build -f Dockerfile . -t jis_llava:r35.3.1

docker run -it --rm --runtime nvidia --network host -v /home/seeed/JIS/jetson-containers/data:/data  jis_llava:r35.3.1 python3 -m run_inference
docker run -it --rm --runtime nvidia --network host -v /home/seeed/JIS/jetson-containers/data:/data  jis_llava:r35.3.1 python3 -m run_inference

docker run -it --rm --runtime nvidia --network host 
    -v /etc/enctune.conf:/etc/enctune.conf
    -v /etc/nv_tegra_release:/etc/nv_tegra_release
    -v /home/seeed/jetson-containers/data:/data
    -v /home/seeed/llava_inference/main.py:/opt/llava_inference/main.py
    --device /dev/snd --device /dev/bus/usb
    llava_jic:v0