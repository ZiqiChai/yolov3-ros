### how to run the server.py and client.py
clone this repo to your `catkin_ws/src` folder:
```
git clone https://github.com/ZiqiChai/yolov3-ros.git
```

clone the submodule `PyTorch_YOLOv3_Py27`:
```
cd yolov3-ros
git submodule init
git submodule update
```

create the Anaconda virtual environment:
```
cd PyTorch_YOLOv3_Py27/anaconda_envs
conda env create -f rosYolov3Py27.yaml
```

activate your Anaconda virtual environment:
```
conda activate rosYolov3Py27
```

firstly, under the rosYolov3Py27 environment, 
run server under `yolov3-ros/src` path as:
```
python server.py
```

secondly, open a new terminal,
run your camera and adjust topic name in `client.py`.

finally, open a new terminal,
run client under `yolov3-ros/src` path as:
```
python client.py
```

### how to test the detect.py script

activate the rosYolov3Py27 environment and run:
```
python  detect.py --image_folder ../PyTorch_YOLOv3_Py27/assets --model_def ../PyTorch_YOLOv3_Py27/config/yolov3-custom.cfg --weights_path ../PyTorch_YOLOv3_Py27/checkpoints/yolov3_ckpt_399.pth --class_path ../PyTorch_YOLOv3_Py27/data/custom/classes.names
```
