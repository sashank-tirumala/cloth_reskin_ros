
## Setup
* Download reskin_sensor repo: https://github.com/thomasweng15/reskin_sensor
    * On yertle, reskin_sensor is installed at `~/tactile_ws/external/reskin_sensor`
    * Install with `pip install -e .`
* Set up QT Py board for the Reskin
    * Plug in the board and verify that it is on `/dev/ttyACM0`
        * Add user to `dialout` group (see [this link](https://arduino.stackexchange.com/questions/31618/cant-connect-to-serial-port/31619))
        * May need to run `sudo chmod a+rw /dev/ttyACM0`
        * May need `sudo arduino` if all else fails
        * May need to write udev rules
            * https://unix.stackexchange.com/questions/25258/ttyusb0-permission-changes-after-restart see ACM answer
            * https://askubuntu.com/questions/82470/what-is-the-correct-way-to-restart-udev
    * Install Arduino and QT Py Board https://learn.adafruit.com/adafruit-qt-py/arduino-ide-setup
    * Go to Tools -> Boards Manager... and install Adafruit SAMD Boards (v1.7.9)
    * Install SparkFun I2C Mux Arduino Library
    * Get the MLX90393 driver
        * `reskin_sensor/arduino/arduino/arduino-MLX90393 && git submodule --init update`
        * Copy `arduino-MLX90393` to arduino installation on your system `arduino/libraries`
    * Upload binary code: 
        * Upload from terminal:
            * `bash bash_scripts/upload_reskin.bash`
            * Or without the bash script:
                * `arduino --upload ~/tactile_ws/external/reskin_sensor/arduino/5X_burst_stream/5X_burst_stream.ino --port /dev/ttyACM0`
                    * Required before uploading the binary version
                    * May have to try multiple times
                * `arduino --upload ~/tactile_ws/external/reskin_sensor/arduino/5X_binary_burst_stream/5X_binary_burst_stream.ino --port /dev/ttyACM0`
        * Alternatively, upload the QT Py with code from `reskin_sensor/arduino/5X_burst_stream/5X_burst_stream.ino`
            * Set Board: Tools->Board->Adafruit QT Py
            * Set Port to whichever port the QT Py is plugged into, (e.g. /dev/ttyACM0). Be careful with this step, if this is incorrect you could fry the QT Py (see below)!
            * Press Upload
        * Verify using the Serial Monitor that the data coming in is correct (last value in each row is temperature in Celsius, it should be around 20-40). Use 115200 baud rate
        * Upload the QT Py with `reskin_sensor/arduino/5X_burst_stream/5X_binary_burst_stream.ino`
            * This does not work unless you upload the non-binary version first, we are not sure why
        * Other notes
            * QT Py can be reset (?) with a long press followed by a short press for bootloading, an LED will appear that will be red then green. Then you can bootload. 
            * When uploading code, connect either the QT Py or the feather M0 to the computer at a time, not both. Sometimes you or the arduino IDE gets the port wrong, and you will have to reset the QT Py.
* Set up the Feather M0 board for the Delta Gripper
    * Plug in the board and verify that it is on `/dev/ttyACM1`
    * Install dependencies to system arduino libraries
        * nanopb: `https://jpa.kapsi.fi/nanopb/download/` (v0.4.5)
        * Adafruit Motorshield V2: https://learn.adafruit.com/adafruit-motor-shield-v2-for-arduino/install-software 
        * Adafruit ADS1X15: git@github.com:adafruit/Adafruit_ADS1X15.git
    * Upload the code to the Feather M0: `arduino --upload ~/tactile_ws/src/tactile_cloth/scripts/linear_delta_test/delta_array_6motors/delta_array_6motors.ino --port /dev/ttyACM1`. Try several times if it fails.
        * Alternatively open in the GUI: `scripts/linear_delta_test/delta_array_6motors/delta_array_6motors.ino`
        * Set Board to Feather M0
        * Set Port to `/dev/ttyACM1`
        * Press Upload
    * Verify successful upload using serial monitor - use 57600 baud rate. However it will only print if there is a command, and you can't run serial monitor simultaneously with ros, so it may be easier to just run the ROS code and see if it moves.

## Execution

### Run delta + reskin manually moving the Franka
1. Set Franka robot in gravity compensation mode
2. `roslaunch tactilecloth delta_reskin.launch`
    * See arguments for no rubbing, vertical rubbing, horizontal rubbing (not implemented yet)
3. Press B on joystick to execute pinch

### Run delta + reskin with Franka for data collection and experiments
See singlearm_ros repo for detailed instructions on setting up and running the robot
1. Start roscore on primary pc `roscore`
2. On franka control pc (iam-soul)
    * `source ~/ws1/devel/setup.bash`
    * `roslaunch panda_moveit_config panda_control_moveit_rviz.launch robot_ip:=172.16.0.2 launch_rviz:=false load_gripper:=false`
3. Launch roslaunch files on primary pc
    * To run data collection: `roslaunch singlearm_ros grasp_policy.launch use_delta:=true exp_name:=dbg cloth_type:=0cloth run_type:=collect_data` 
    * To run experiments: `roslaunch singlearm_ros grasp_policy.launch use_delta:=true exp_name:=dbg cloth_type:=0cloth run_type:=run_experiment` 
4. Move to reset position
    * WARNING: make sure collision geometries of the robot and the environment are set!
    * `rostopic pub -1 /reset std_msgs/String "data: "`
        * Confirm movement in RViz and then enter y to run the action
5. Debug grasping and moving (optional, not sure if it works for data_collect)
    * Example commands
        * `rostopic pub -1 /run_experiment std_msgs/String "data: move_-0.01_-0.01"`
        * `rostopic pub -1 /run_experiment std_msgs/String "data: move+pinch_-0.01_-0.01"`
7. Collect data 
    * `rostopic pub -1 /collect_data std_msgs/String "data: start"`
        * Launches automatic data collection where the gripper moves in a snake pattern. 
        * Reset robot position: `rostopic pub -1 /reset std_msgs/String "data: '1'"`
        * Manually set height `rostopic pub -1 /run_experiment std_msgs/String "data: move_-0.01_0.0"` (TODO check this or use the debug rostopic pub below)
        * Set initial position of arm to closest corner of cloth (see rostopic pub below)
6. Run experiments
    * When switching cloths, `rosparam set /app/cloth_type 1cloth`
    * Manual mode:
        * `rosparam set robot_app/policy manual` (this is the default)
        * Send pinch and move command`rostopic pub -1 /run_experiment std_msgs/String "data: pinch+move_-0.01_-0.01"`
            * TODO check that classifier stops running after pinch
    * Random (send number of layers to grasp): 
        * `rosparam set robot_app/policy random`
        * Send number of cloths `rostopic pub -1 /run_experiment std_msgs/String "data: '1'"`
    * Policy same as random
        * `rosparam set robot_app/policy closedloop`
        * Send number of cloths `rostopic pub -1 /run_experiment std_msgs/String "data: '1'"`
7. Run paper experiments
    * Modify rand exp cfg for all the experiments you want to include
    * `roslaunch singlearm_ros grasp_policy.launch exp_name:=dbg use_delta:=true cloth_type:=1cloth grasp_type:=norub run_type:=run_random_exp save_dir:=/media/ExtraDrive3/fabric_touch/paper_experiments`
    * `rostopic pub -1 /run_experiment std_msgs/String "data: ''"`
    * `roslaunch singlearm_ros grasp_policy.launch exp_name:=simplepolicy_5class_knn_1layer use_delta:=true cloth_type:=1cloth grasp_type:=norub run_type:=run_random_exp save_dir:=/media/ExtraDrive4/fabric_touch/paper_experiments classifier_type:=all rand_exp_dir:=2022-02-26-19-37-27_simplepolicy_5class_knn_1layer`
8. Run real world data collection
  * Set openloop_noise = 0.001 in config.yaml
  * `roslaunch singlearm_ros grasp_policy.launch exp_name:=finetune_noisymanual use_delta:=true cloth_type:=3cloth grasp_type:=norub run_type:=run_random_exp save_dir:=/media/ExtraDrive4/fabric_touch/datasets classifier_type:=tactile rand_exp_cfg:=/home/tweng/tactile_ws/src/bimanual_folding/singlearm_ros/config/rand_exp_cfgs/finetune_datacollection.yaml use_dslr:=false`

9. More detailed instructions for running paper experiments
  * Set up the DLSR
    * Check that there is a full battery
    * Focus the lens manually
      * Unplug camera from computer
      * Pull the "LV" switch to go to live mode
      * Focus the camera using the lens
      * Plug camera back in, it will automatically exit live mode
    * Unmount the camera from the filesystem
  * Set up ROS
    * Run roscore and the panda launch file on iam-soul as in the instructions above. Make sure robot is unlocked
  * Check that reskin is running
    * After launching grasp_policy.launch, try running rostopic echo reskin, if nothing comes up, reskin is not connected
    * Shutdown roscore and unplug and replug reskin, also unplug delta. Make sure to plug in reskin first so it ends up on /dev/ttyACM0
    * Run bash_scripts/upload_reskin.bash
      * This will automatically kill ROS if it is running in case you forgot to kill it
      * After it successfully uploads the 5x_burst_stream script, use the Arduino Serial Monitor to check that the values are correct
    * Run ROS again and verify that you are getting values on rostopic echo reskin
  * If webcam dies during data collection
    * You can tell it dies if you see the node die in the terminal, or if the RViz image freezes
    * Unplug and replug the webcam
    * Open another terminal and run rosrun tactilecloth webcam.py and continue running experiments
  * If you get weird predictions from the image or tactile classifiers
    * Check that webcam or reskin are running
  * commonly used flags to grasp_policy.launch
    * `rand_exp_cfg`: change which experiment config you use
    * `rand_exp_dir`: by default the script saves to a new dir with a new timestamp, if you want to save into an old dir to continue and experiment pass the folder name to this arg
  * Common pitfalls
    * When running the image classifier you have to wait for the classifier to load. I set it so that you can start randomly sampling non-image-based methods before the image classifier is ready, but it takes about 2 minutes for the weights to load, due to Python 3 weights being converted to Python 2. Even though the weights are already converted offline to Python 2, loading them seems to take this long and I am not sure why. 
    * If you have static and you "shock" the reskin, there is a high chance it will start producing invalid values. The solution is to unplug/replug the reskin and reupload the code as described above. 
    * To run the image classifier you need to run rosparam set /robot_app/imgclf_ready true, it gets set to false by something in the code constantly, so before you run a trial you need to set this rosparam every time. We should fix this bug asap


gif from mp4
```
ffmpeg -y -i %s -vf "fps=10,scale=320:-1:flags=lanczos,split[s0][s1];[s0]palettegen[p];[s1][p]paletteuse" -loop 0 %s.gif
```

rostopic pub for moving initial position of arm
```
rostopic pub -1 /move_and_grasp/goal singlearm_ros/MoveAndGraspActionGoal "header:
  stamp:
    secs: 0
    nsecs: 0
  frame_id: ''
goal_id:id: ''
  stamp:
    secs: 0
    nsecs: 0
  id: ''s: 0
goal: ''
  command: {data: 'move'}
  height_diff: {data: -100.0}
  slide_diff: {data: -0.05}0}
  x_diff: {data: 0.0}-0.05}
  height_absolute: {data: 0.275}"
```


```
rostopic pub -1 /move_and_grasp/goal singlearm_ros/MoveAndGraspActionGoal "header:
  seq: 0
  stamp:
    secs: 0
    nsecs: 0
  frame_id: ''
goal_id:
  stamp:
    secs: 0
    nsecs: 0
  id: ''
goal:
  command:
    data: 'move_x'
  height_diff:
    data: 0.0
  slide_diff:
    data: 0.0
  x_diff:
    data: 0.07"
```

<!-- 5. Debug grasp execution by publishing a single command, height, and slide difference: `rostopic pub -1 /move_and_grasp std_msgs/String "data: pinchlift_0.01_-0.04"`
    * Negative value for slide moves into the cloth, positive height moves upward. 
        * Keep slide diff -0.04 or some other fixed value
    * Command can be: [move, pinch, or pinchlift]. 
        * move does not execute a pinch. 
        * pinch executes the pinch but release immediately and does not move.
        * pinchlift executes the pinch, then either lifts (not implemented yet) or releases and moves back to the starting position.  -->

### Run just Pinching and Classifier
1. `roslaunch tactilecloth delta_reskin.launch collection_mode:=joy grasp_type:=norub cloth_type:=0cloth exp_name:=dbg save_dir:=/media/tweng/ExtraDrive2/fabric_touch/experiments`
2. publish debug command
```
rostopic pub -1 /move_fingers/goal singlearm_ros/MoveFingersActionGoal "header:
  seq: 0
  stamp:
    secs: 0
    nsecs: 0
  frame_id: ''
goal_id:
  stamp:
    secs: 0
    nsecs: 0
  id: ''
goal:
  command:
    data: 'dbg_close'"
```

### Running just the Classifier
1. Need to install plotjuggler to visualize all the data in real time `sudo apt install ros-melodic-plotjuggler-ros`
2. `roslaunch classifier.launch`
    * Need to send the directory where you saved the classifier and the scaler as an argument `clf_dir_name`
3. Publish to `/classifier_commands` topic `start` or `stop`.
    * Can use `rostopic pub /classifier_commands std_msgs/String "start"` from the command line.

### Running the Image Classifier
On its own:
1. `initconda()`
2. `conda activate torch_env`
3. `python model_utils/model_load_util.py --weight_path weight.pth` to get `weight.txt` in same dir
4. `conda deactivate` or make sure you are in a non-conda shell
4. roslaunch with image classifier args
    `roslaunch singlearm_ros grasp_policy.launch use_delta:=true exp_name:=dbg cloth_type:=1cloth run_type:=run_experiment classifier_type:=image clf_dir_name:=/home/tweng/tactile_ws/src/tactile_cloth/config/reskin_img_classifier_00.txt`

## Other notes
* Always leave Gripper in open position, Robot in white mode when leaving the room to prevent hardware damage

### Hardware Linear DeltaZ Actuation:
WIRING: (see last page for pin numbering) 
* 1 - Orange – Feedback Potentiometer negative reference rail 
* 2 - Purple  – Feedback Potentiometer wiper 
* 3 - Red  – Motor V+ (6V or 12V) 
* 4 - Black     – Motor V- (Ground) 
* 5 - Yellow   – Feedback Potentiometer positive reference rail 

Pins 1 & 5 determine the range, let us say 1 is given 3V and 5 is given 12V by us. Then pin 2 is an output pin from the actuator that varies between 3V and 12V depending on the extension of the actuator. (So 7.5V basically will be halfway extended etc). 4 should be ground. 3 should be varied to control the position of the actuator. Ideally a specific voltage will be a specific position, however since we have a feedback signal we can use a PID Controller to determine the voltage to send to pin 3. 
* Note: Why dont we use the L12I --> It comes with a built in motor-controller industrial standard and should work well for our purposes. 
* Note: https://arxiv.org/pdf/2008.03596.pdf
