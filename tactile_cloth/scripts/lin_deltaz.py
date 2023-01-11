#!/usr/bin/python
import rospy
# from std_msgs.msg import Float64MultiArray, MultiArrayDimension
from std_msgs.msg import String
from sensor_msgs.msg import Joy
# from inputs import get_gamepad
import numpy as np
from serial import Serial
from lin_deltaz_utils import move_to_xyz_pos
import numpy as np
import linear_actuator_pb2
from time import sleep
from singlearm_ros.msg import MoveFingersAction, MoveFingersActionResult
import actionlib

BTN_Y_INDEX = 3
BTN_B_INDEX = 1

class DeltaZ:
    def __init__(self):
        rospy.init_node('linear_delta_node')
        self.publish_deltaZ = False
        self.vertical_control = False

        self.trials_to_collect = 0
        self.pub = rospy.Publisher('/grasp_startstop', String, queue_size=10)
        self.joy_sub = rospy.Subscriber("/joy", Joy, self.joy_callback)
        self.delta_actionserver = actionlib.SimpleActionServer('move_fingers', MoveFingersAction, execute_cb=self.run_command, auto_start=False) 
        self.delta_actionserver.start()
        self.clf_pub = rospy.Publisher('/classifier_commands', String, queue_size=10)
        self.record_pub = rospy.Publisher('/record_trial', String, queue_size=10)
        self.arduino = Serial('/dev/ttyACM1', 57600)
        self.delta_message = linear_actuator_pb2.lin_actuator()
        self.delta_message.id = 1
        self.pinch = False
        self.rub = False
        self.is_pinch = rospy.get_param('/app/grasp_type') == 'norub'
        self.is_rub = rospy.get_param('/app/grasp_type') == 'vertrub'

        self.no_of_rubs = 2

    def execute_open(self):
        rospy.loginfo("Opening gripper")
        y_offset = rospy.get_param('/app/y_offset')
        push_force = rospy.get_param('/app/push_force')
        base_pos = np.array([-push_force,y_offset,3,push_force,y_offset,3])
        move_to_xyz_pos(base_pos, self.delta_message, self.arduino)
        sleep(0.5)
        
    def execute_close(self):
        rospy.loginfo("Closing gripper")
        y_offset = rospy.get_param('/app/y_offset')
        push_force=rospy.get_param('/app/push_force')
        base_pos = np.array([push_force,y_offset,3,-push_force,0,3]) # Changed from -1.7
        sleep(0.1)
        rospy.loginfo("calling close")
        move_to_xyz_pos(base_pos, self.delta_message, self.arduino)
        self.clf_pub.publish("start")#start classifier
        sleep(0.5)
        self.clf_pub.publish("end")#start classifier

    def execute_robustify_grasp(self):
        rospy.loginfo("Adjusting gripper")
        y_offset = rospy.get_param('/app/y_offset')
        push_force=rospy.get_param('/app/push_force')
        # robusify_cfg = rospy.get_param('/app/robustify_cfg')
        # base_pos = np.array([-1.3,y_offset,3.8,-push_force,0,3])

        base_pos = np.array([1.9,y_offset,3,-0.5,0,3])
        move_to_xyz_pos(base_pos, self.delta_message, self.arduino)

        z_in = 3.8
        poses = [
            # np.array([-1.8,y_offset,z_in,-0.5,0,3]), # scoop bottom finger
            # np.array([1.7,y_offset,z_in,-push_force,0,z_in+0.1]),
            # np.array([1.7,y_offset,z_in,-push_force,0,z_in]),
            # np.array([1.7,y_offset,3,-push_force,0,3])

            np.array([-1.0,y_offset,z_in,-0.5,0,3]), # scoop bottom finger
            np.array([push_force,y_offset,z_in,-push_force,0,z_in+0.1]),
            np.array([push_force,y_offset,z_in,-push_force,0,z_in]),
            np.array([push_force,y_offset,3,-push_force,0,3])
        ]
        for i in range(2):
            for pose in poses:
                move_to_xyz_pos(pose, self.delta_message, self.arduino)

    def execute_pinch(self):
        self.pub.publish("start")
        self.record_pub.publish("start_%s_%f_%f_%d" % ("pinch", 0, 0,0))
        sleep(1)
        push_force=rospy.get_param('/app/push_force')
        
        # base_pos=np.array([1.7,y_offset,3,-1.7,0,3])
        # final_pos = np.array([-1.7,y_offset,3,1.7,0,3])

        y_offset = rospy.get_param('/app/y_offset')

        base_pos=np.array([push_force,y_offset,3,-push_force,0,3])
        final_pos = np.array([-push_force,y_offset,3,push_force,0,3])

        if rospy.get_param('/app/random_grasp_pos') == 1:
            stop = False
            while not stop:
                yrand, xrand = np.random.uniform(low=-0.75, high=0.75, size=2) #changed from 1
                rand_base_pos = base_pos + np.array([0,yrand,xrand,0,yrand,xrand])
                finger1_norm = np.linalg.norm(rand_base_pos[:3] - final_pos[:3])
                finger2_norm = np.linalg.norm(rand_base_pos[3:] - final_pos[3:])
                stop = finger1_norm > 0.1 and finger2_norm > 0.1
            rospy.loginfo(rand_base_pos)
        else:
            rand_base_pos = base_pos

        rospy.loginfo("Pinching")
        move_to_xyz_pos(rand_base_pos, self.delta_message, self.arduino)
        self.clf_pub.publish("start")#start classifier
        sleep(1.0)
        self.clf_pub.publish("end")#end classifier

        rospy.loginfo("Releasing")
        rospy.loginfo(final_pos)
        move_to_xyz_pos(final_pos, self.delta_message, self.arduino)

        sleep(0.1)
        self.record_pub.publish("stop")
        self.pub.publish("stop")
        sleep(1)

    def execute_rub(self, is_vert_rub=True, time_per=0.2):
        # move_to_xyz_pos([-1.5,1,3,1.5,0,3], self.delta_message, self.arduino)
        # sleep(0.5)

        # self.pub.publish("start")
        # sleep(1)

        push_force = rospy.get_param('/app/push_force')
        is_finger1 = rospy.get_param('/app/is_finger_1')
        
        if is_vert_rub:
            z1 = 3.8 
            z2 = 1.8
            if is_finger1:
                rub_start_xyz = [push_force,0,z1,-push_force,0,3]
                rub_end_xyz = [push_force,0,z2,-push_force,0,3]
            else:
                rub_start_xyz = [push_force,0,3,-push_force,0,z1]
                rub_end_xyz = [push_force,0,3,-push_force,0,z2]

        else: # horz rub
            y1 = 2
            y2 = -2
            if is_finger1:
                rub_start_xyz = [push_force,y1,3,-push_force,0,3]
                rub_end_xyz = [push_force,y2,3,-push_force,0,3]
            else:
                rub_start_xyz = [push_force,0,3,-push_force,y1,3]
                rub_end_xyz = [push_force,0,3,-push_force,y2,3]

        move_to_xyz_pos([1.5,0,3,-1.5,0,3.8], self.delta_message, self.arduino)
        for i in range(self.no_of_rubs):
            move_to_xyz_pos(rub_start_xyz, self.delta_message, self.arduino)
            sleep(time_per)
            move_to_xyz_pos(rub_end_xyz, self.delta_message, self.arduino)
            sleep(time_per)
        move_to_xyz_pos([-1.5,0,3,1.5,0,3], self.delta_message, self.arduino)

        # sleep(0.1)
        # self.pub.publish("stop")

        sleep(1)

    def execute_dbg(self): #was called moveforward
        """
        After a grasp we want to move the fingers slightly forward in order to get a much better grasp of the cloth. 
        """
        rospy.loginfo("Moving gripper forward")
        y_offset = rospy.get_param('/app/y_offset')
        push_force=rospy.get_param('/app/push_force')
        z_forward = 1
        push_force_reduce = 1.7
        final_pose = np.array([push_force-push_force_reduce,y_offset,3,-(push_force - push_force_reduce),0,3])
        move_to_xyz_pos(final_pose, self.delta_message, self.arduino)
        sleep(0.1)
        final_pose = np.array([push_force-push_force_reduce,y_offset,3+z_forward,push_force,0,3+z_forward])      
        move_to_xyz_pos(final_pose, self.delta_message, self.arduino)
        sleep(0.1)
        final_pose = np.array([push_force,y_offset,3+z_forward,-(push_force),0,3+z_forward])
        self.clf_pub.publish("start")#start classifier
        sleep(0.5)
        self.clf_pub.publish("end")#start classifier
    
    def execute_moveforward(self):
        """
        After a grasp we want to move the fingers slightly forward in order to get a much better grasp of the cloth. 
        """
        rospy.loginfo("Moving gripper forward")
        y_offset = rospy.get_param('/app/y_offset')
        push_force=rospy.get_param('/app/push_force')
        z_forward = 1
        push_force_reduce = 1
        final_pose = np.array([push_force-push_force_reduce,y_offset,3,-(push_force),0,3])
        move_to_xyz_pos(final_pose, self.delta_message, self.arduino)
        sleep(0.1)
        final_pose = np.array([push_force-push_force_reduce,y_offset,3,-push_force,0,3+z_forward])      
        move_to_xyz_pos(final_pose, self.delta_message, self.arduino)
        sleep(0.1)
        final_pose = np.array([push_force,y_offset,3,-push_force,0,3])
        move_to_xyz_pos(final_pose, self.delta_message, self.arduino)
        self.clf_pub.publish("start")#start classifier
        sleep(1)
        self.clf_pub.publish("end")#start classifier


    def joy_callback(self, data):
        if data.buttons[BTN_B_INDEX] != 1:
            return

        self.collection_mode = rospy.get_param('/app/collection_mode')
        self.num_auto_trials = rospy.get_param('/app/num_auto_trials')
        self.trials_to_collect = self.num_auto_trials if self.collection_mode == "auto" else 1
        
        while self.trials_to_collect > 0:            
            if self.is_pinch:
                self.execute_pinch()
                self.trials_to_collect -= 1
                if self.trials_to_collect == 0:
                    self.pinch = False
            elif self.is_rub:
                self.execute_open()
                self.execute_close()
                self.execute_rub(is_vert_rub=True)
                self.execute_open()
                self.trials_to_collect -= 1
                if self.trials_to_collect == 0:
                    self.rub = False
            else:
                rospy.logerr("neither pinch nor rub command sent to linear delta")
                raise NotImplementedError

    def run_command(self, msg):
        command = msg.command.data
        if 'dbg_' in command:
            command = command.replace('dbg_', '')
            debug = True
        else:
            debug = False

        rospy.loginfo("Delta action command received: %s" % command)

        if debug:
            self.record_pub.publish("start_%s_%f_%f_%d" % ("command", 0, 0, 0))
            sleep(0.5)

        if command == 'open':
            self.execute_open()
        elif command == 'close':
            self.execute_close()
        elif command == 'robustify':
            self.execute_robustify_grasp()
        elif command == 'pinch':
            self.execute_pinch()
        elif command == 'vertrub':
            self.execute_rub(is_vert_rub=True)
        elif command == 'horzrub':
            self.execute_rub(is_vert_rub=False)
        elif command == 'moveforward':
            self.execute_moveforward()
        else:
            rospy.logerr("unrecognized command sent to linear delta: %s" % command)
            raise NotImplementedError

        if debug:
            sleep(0.5)
            self.record_pub.publish("stop")

        result = MoveFingersActionResult()
        self.delta_actionserver.set_succeeded(result)

if (__name__=="__main__"):
    dz = DeltaZ()
    rospy.spin()




        # while not rospy.is_shutdown():            
        #     if self.is_pinch == True and self.trials_to_collect > 0:
        #         self.pub.publish("start")
        #         sleep(1)

        #         base_pos=np.array([1.7,1,3,-1.7,0,3])
        #         final_pos = np.array([-1.7,1,3,1.7,0,3])

        #         if rospy.get_param('random_grasp_pos') == 1:
        #             stop = False
        #             while not stop:
        #                 yrand, xrand = np.random.uniform(low=-0.75, high=0.75, size=2) #changed from 1
        #                 rand_base_pos = base_pos + np.array([0,yrand,xrand,0,yrand,xrand])
        #                 finger1_norm = np.linalg.norm(rand_base_pos[:3] - final_pos[:3])
        #                 finger2_norm = np.linalg.norm(rand_base_pos[3:] - final_pos[3:])
        #                 stop = finger1_norm > 0.1 and finger2_norm > 0.1
        #         else:
        #             rand_base_pos = base_pos

        #         rospy.loginfo("Pinching")
        #         rospy.loginfo(rand_base_pos)
        #         move_to_xyz_pos(rand_base_pos, self.delta_message, self.arduino)
        #         sleep(1)
        #         rospy.loginfo("Releasing")
        #         rospy.loginfo(final_pos)
        #         move_to_xyz_pos(final_pos, self.delta_message, self.arduino)

        #         sleep(0.1)
        #         self.pub.publish("stop")
        #         self.trials_to_collect -= 1
        #         if self.trials_to_collect == 0:
        #             self.pinch = False
        #         sleep(1)
        #     if self.is_rub == True and self.trials_to_collect > 0:                
        #         time_per = 0.2
        #         move_to_xyz_pos([-1.5,1,3,1.5,0,3], self.delta_message, self.arduino)
        #         sleep(0.5)

        #         self.pub.publish("start")
        #         sleep(1)
                
        #         move_to_xyz_pos([1.5,1.1,3,-1.5,0,3.8], self.delta_message, self.arduino)
        #         for i in range(self.no_of_rubs):
        #             move_to_xyz_pos([1.5,1.1,3,-1.5,0,3.8], self.delta_message, self.arduino)
        #             sleep(time_per)
        #             move_to_xyz_pos([1.5,1.1,3,-1.5,0,1.8], self.delta_message, self.arduino)
        #             sleep(time_per)
        #         move_to_xyz_pos([-1.5,1,3,1.5,0,3], self.delta_message, self.arduino)

        #         sleep(0.1)
        #         self.pub.publish("stop")
        #         self.trials_to_collect -= 1
        #         if self.trials_to_collect == 0:
        #             self.rub = False
        #         sleep(1)