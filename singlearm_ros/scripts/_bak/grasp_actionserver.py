#!/usr/bin/env python
import rospy 
import actionlib
from singlearm_ros.msg import MoveFingersAction, MoveFingersActionResult
from delta_fingers.linear_deltaz import LinearDelta

class GraspActionServer:
    def __init__(self):
        rospy.init_node('delta_controller')

        self.delta = LinearDelta()

        self.delta_actionserver = actionlib.SimpleActionServer('move_fingers', MoveFingersAction, execute_cb=self.run_command, auto_start=False) 
        self.delta_actionserver.start()

    def run_command(self, msg):
        rospy.loginfo("Delta action request received")

        print(msg.command.data)

        # TODO run command

        result = MoveFingersActionResult()
        self.delta_actionserver.set_succeeded(result)

    def spin(self):
        rospy.spin()

if __name__ == "__main__":
    server = GraspActionServer()
    server.spin()
