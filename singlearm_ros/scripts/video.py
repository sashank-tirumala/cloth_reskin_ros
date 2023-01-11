#!/usr/bin/python
from threading import Thread
import gphoto2 as gp
import os
import time
import rospy
from std_msgs.msg import String 

class VideoRecorder:
    def __init__(self):
        self.camera = gp.Camera()
        self.camera.init()
        self.recording = False
        self.watch_thread = None
        self.save_dir = ""

    def start_movie(self, save_dir=""):
        if save_dir != "":
            self.save_dir = save_dir
        if self.recording:
            print("already recording, ignoring...")
            return
        self.recording = True
        cfg = self.camera.get_config()
        movie_cfg = cfg.get_child_by_name("movie")
        movie_cfg.set_value(1)
        self.camera.set_config(cfg)
        self.watch_thread = Thread(target=self._thread)
        self.watch_thread.start()

    def stop_movie(self):
        self.recording = False
        self.watch_thread.join()
        self.watch_thread = None

        cfg = self.camera.get_config()
        movie_cfg = cfg.get_child_by_name("movie")
        movie_cfg.set_value(0)
        self.camera.set_config(cfg)

        print("STOPPING MOVIE")

        timeout=2000
        while True:
            event_type, event_data = self.camera.wait_for_event(timeout)
            if event_type == gp.GP_EVENT_CAPTURE_COMPLETE:
                print("GP_EVENT_CAPTURE_COMPLETE")
            if event_type == gp.GP_EVENT_FILE_ADDED:
                cam_file = self.camera.file_get(
                    event_data.folder, event_data.name, gp.GP_FILE_TYPE_NORMAL
                )
                target_path = os.path.join(self.save_dir, event_data.name)
                print("Image is being saved to {}".format(target_path))
                cam_file.save(target_path)
                print("Saved to {}".format(target_path))
                break
            if event_type == gp.GP_EVENT_FILE_CHANGED:
                print("GP_EVENT_FILE_CHANGED")
            if event_type == gp.GP_EVENT_FOLDER_ADDED:
                print("GP_EVENT_FOLDER_ADDED")
            if event_type == gp.GP_EVENT_TIMEOUT:
                print("GP_EVENT_TIMEOUT")
            if event_type == gp.GP_EVENT_UNKNOWN:
                # print("GP_EVENT_UNKNOWN")
                pass
            else:
                print(event_type)

    def exit(self):
        self.camera.exit()

    def _thread(self):
        timeout = 10000

        while self.recording:
            event_type, event_data = self.camera.wait_for_event(timeout)
            if event_type == gp.GP_EVENT_CAPTURE_COMPLETE:
                print("GP_EVENT_CAPTURE_COMPLETE")
            if event_type == gp.GP_EVENT_FILE_ADDED:
                print("GP_EVENT_FILE_ADDED")
            if event_type == gp.GP_EVENT_FILE_CHANGED:
                print("GP_EVENT_FILE_CHANGED")
            if event_type == gp.GP_EVENT_FOLDER_ADDED:
                print("GP_EVENT_FOLDER_ADDED")
            if event_type == gp.GP_EVENT_TIMEOUT:
                print("GP_EVENT_TIMEOUT")
            if event_type == gp.GP_EVENT_UNKNOWN:
                # print("GP_EVENT_UNKNOWN")
                pass
            else:
                print(event_type)

class Video_Node:
    def __init__(self):
        name = "video_recorder"
        rospy.init_node(name + "_node")

        self.sub = rospy.Subscriber('dbg_image_record', String, callback=self.set_record)
        self.vr = VideoRecorder()

    def set_record(self, msg):
        if msg.data == 'start_dbg_record':
            self.save_dir = rospy.get_param("trial_type_dir")
            rospy.logwarn('starting movie %s' % self.save_dir)
            self.vr.start_movie()
        elif msg.data == 'end_dbg_record':
            rospy.logwarn("ending movie")
            self.vr.stop_movie()

if __name__ == '__main__':
    vn = Video_Node()
    rospy.spin()