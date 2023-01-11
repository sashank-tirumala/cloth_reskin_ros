import cv2
import argparse
def mark_frames(video_path):
    video = cv2.VideoCapture(video_path)
    cv2.namedWindow("video", cv2.WINDOW_AUTOSIZE)
    count = 0
    while video.isOpened():
        ret, frame = video.read()
        if(ret == True):
            cv2.imshow(str(count), frame)
            key = cv2.waitKey(0)
            while key not in [ord('q'), ord('k')]:
                key = cv2.waitKey(0)
            # Quit when 'q' is pressed
            if key == ord('q'):
                break
            count = count+1
        else:
            break
    video.release()
    cv2.destroyAllWindows()
    frame_start = int(input("At which frame did contact start?"))
    frame_end = int(input("At which frame did contact end?"))
    final_str = str(frame_start)+","+str(frame_end)
    return final_str

if(__name__ == "__main__"):
    parser = argparse.ArgumentParser(description='mark the video file')
    parser.add_argument('--name', type=str, required=True)
    args = parser.parse_args()
    name = args.name
    vid_path = '/home/sashank/catkin_ws/src/tactilecloth/videos/'+str(name)+'_bx_by_bz.avi'
    finale_str = mark_frames(vid_path)
    file1 = open('/home/sashank/catkin_ws/src/tactilecloth/video_markers/'+str(name)+'.txt',"w+")
    file1.write(finale_str)
    file1.close()
    pass