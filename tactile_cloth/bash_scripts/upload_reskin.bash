if [[ ! -c /dev/ttyACM0 ]] || [[ ! -c /dev/ttyACM1 ]] || [[ -c /dev/ttyACM2 ]]; then
	echo "check tty ports"
  exit 1
fi

echo "killing ROS"
sudo killall -9 rosmaster

echo "Uploading burst stream"
while true
do
	if arduino --upload ~/tactile_ws/external/reskin_sensor/arduino/5X_burst_stream/5X_burst_stream.ino --port /dev/ttyACM0; then
		break
	else
		sleep 5
	fi
done

echo "Serial monitor output, check values:"
# sleep 5
# timeout 1s head /dev/ttyACM0

sleep 10
echo "Uploading binary burst stream"
while true 
do
	if arduino --upload ~/tactile_ws/external/reskin_sensor/arduino/5X_binary_burst_stream/5X_binary_burst_stream.ino --port /dev/ttyACM0; then
		break
	else
		sleep 5
	fi
done
