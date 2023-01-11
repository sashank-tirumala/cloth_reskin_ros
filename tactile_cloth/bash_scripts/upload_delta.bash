if [[ ! -c /dev/ttyACM0 ]] || [[ ! -c /dev/ttyACM1 ]]; then
	echo "check tty ports"
	exit 1
fi

echo "Uploading delta"
while true
do
	if arduino --upload /home/tweng/tactile_ws/src/tactile_cloth/scripts/linear_delta_test/delta_array_6motors/delta_array_6motors.ino --port /dev/ttyACM1; then
		break
	else
		sleep 5
	fi
done
