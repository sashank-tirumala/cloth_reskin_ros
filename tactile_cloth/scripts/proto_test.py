import linear_actuator_pb2
import numpy as np
delta_message = linear_actuator_pb2.lin_actuator()
delta_message.id = 1
vals =  [0.0308392029256,  0.0308538898826,  0.0308538898826, 0.0303948651999, 0.0312020219862, 0.0312020219862, 0.0299999993294, 0.0299999993294, 0.0299999993294, 0.0299999993294]
def create_joint_positions(vals, delta_message):
    for i in range(12):
        if i<len(vals):
            delta_message.joint_pos.append(np.around(np.clip(vals[i], 0.02/100.0, 9.98/100.0),4))
        else:
            delta_message.joint_pos.append(0.03)
create_joint_positions(vals, delta_message)
res = delta_message.SerializeToString()
print(res.decode())