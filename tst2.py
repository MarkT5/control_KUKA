from KUKA import KUKA
import time

robot = KUKA("192.168.88.22", camera_enable=False)

while True:
    try:
        exec(input())
        exec(input())
    except Exception as inst:
        print(inst)



#robot.post_to_send_data(1, bytes(f'/arm:0;{m1};{m2};{m3};{m4};{m5}^^^', encoding='utf-8'))
#robot.post_to_send_data(1, bytes(f'/arm_vel:0;1;1;1;1;1^^^', encoding='utf-8'))
#robot.move_arm(-2.4437499999999943, -1.0062499999999943, -0.25, -88.45833333333333, 2.11041666666668, 1.9, arm_ID=0)
#robot.set_arm_vel(-1,-1,-1,-1,-1)
