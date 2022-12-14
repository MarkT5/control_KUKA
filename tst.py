from KUKA import KUKA
import time
import threading as thr

def dance1():
    robot = KUKA("192.168.88.22", camera_enable=False)
    dance_delay = 2500

    robot.move_arm(-0, 0, -40, 0, 0, 2, arm_ID=0)
    time.sleep(0.1)
    robot.move_arm(0, -45, -75, 0, 0, arm_ID=1)
    time.sleep(5)
    genius = 0
    speed = 0.02
    rot_speed=0
    for i in range(10):
        genius += 1
        robot.set_arm_vel(rot_speed, 12, 18, 12, 0, arm_ID=0)
        time.sleep(0.1)
        robot.set_arm_vel(rot_speed, -12, -18, -12, 0, arm_ID=1)
        time.sleep(2.2)
        robot.set_arm_vel(rot_speed, 14, -8, -18, 0, arm_ID=0)
        time.sleep(0.1)
        robot.set_arm_vel(rot_speed, -14, 8, 18, 0, arm_ID=1)
        time.sleep(2.2)
        robot.set_arm_vel(rot_speed, -12, -18, -12, 0, arm_ID=0)
        time.sleep(0.1)
        robot.set_arm_vel(rot_speed, 12, 18, 12, 0, arm_ID=1)
        time.sleep(2.2)
        robot.set_arm_vel(rot_speed, -14, 8, 18, 0, arm_ID=0)
        time.sleep(0.1)
        robot.set_arm_vel(rot_speed, 14, -8, -18, 0, arm_ID=1)
        time.sleep(2.2)

        if genius % 2 == 0:
            base_iter = genius % 8 // 2
            print(base_iter)
            if base_iter == 1:
                rot_speed = 3
                robot.move_base(speed, speed, 0)
                time.sleep(0.1)
            if base_iter == 2:
                rot_speed = 0
                robot.move_base(-speed, speed, 0)
                time.sleep(0.1)
            if base_iter == 3:
                rot_speed = -3
                robot.move_base(-speed, -speed, 0)
                time.sleep(0.1)
            if base_iter == 4:
                rot_speed = 0
                robot.move_base(speed, -speed, 0)
                time.sleep(0.1)

    robot.move_base(0, 0, 0)
    robot.set_arm_vel(0, 0, 0, 0, 0, arm_ID=0)
    time.sleep(0.1)
    robot.set_arm_vel(0, 0, 0, 0, 0, arm_ID=1)



def dance2():
    from KUKA import KUKA
    import time

    robot = KUKA("192.168.88.21", camera_enable=False)
    dance_delay = 2500

    robot.move_arm(-0, 0, -40, 0, 0, 2, arm_ID=0)
    time.sleep(0.1)
    robot.move_arm(0, -45, -75, 0, 0, arm_ID=1)
    time.sleep(5)
    genius = 0
    speed = 0.02
    rot_speed = 0
    for i in range(10):
        genius += 1
        robot.set_arm_vel(rot_speed, 12, 18, 12, 0, arm_ID=0)
        time.sleep(0.1)
        robot.set_arm_vel(rot_speed, -12, -18, -12, 0, arm_ID=1)
        time.sleep(2.2)
        robot.set_arm_vel(rot_speed, 14, -8, -18, 0, arm_ID=0)
        time.sleep(0.1)
        robot.set_arm_vel(rot_speed, -14, 8, 18, 0, arm_ID=1)
        time.sleep(2.2)
        robot.set_arm_vel(rot_speed, -12, -18, -12, 0, arm_ID=0)
        time.sleep(0.1)
        robot.set_arm_vel(rot_speed, 12, 18, 12, 0, arm_ID=1)
        time.sleep(2.2)
        robot.set_arm_vel(rot_speed, -14, 8, 18, 0, arm_ID=0)
        time.sleep(0.1)
        robot.set_arm_vel(rot_speed, 14, -8, -18, 0, arm_ID=1)
        time.sleep(2.2)

        if genius % 2 == 0:
            base_iter = genius % 8 // 2
            print(base_iter)
            if base_iter == 1:
                rot_speed = -3
                robot.move_base(speed, -speed, 0)
                time.sleep(0.1)
            if base_iter == 2:
                rot_speed = 0
                robot.move_base(-speed, -speed, 0)
                time.sleep(0.1)
            if base_iter == 3:
                rot_speed = 3
                robot.move_base(-speed, speed, 0)
                time.sleep(0.1)
            if base_iter == 4:
                rot_speed = 0
                robot.move_base(speed, speed, 0)
                time.sleep(0.1)

    robot.move_base(0, 0, 0)
    robot.set_arm_vel(0, 0, 0, 0, 0, arm_ID=0)
    time.sleep(0.1)
    robot.set_arm_vel(0, 0, 0, 0, 0, arm_ID=1)


robo1 = thr.Thread(target=dance1, args=())
robo2 = thr.Thread(target=dance2, args=())
robo1.start()
robo2.start()





