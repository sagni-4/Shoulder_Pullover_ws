#!/usr/bin/env python3
"""External, independent speed-ceiling safety monitor for the pull-over MRM
project. Subscribes directly to /localization/kinematic_state and, if speed
ever exceeds --ceiling, immediately (a) stands down the pull-over manager and
(b) requests operation_mode -> STOP, repeating both for a few seconds to
make sure they land. Deliberately outside the pull-over manager's own
process, so a bug in that node cannot also disable this.

Usage (after sourcing ROS + autoware + shoulder_pullover_ws setup.bash):
  python3 external_speed_monitor.py --ceiling 3.0
"""
import argparse
import sys
import time

import rclpy
from rclpy.node import Node
from rclpy.qos import QoSDurabilityPolicy, QoSProfile
from nav_msgs.msg import Odometry
from tier4_system_msgs.srv import OperateMrm
from autoware_adapi_v1_msgs.srv import ChangeOperationMode


class ExternalSpeedMonitor(Node):
    def __init__(self, ceiling: float, heartbeat_period: float):
        super().__init__('external_speed_monitor')
        self.ceiling = ceiling
        self.tripped = False
        self.trip_time = None
        self.last_heartbeat = 0.0
        self.heartbeat_period = heartbeat_period

        self.operate_client = self.create_client(
            OperateMrm, '/system/mrm/pull_over_manager/operate')
        self.stop_client = self.create_client(
            ChangeOperationMode, '/api/operation_mode/change_to_stop')

        self.sub = self.create_subscription(
            Odometry, '/localization/kinematic_state', self.on_odom, 10)

        self.get_logger().warn(
            f'ARMED: ceiling={self.ceiling:.2f} m/s. Watching '
            '/localization/kinematic_state.')

    def on_odom(self, msg: Odometry):
        vx = msg.twist.twist.linear.x
        vy = msg.twist.twist.linear.y
        speed = (vx * vx + vy * vy) ** 0.5
        now = time.time()

        if now - self.last_heartbeat >= self.heartbeat_period:
            self.last_heartbeat = now
            print(f'speed={speed:.3f} m/s (ceiling={self.ceiling:.2f})', flush=True)

        if speed > self.ceiling:
            if not self.tripped:
                self.tripped = True
                self.trip_time = now
                print(
                    f'!!! TRIPPED !!! speed={speed:.3f} m/s > ceiling={self.ceiling:.2f} '
                    'm/s -- sending stand-down + change_to_stop', flush=True)
                self.get_logger().error(
                    f'SPEED CEILING EXCEEDED: {speed:.3f} > {self.ceiling:.2f} -- aborting')
            # Keep re-sending every cycle for as long as we're over ceiling --
            # a one-shot or time-boxed resend can be undone by anything else
            # (including a human) re-engaging after the window closes, which
            # happened live 2026-08-13. Only clear once back under ceiling.
            self._send_abort()
        elif self.tripped:
            print(
                f'speed={speed:.3f} m/s back under ceiling={self.ceiling:.2f} -- '
                'clearing trip (still stood down; re-engage is a separate, '
                'deliberate action)', flush=True)
            self.tripped = False
            self.trip_time = None

    def _send_abort(self):
        if self.operate_client.service_is_ready():
            req = OperateMrm.Request()
            req.operate = False
            self.operate_client.call_async(req)
        else:
            print('  (operate service not ready yet)', flush=True)

        if self.stop_client.service_is_ready():
            req2 = ChangeOperationMode.Request()
            self.stop_client.call_async(req2)
        else:
            print('  (change_to_stop service not ready yet)', flush=True)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--ceiling', type=float, default=3.0)
    parser.add_argument('--heartbeat-period', type=float, default=1.0)
    args = parser.parse_args()

    rclpy.init()
    node = ExternalSpeedMonitor(args.ceiling, args.heartbeat_period)
    try:
        rclpy.spin(node)
    except KeyboardInterrupt:
        pass
    finally:
        node.destroy_node()
        rclpy.shutdown()


if __name__ == '__main__':
    main()
