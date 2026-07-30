import os
from glob import glob

from setuptools import find_packages, setup

package_name = "shoulder_detection_ros"

setup(
    name=package_name,
    version="0.1.0",
    packages=find_packages(exclude=["test"]),
    data_files=[
        ("share/ament_index/resource_index/packages", ["resource/" + package_name]),
        ("share/" + package_name, ["package.xml"]),
        (os.path.join("share", package_name, "launch"), glob("launch/*.launch.xml")),
        (os.path.join("share", package_name, "rviz"), glob("rviz/*.rviz")),
    ],
    install_requires=["setuptools"],
    zip_safe=True,
    maintainer="dev",
    maintainer_email="dev@example.com",
    description="DINOv2 road-shoulder detection on Autoware's CARLA camera feed, with RViz and a live overlay window.",
    license="Apache-2.0",
    entry_points={
        "console_scripts": [
            "shoulder_detection_node = shoulder_detection_ros.shoulder_detection_node:main",
            "shoulder_overlay_viewer_node = shoulder_detection_ros.shoulder_overlay_viewer_node:main",
        ],
    },
)
