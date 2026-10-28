from glob import glob

from setuptools import setup

package_name = "jbl_nodes"

setup(
    name=package_name,
    version="0.1.0",
    packages=[package_name],
    data_files=[
        ("share/ament_index/resource_index/packages", ["resource/" + package_name]),
        ("share/" + package_name, ["package.xml"]),
        ("share/" + package_name + "/launch", glob("launch/*.launch.py")),
        ("share/" + package_name + "/config", glob("config/*.yaml")),
    ],
    install_requires=["setuptools"],
    zip_safe=True,
    maintainer="Jakhongir Nodirov",
    maintainer_email="jakhon37@gmail.com",
    description="Camera, docking judge, baseline, policy runner and simulator for autoJetsonBot.",
    license="Apache-2.0",
    tests_require=["pytest"],
    entry_points={
        "console_scripts": [
            "camera_node = jbl_nodes.camera_node:main",
            "dock_judge = jbl_nodes.dock_judge:main",
            "servo_baseline = jbl_nodes.servo_baseline:main",
            "policy_runner = jbl_nodes.policy_runner:main",
            "fake_robot = jbl_nodes.fake_robot:main",
            "episode = jbl_nodes.episode_cli:main",
            "trial_logger = jbl_nodes.trial_logger:main",
        ],
    },
)
