from setuptools import find_packages, setup
import os
from glob import glob

package_name = 'perception'

setup(
    name=package_name,
    version='0.0.0',
    packages=find_packages(exclude=['test']),
    data_files=[
        ('share/ament_index/resource_index/packages',
            ['resource/' + package_name]),
        ('share/' + package_name, ['package.xml']),
        (os.path.join('share', package_name), glob('launch/*launch.[pxy][yma]*')),
    ],
    install_requires=['setuptools'],
    zip_safe=True,
    maintainer='rl16432',
    maintainer_email='rluo52237@gmail.com',
    description='TODO: Package description',
    license='TODO: License declaration',
    tests_require=['pytest'],
    entry_points={
        'console_scripts': [
            'localize = perception.localize:main',
            'stereo_localize = perception.stereo_localize:main',
            'trajectory = perception.trajectory:main',
            'evaluation = perception.evaluation:main',
            'state_estimation = perception.state_estimation:main',
            'bev_track = perception.bev_track:main'
        ],
    },
)
