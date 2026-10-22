from setuptools import find_packages, setup

package_name = 'lidar_semantic'

setup(
    name=package_name,
    version='0.1.0',
    packages=find_packages(exclude=['test']),
    data_files=[
        ('share/ament_index/resource_index/packages', ['resource/' + package_name]),
        ('share/' + package_name, ['package.xml']),
        ('share/' + package_name + '/launch', ['launch/semantic.launch.py']),
        ('share/' + package_name + '/config', ['config/semantic.yaml']),
        ('share/' + package_name + '/weights', ['weights/semantickitti_tsunghan_1d.tar']),
    ],
    install_requires=['setuptools'],
    zip_safe=True,
    maintainer='Kaan Ural',
    maintainer_email='kaan549@gmail.com',
    description='RandLA-Net semantic segmentation for FastLIO2 point clouds',
    license='Apache-2.0',
    tests_require=['pytest'],
    entry_points={
        'console_scripts': [
            'randlanet_node = lidar_semantic.randlanet_node:main',
        ],
    },
)
