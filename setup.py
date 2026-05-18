from setuptools import setup, find_packages

setup(
    name='cade_vision',
    version='0.1.0',
    packages=find_packages('src'),
    package_dir={'': 'src'},
    install_requires=[
        'rospy',
        'ultralytics',
        'opencv-python',
        'pyrealsense2',
        'numpy',
    ],
)
