import sys
from cx_Freeze import setup, Executable


setup(name="Button Object Detection",
      version="1",
      description="This software detects objects in realtime with button",
      executables=[Executable("DRIVING.py")]
      )

