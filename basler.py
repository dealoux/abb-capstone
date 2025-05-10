from pypylon import pylon
import cv2

# List available cameras
factory = pylon.TlFactory.GetInstance()
devices = factory.EnumerateDevices()
print(f"Found {len(devices)} devices")

for i, device in enumerate(devices):
    print(f"Device {i}: {device.GetFriendlyName()}")

# Try connecting to first camera
if devices:
    camera = pylon.InstantCamera(factory.CreateDevice(devices[0]))
    camera.Open()
    print(f"Connected to {camera.GetDeviceInfo().GetModelName()}")
    camera.Close()
else:
    print("No cameras found")
