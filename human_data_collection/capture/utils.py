"""
Utility Functions Module
"""
import subprocess


def check_adb_setup():
    """Check ADB connection setup"""
    try:
        # Get device list
        result = subprocess.run(['adb', 'devices'], capture_output=True, text=True)
        lines = result.stdout.strip().split('\n')[1:]  # Skip header line

        # Find available devices
        available_devices = []
        for line in lines:
            if line.strip() and '\t' in line:
                device_id, status = line.split('\t')
                if status == 'device':
                    available_devices.append(device_id)

        if not available_devices:
            raise AssertionError("No available ADB devices found")

        # If multiple devices, prioritize physical devices (non-emulators)
        target_device = None
        for device in available_devices:
            if not device.startswith('emulator-'):
                target_device = device
                break

        # If no physical device found, use first available device
        if target_device is None:
            target_device = available_devices[0]

        print(f"Selected ADB device: {target_device}")

        # Use specified device for port forwarding
        adb_reverse = subprocess.run(['adb', '-s', target_device, 'reverse', 'tcp:8012', 'tcp:8012'],
                                   capture_output=True, text=True)
        print("adb reverse output:")
        print(adb_reverse.stdout)
        if "error" in adb_reverse.stdout:
            raise AssertionError("adb reverse failed")
        print(f"adb setup is ok, local port 8012 is forwarded to VR device {target_device} port 8012")
    except FileNotFoundError:
        raise AssertionError("adb is not installed")
