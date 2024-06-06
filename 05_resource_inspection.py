#! /usr/bin/env python
# nvidia smi is  maintained by Nvidia corp themselves
# careful here, the pip package is called nvidia-ml-py3 (nvidia-smi is something else)
from nvidia_smi import *  # careful here, the pip package is called nvidia-ml-py3 (nvidia-smi is something else)
import psutil

def inspect_gpu():
    nvmlInit()
    print("========== GPU INFO =================")
    print(f"Driver Version: {nvmlSystemGetDriverVersion().decode('utf-8') }")

    device_count = nvmlDeviceGetCount()
    print(f"number of devices {device_count}")
    mib = 1024 * 1024
    for i in range(device_count):
        handle = nvmlDeviceGetHandleByIndex(i)
        devname = nvmlDeviceGetName(handle)
        meminfo = nvmlDeviceGetMemoryInfo(handle)

        print()
        print(f"device {i} is {devname}")
        print(f"total mem  {meminfo.total/mib:.0f} MiB")
        print(f"in use mem {meminfo.used/mib:.0f} MiB")
        print(f"free mem   {meminfo.free/mib:.0f} MiB")

    nvmlShutdown()


def inspect_cpu():
    print("========== CPU INFO =================")
    cpu_count = psutil.cpu_count()
    print(f"Number of CPUs: {cpu_count}")

    # Get CPU Usage for Each Core
    cpu_percent: list[float] = psutil.cpu_percent(percpu=True)
    for idx, usage in enumerate(cpu_percent):
        print(f"CORE_{idx+1}: {usage}%")

    # Get RAM Information
    mem_usage = psutil.virtual_memory()

    print(f"Total: {mem_usage.total/(1024**3):.0f} Gib")
    print(f"Used: {mem_usage.used/(1024**3):.0f} Gib")
    print(f"Free: {(mem_usage.total - mem_usage.used)/(1024**3):.0f} Gib")


def main():
    inspect_gpu()
    inspect_cpu()
    # todo - which models can I use, given these numbers

#################################
if __name__ == "__main__":
    main()
