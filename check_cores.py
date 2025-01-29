import multiprocessing
import os

try:
    import psutil

    psutil_available = True
except ImportError:
    psutil_available = False
    print("psutil library not found. Physical core count will not be available.")


def get_cpu_info():
    logical_cores_multiprocessing = multiprocessing.cpu_count()
    logical_cores_os = os.cpu_count()

    print(f"Logical CPU Cores (multiprocessing.cpu_count()): {logical_cores_multiprocessing}")
    print(f"Logical CPU Cores (os.cpu_count()): {logical_cores_os}")

    if psutil_available:
        physical_cores = psutil.cpu_count(logical=False)
        print(f"Physical CPU Cores (psutil.cpu_count(logical=False)): {physical_cores}")
    else:
        print("Physical CPU Cores: Not Available (psutil not installed)")


if __name__ == "__main__":
    get_cpu_info()
