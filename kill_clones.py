import psutil
import sys

count = 0
for proc in psutil.process_iter(['pid', 'name', 'cmdline']):
    try:
        cmdline = proc.info['cmdline']
        if cmdline and 'simple_voice_clone.py' in ' '.join(cmdline):
            print(f"Killing process {proc.info['pid']}: {cmdline}")
            proc.kill()
            count += 1
    except (psutil.NoSuchProcess, psutil.AccessDenied, psutil.ZombieProcess):
        pass

print(f"Killed {count} processes.")
