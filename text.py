import importlib

modules = [
    "os", "cv2", "time", "math", "smtplib", "threading",
    "numpy", "mediapipe", "flask", "email.mime.text"
]

for m in modules:
    try:
        importlib.import_module(m)
        print(f"{m} ✅ Installed")
    except ImportError:
        print(f"{m} ❌ Not Installed")