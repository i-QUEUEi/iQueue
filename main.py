import subprocess
import sys


print("🚀 iQueue System Starting...")

subprocess.run([sys.executable, "src/model_implementation/train_model.py"], check=True)

print("\n🎯 Training complete. You can now run prediction:")
subprocess.run([sys.executable, "src/predict.py"], check=True)