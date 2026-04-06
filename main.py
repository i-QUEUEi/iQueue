import os

print("🚀 iQueue System Starting...")

os.system("python src/train_model.py")

print("\n🎯 Training complete. You can now run prediction:")
os.system("python src/predict.py")