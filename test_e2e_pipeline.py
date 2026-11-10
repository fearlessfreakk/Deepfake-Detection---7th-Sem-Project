import os
import sys

# Ensure Windows torch DLL loading
if sys.platform == 'win32':
    torch_lib = os.path.join(sys.prefix, 'Lib', 'site-packages', 'torch', 'lib')
    if os.path.exists(torch_lib):
        try:
            os.add_dll_directory(torch_lib)
        except Exception:
            pass

import django
os.environ.setdefault('DJANGO_SETTINGS_MODULE', 'project_settings.settings')
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), 'Django Application')))
django.setup()

import torch
import numpy as np
import cv2
from ml_app.views import Model, predict, get_accurate_model, train_transforms, validation_dataset

def test_model_loading_and_forward():
    print("--- 1. Testing Model Architecture & Forward Pass ---")
    model = Model(num_classes=2).cpu()
    dummy_input = torch.zeros(1, 10, 3, 112, 112)
    fmap, logits = model(dummy_input)
    print("Forward pass successful!")
    print(f"Feature map shape: {fmap.shape}, Logits shape: {logits.shape}")

def test_weights_loading():
    print("\n--- 2. Testing Checkpoint Loading ---")
    model_path = get_accurate_model(10)
    print(f"Selected model path for 10 frames: {model_path}")
    assert os.path.exists(model_path), f"Model path does not exist: {model_path}"
    
    model = Model(num_classes=2).cpu()
    state_dict = torch.load(model_path, map_location='cpu')
    model.load_state_dict(state_dict)
    model.eval()
    print("Checkpoint loaded into model successfully!")

    dummy_input = torch.zeros(1, 10, 3, 112, 112)
    pred_idx, conf = predict(model, dummy_input, video_file_name="test_dummy")
    print(f"Prediction: {pred_idx} (0=FAKE, 1=REAL), Confidence: {conf:.2f}%")

def test_video_processing():
    print("\n--- 3. Testing Synthetic Video Creation & Dataset Processing ---")
    video_dir = os.path.join('Django Application', 'uploaded_videos')
    os.makedirs(video_dir, exist_ok=True)
    test_video_path = os.path.join(video_dir, 'synthetic_test.mp4')

    # Create a 20-frame synthetic video (128x128)
    fourcc = cv2.VideoWriter_fourcc(*'mp4v')
    out = cv2.VideoWriter(test_video_path, fourcc, 10, (128, 128))
    for i in range(20):
        frame = np.full((128, 128, 3), (i * 10) % 255, dtype=np.uint8)
        # Draw a circle on each frame
        cv2.circle(frame, (64, 64), 30, (255, 255, 255), -1)
        out.write(frame)
    out.release()
    print(f"Synthetic test video created at: {test_video_path}")

    # Process via validation_dataset
    dataset = validation_dataset([test_video_path], sequence_length=10, transform=train_transforms)
    frames_tensor = dataset[0]
    print(f"Dataset extracted frames tensor shape: {frames_tensor.shape}")

    model_path = get_accurate_model(10)
    model = Model(num_classes=2).cpu()
    model.load_state_dict(torch.load(model_path, map_location='cpu'))
    model.eval()

    pred_idx, conf = predict(model, frames_tensor, video_file_name="synthetic_test")
    label = "REAL" if pred_idx == 1 else "FAKE"
    print(f"End-to-End Prediction on Synthetic Video: {label} ({conf:.2f}%)")
    print("\n>>> ALL TESTS PASSED SUCCESSFULLY! <<<")

if __name__ == '__main__':
    test_model_loading_and_forward()
    test_weights_loading()
    test_video_processing()
