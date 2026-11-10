# Deepfake Detection Django Application

## Requirements:

- Python >= 3.8
- Django >= 3.0
- PyTorch & Torchvision
- OpenCV & face_recognition

Requirements are listed in `requirements.txt` and `requirements-local.txt`.

## Directory Structure

- `ml_app` -> Core application containing detection views and logic
- `project_settings` -> Django project configuration
- `static` -> Static assets (CSS, JS, face-api model weights)
- `templates` -> HTML templates
- `models` -> Trained PyTorch model files (`.pt`)

---

# Running Application Locally

### Prerequisite
Ensure trained model weights (`.pt` files) are placed in the `models/` folder.

#### Step 1: Navigate to the Django Application directory
```bash
cd "Django Application"
```

#### Step 2: Create a virtual environment (optional)
```bash
python -m venv venv
```

#### Step 3: Activate virtual environment
On Windows:
```bash
venv\Scripts\activate
```
On Linux/macOS:
```bash
source venv/bin/activate
```

#### Step 4: Install dependencies
```bash
pip install -r requirements-local.txt
```

#### Step 5: Start the development server
```bash
python manage.py runserver
```

Open your browser at `http://127.0.0.1:8000`.

---

# Running Application with Docker

#### Step 1: Install Docker Desktop and start the Docker daemon

#### Step 2: Build and run the container
```bash
docker build -t deepfake-app .
docker run -p 8000:8000 deepfake-app
```

#### Step 3: Access the application
Open your browser at `http://localhost:8000`.
