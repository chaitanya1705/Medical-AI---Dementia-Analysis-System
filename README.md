# Medical AI – Dementia Analysis System

This project requires three terminals to run simultaneously. Follow the steps below carefully.

---

## Terminal 1 – Node.js Server (API)

```bash
cd server
npm install
npm start
```

---

## Terminal 2 – Flask Backend (Model Inference)

Install all required Python packages:

```bash
cd models
pip install torch torchvision torchaudio transformers flask flask-cors pandas numpy pillow python-dotenv requests scikit-learn timm albumentations joblib google-generativeai openpyxl
```

Then run the backend:

```bash
python app.py
```

---

## Terminal 3 – React Frontend

```bash
npm install
npm run dev
```

---

⚠️ **Note:** Make sure you finish all installations first before running the servers.
