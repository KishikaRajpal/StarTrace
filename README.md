# ⭐ STARTRACE
### AI-Enabled Detection and Classification of Exoplanets from Noisy Astronomical Light Curves

> **Tagline:** Discovering Hidden Worlds Through Stellar Light.  
> **Academic Context:** B.Tech Computer Science Engineering (3rd Year Academic Mini-Project)  
> **Mission Data:** NASA Transiting Exoplanet Survey Satellite (TESS) via Barbara A. Mikulski Archive for Space Telescopes (MAST)

---

## 🌌 1. Project Overview

**StarTrace** is a full-stack, AI-powered astronomical data analysis platform that identifies and classifies potential exoplanet transit signals from noisy stellar light curves.

The platform provides a complete, working pipeline that integrates:
1. **Photometric Data Ingestion:** Live streaming from NASA MAST using Lightkurve (with deterministic offline demonstration fallbacks).
2. **Signal Conditioning:** 5-$\sigma$ cosmic ray outlier rejection, transit-protected Savitzky-Golay baseline detrending, and median normalization.
3. **Periodicity Search:** Box Least Squares (BLS) periodogram search extracting orbital period ($P$), transit epoch ($t_0$), duration, and Signal-to-Noise Ratio (SNR).
4. **Dual-View Feature Extraction:** Generates paired **Global (2001 points)** and **Local (201 points)** phase-folded median-binned vectors.
5. **Deep Learning Classification:** A PyTorch **Dual-Branch 1D Convolutional Neural Network (1D-CNN)** classifying signals into four distinct astrophysical categories:
   - **Class 0:** Exoplanet Transit (Candidate)
   - **Class 1:** Eclipsing Binary System
   - **Class 2:** Stellar Background Blend
   - **Class 3:** Stellar Activity / Instrumental Noise
6. **Transit Parameter Estimation:** Derives physical exoplanet candidate properties: planet radius ratio ($R_p/R_*$), estimated planet radius ($R_\oplus$), semi-major axis ($a$), and equilibrium temperature ($T_{eq}$).
7. **Interactive Visualizations:** High-performance, responsive Plotly.js charts with zoom, pan, hover tooltips, and PNG export.
8. **Exportable Reports:** One-click PDF analysis report generation (ReportLab) and light curve CSV data export.

---

## 🛠️ 2. Technology Stack

### Frontend
- **Framework:** React.js 18 with Vite
- **Styling:** Tailwind CSS with deep-space cosmic theme (midnight navy, neon cyan, electric blue accents)
- **Charts & Visualizations:** Plotly.js (`react-plotly.js`)
- **Icons:** Lucide React
- **HTTP Client:** Axios (configured with proxy and automatic error handling)

### Backend
- **Framework:** Python 3.11+, FastAPI (REST API with OpenAPI Swagger UI)
- **ASGI Server:** Uvicorn
- **Data Validation:** Pydantic
- **Database ORM:** SQLAlchemy ORM with dual-engine support:
  - **PostgreSQL** (Production / Configured via `DATABASE_URL`)
  - **SQLite** (Zero-configuration local fallback for seamless student laptop evaluation)
- **PDF Generation:** ReportLab

### Machine Learning
- **Deep Learning Framework:** PyTorch (CPU-optimized, runs smoothly without a dedicated GPU)
- **Scientific Computing:** NumPy, Pandas, Scikit-learn, SciPy

### Astronomical Processing
- **TESS Light Curve Retrieval:** Lightkurve (NASA STScI MAST Archive)
- **Period Detection:** Astropy (`astropy.timeseries.BoxLeastSquares`)
- **Signal Detrending:** SciPy (`scipy.signal.savgol_filter` with transit masking)

---

## 🏛️ 3. System Architecture

```
StarTrace Architecture
├── Frontend (React + Vite, Port 5173)
│   ├── Landing Page (Mission intro, transit photometry explainer, 4-class cards)
│   ├── Mission Control Dashboard (Live database KPIs, recent analyses table)
│   ├── New Analysis (TIC search, 4 demo presets, CSV drag & drop, 8-step progress)
│   ├── Analysis Results (Classification, probabilities, parameters, 5 Plotly graphs, PDF/CSV)
│   ├── History (Filter by category, search by TIC ID, view details, delete)
│   └── Model Information (1D-CNN architecture, scientific rationale, confusion matrix)
│
├── REST API Layer (FastAPI, Port 8000)
│   ├── POST /api/analysis/run (Execute TIC pipeline)
│   ├── POST /api/analysis/upload (Execute CSV pipeline)
│   ├── GET  /api/analysis/{id} (Retrieve analysis record)
│   ├── GET  /api/analysis/{id}/visualization (Light curve Plotly arrays)
│   ├── GET  /api/analysis/dashboard (Aggregated DB statistics)
│   ├── GET  /api/analysis (Paginated history)
│   ├── DELETE /api/analysis/{id} (Remove record & data files)
│   ├── GET  /api/model/status (Model weights & hardware status)
│   ├── GET  /api/model/metrics (Accuracy, precision, recall, confusion matrix)
│   ├── GET  /api/report/{id}/pdf (Downloadable PDF analysis report)
│   ├── GET  /api/report/{id}/csv (Downloadable light curve data)
│   └── GET  /api/health (System status check)
│
├── Core Pipeline Modules
│   ├── astronomy/data_loader.py (Lightkurve MAST query + synthetic fallbacks)
│   ├── astronomy/preprocessing.py (5-sigma clipping + Savitzky-Golay detrending)
│   ├── astronomy/period_detection.py (Astropy BLS periodogram)
│   ├── astronomy/phase_folding.py (Global 2001 & Local 201 median binning)
│   └── astronomy/parameter_estimation.py (Physical Rp, a, Teq modeling)
│
└── ML Module
    ├── ml/cnn_model.py (DualBranchCNN PyTorch architecture)
    ├── ml/inference.py (Calibrated multi-branch inference)
    ├── ml/train.py (Model training script with synthetic generator)
    ├── ml/evaluate.py (Confusion matrix & metrics generation)
    └── ml/checkpoints/weights.pth (Trained checkpoint)
```

---

## ⚡ 4. Quickstart: Running on Localhost

Both the backend and frontend are pre-configured to run out-of-the-box on standard Windows / macOS / Linux laptops.

### Prerequisites
- **Python 3.10+** (Python 3.11 recommended)
- **Node.js 18+** and **npm**

### Step 1: Clone and Navigate to Directory
```bash
git clone https://github.com/your-username/StarTrace.git
cd StarTrace
```

### Step 2: Start the FastAPI Backend
```bash
# Navigate to backend directory
cd backend

# Install Python dependencies
pip install -r requirements.txt

# Launch FastAPI server
python main.py
```
> The backend server starts at **`http://127.0.0.1:8000`**.  
> Interactive Swagger API docs are accessible at **`http://127.0.0.1:8000/api/docs`**.

### Step 3: Start the React Frontend
Open a new terminal window:
```bash
# Navigate to frontend directory
cd frontend

# Install npm dependencies
npm install

# Start Vite development server
npm run dev
```
> The frontend application starts at **`http://127.0.0.1:5173`**.

---

## 🗄️ 5. Database Setup (PostgreSQL & SQLite)

StarTrace includes **automatic database fallback** designed specifically for academic evaluation:
1. When a PostgreSQL database is reachable at the `DATABASE_URL` specified in `.env`, StarTrace automatically connects to PostgreSQL and creates the necessary tables (`analyses`, `classification_results`, `model_metrics`).
2. If PostgreSQL is offline, uninstalled, or credentials fail, StarTrace **automatically falls back to a local SQLite database (`backend/startrace.db`) without crashing**.

### Configuring PostgreSQL (Optional):
Create a `.env` file in the `backend/` directory:
```env
DATABASE_URL=postgresql://postgres:yourpassword@localhost:5432/startrace
HOST=127.0.0.1
PORT=8000
```

---

## 🧪 6. Sample Datasets for Demonstration

StarTrace includes 4 pre-calibrated sample light curves located in `backend/sample_data/`:

| TIC ID | Preset Target | Category | Period | Depth | Key Features |
|---|---|---|---|---|---|
| **25155310** | Kepler-1649c | **Exoplanet Transit** | 8.689 d | 0.22% | Shallow, symmetric U-shaped dip; Earth-sized candidate |
| **149603524** | Detached Binary | **Eclipsing Binary** | 2.342 d | 8.50% | Deep primary V-shape with distinct secondary eclipse at phase 0.5 |
| **38846515** | Blend Source | **Background Blend** | 3.754 d | 0.85% | Asymmetric ingress/egress profile from aperture contamination |
| **261136638** | Active Star | **Stellar Activity** | 1.428 d | 0.55% | Quasi-periodic rotational starspot modulation and flares |

Users can test these presets directly from the **New Analysis > Sample Demonstration Presets** tab with a single click.

---

## 🤖 7. Model Training & Evaluation

The Dual-Branch 1D-CNN model architecture can be re-trained or evaluated using the built-in CLI scripts:

### Evaluate Model Checkpoint
```bash
cd backend
python -c "from ml.evaluate import get_default_evaluation_metrics; print(get_default_evaluation_metrics())"
```

### Train Model on Synthetic Light Curve Grid
```bash
cd backend
python -m ml.train --epochs 15 --checkpoint ml/checkpoints/weights.pth
```

### Run Automated PyTest Suite
StarTrace includes an automated test suite covering signal conditioning, outlier removal, BLS, phase-folding, tensor dimensions, inference, and API endpoints:
```bash
cd backend
python -m pytest tests/test_backend.py -v
```
*(All 10 tests validate in ~25 seconds on CPU).*

---

## 📄 8. REST API Reference

| Method | Endpoint | Description |
|---|---|---|
| `GET` | `/api/health` | System health, database mode, and PyTorch model status |
| `POST` | `/api/analysis/run` | Execute astronomical pipeline for a given TIC ID |
| `POST` | `/api/analysis/upload` | Ingest and analyze user-uploaded CSV light curve |
| `GET` | `/api/analysis/{id}` | Retrieve complete analysis parameters and classification |
| `GET` | `/api/analysis/{id}/visualization` | Retrieve Plotly-formatted time and flux arrays |
| `GET` | `/api/analysis` | Paginated analysis history with search and category filters |
| `DELETE` | `/api/analysis/{id}` | Delete analysis record and associated compressed data file |
| `GET` | `/api/analysis/dashboard` | Aggregated dashboard KPI counts and recent analyses |
| `GET` | `/api/model/status` | Model weights path, device (CPU), and parameters |
| `GET` | `/api/model/metrics` | Accuracy, per-class precision/recall, and confusion matrix |
| `GET` | `/api/model/architecture` | Network topology breakdown |
| `GET` | `/api/samples` | List of built-in sample presets |
| `GET` | `/api/report/{id}/pdf` | Download formatted scientific PDF report |
| `GET` | `/api/report/{id}/csv` | Download CSV light curve summary |

---

## 📜 9. Academic Attribution & Disclaimer

- **Academic Project:** Developed as a B.Tech Computer Science and Engineering third-year mini-project.
- **Data Source:** NASA TESS mission photometric data retrieved via the Mikulski Archive for Space Telescopes (MAST) at the Space Telescope Science Institute (STScI).
- **Physical Classification Notice:** Detections flagged as "Exoplanet Transit" represent **Potential Exoplanet Candidates** requiring ground-based spectroscopic radial-velocity follow-up for physical confirmation. Planetary parameters are model-fitted estimations.
