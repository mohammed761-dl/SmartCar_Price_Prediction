<div align="center">

# CarMarket Pro: AI-Powered Price Prediction Service

[![Python](https://img.shields.io/badge/Python-3.11+-3776AB?style=for-the-badge&logo=python&logoColor=white)](https://python.org)
[![FastAPI](https://img.shields.io/badge/FastAPI-0.100+-009688?style=for-the-badge&logo=fastapi&logoColor=white)](https://fastapi.tiangolo.com)
[![Docker](https://img.shields.io/badge/Docker-Ready-2496ED?style=for-the-badge&logo=docker&logoColor=white)](https://docker.com)
[![scikit-learn](https://img.shields.io/badge/scikit--learn-ML%20Engine-F7931E?style=for-the-badge&logo=scikit-learn&logoColor=white)](https://scikit-learn.org)

**A production-grade Machine Learning microservice for intelligent vehicle valuations.**  
_From exploratory data analysis to containerized deployment — a complete MLOps workflow._

[Features](#key-features) • [Architecture](#project-architecture) • [Quick Start](#quick-start) • [API Reference](#api-reference) • [Docker](#docker-deployment)

</div>

---

## Key Features

| Feature                    | Description                                               |
| -------------------------- | --------------------------------------------------------- |
| **ML-Powered Predictions** | Random Forest Regressor trained on 2,000+ vehicle records |
| **Currency Normalization** | Automatic PKR → USD conversion for global accessibility   |
| **Container-Ready**        | Production Dockerfile with optimized layer caching        |
| **RESTful API**            | FastAPI with automatic OpenAPI/Swagger documentation      |
| **Dynamic Pathing**        | Environment-agnostic model loading (Windows/Linux/Docker) |

---

## Project Architecture

```text
SmartCar_Price_Prediction/
│
├── research/                           # Research & Development
│   ├── car_dataset_2000_rows.csv       # Raw training data
│   └── Smartcar_price_predection.ipynb # EDA & Model Training
│
├── app/                                # Production API
│   ├── main.py                         # FastAPI application
│   ├── requirements.txt                # Python dependencies
│   └── models/                         # Serialized ML Artifacts
│       ├── car_price_model.pkl         # Trained Random Forest
│       └── label_encoders.pkl          # Categorical encoders
│
├── Dockerfile                          # Container configuration
└── README.md
```

---

## The ML Journey

### Phase 1: Data Discovery & Integrity

| Aspect                   | Finding                       | Action Taken                                    |
| ------------------------ | ----------------------------- | ----------------------------------------------- |
| **Dataset Size**         | 2,000 vehicle records         | Sufficient for baseline model                   |
| **Currency**             | Prices in PKR (regional)      | Added USD conversion layer (`÷280`)             |
| **Categorical Features** | Brand, Model, Fuel Type, etc. | Applied `LabelEncoder` transformation           |
| **Outlier Detection**    | Skewed price distribution     | Used **Boxplots (Mustage Graphs)** for analysis |

### Phase 2: Model Selection & Rationale

We evaluated multiple regression approaches before selecting our production model:

| Model                       | Pros                                      | Cons                                        | Verdict                |
| --------------------------- | ----------------------------------------- | ------------------------------------------- | ---------------------- |
| **Linear Regression**       | Simple, interpretable                     | Cannot capture non-linear depreciation      | Too simplistic         |
| **XGBoost**                 | High accuracy, handles complexity         | Overkill for dataset size, slower inference | Unnecessary complexity |
| **Random Forest Regressor** | Handles non-linearity, robust to outliers | Slightly larger model size                  | **Selected**           |

> **Why Random Forest?**  
> Car depreciation is inherently **non-linear** — a vehicle doesn't lose value at a constant rate. Random Forest's ensemble of decision trees captures these complex relationships while remaining robust against overfitting on our 2,000-row dataset.

### Phase 3: Feature Engineering Pipeline

```python
# Categorical columns transformed via LabelEncoder
categorical_features = [
    'brand',        # Toyota, Honda, Suzuki, etc.
    'model',        # Corolla, Civic, Swift, etc.
    'fuel_type',    # Petrol, Diesel, Hybrid, Electric
    'transmission', # Manual, Automatic, CVT
    'car_type',     # Sedan, Hatchback, SUV, Truck
    'drive_type'    # FWD, RWD, AWD
]
```

---

## Deep Dive: Engineering & Methodology

### Why Regression?

For this project, we needed to predict a **continuous numerical value** (the Price). In Machine Learning, this is known as a **Regression** problem.

**Other Options Considered:**

- **Classification:** We could have used classification to predict "Price Ranges" (e.g., $10k-$15k), but that is less precise for a professional valuation tool.
- **Time-Series Forecasting:** Useful if we were predicting the price of _one_ car over 10 years, but not for comparing different cars today.

**The Verdict:** Regression provides the exact dollar amount needed for a competitive marketplace.

### Why Random Forest?

While we tested **Linear Regression**, it was too "stiff"—it assumes that if a car is 2x older, it's 2x cheaper. Real life isn't like that.

- **The Power of Ensembles:** Random Forest builds hundreds of "Decision Trees" and averages them.
- **Non-Linearity:** It understands that a BMW might lose value faster than a Toyota, or that a Diesel engine might be more valuable in certain body types. It captures the **complexity** of the 2,000 rows without "overfitting" (memorizing) the data.

---

## Step-by-Step Research Pipeline

Our `Smartcar_price_predection.ipynb` follows a strict 5-step engineering process:

### 1. Data Ingestion

Loading the 2,000-row dataset and performing a first "sanity check" on the price column.

### 2. Exploratory Data Analysis (EDA)

- We generated **Mustage Graphs (Boxplots)** to identify outliers—cars that were suspiciously cheap or expensive.
- We analyzed the distribution of the "Year" to ensure our model wasn't biased toward only old cars.

### 3. Feature Encoding

- Computers can't read "Toyota." We used **Label Encoding** to map every unique string to a unique number (e.g., Toyota = 1, Honda = 2).
- **Crucial Step:** We saved these encoders as `.pkl` files so the FastAPI server can "translate" the user's text back into the same numbers the AI learned.

### 4. The Train-Test Split (80/20)

- We hid 20% of the data from the AI during training.
- We then used this hidden data to calculate the **R² Score** to prove the model actually works on cars it has never seen before.

### 5. Artifact Export

We used `joblib` to "freeze" the brain of our model, allowing it to be moved from a research notebook into a production API.

---

## API Architecture & DevOps Integration

### Dynamic Model Loading

The API uses **absolute path resolution** to ensure cross-platform compatibility:

```python
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
MODEL_PATH = os.path.join(BASE_DIR, "models", "car_price_model.pkl")
ENCODER_PATH = os.path.join(BASE_DIR, "models", "label_encoders.pkl")
```

> This pattern guarantees the API locates model artifacts whether running on Windows, Linux, or inside a Docker container.

### Currency Conversion Layer

```python
# Model predicts in PKR → Convert to USD for global usability
usd_price = raw_prediction / 280
```

### Response Schema

```json
{
  "estimated_price_usd": 4500.0,
  "currency": "USD",
  "formatted_price": "$4,500.00"
}
```

---

## Quick Start

### Prerequisites

- Python 3.11+
- pip or conda

### Installation

```bash
# Clone the repository
git clone https://github.com/mohammed761-dl/SmartCar_Price_Prediction.git
cd SmartCar_Price_Prediction

# Install dependencies
pip install -r app/requirements.txt
```

### Run the API Server

```bash
uvicorn app.main:app --reload
```

**API Documentation:** [http://127.0.0.1:8000/docs](http://127.0.0.1:8000/docs)

---

## Docker Deployment

### Build the Image

```bash
docker build -t carmarket-pro:latest .
```

### Run the Container

```bash
docker run -d -p 8000:8000 --name carmarket-api carmarket-pro:latest
```

### Verify Deployment

```bash
curl http://localhost:8000/
```

Expected response:

```json
{
  "message": "Welcome to CarMarket Pro AI Pricing Engine",
  "status": "Online",
  "currency": "USD"
}
```

---

## API Reference

### `GET /`

Health check endpoint.

### `POST /predict`

Generate a price prediction for a vehicle.

#### Request Body

| Field          | Type    | Description              | Example       |
| -------------- | ------- | ------------------------ | ------------- |
| `brand`        | string  | Vehicle manufacturer     | `"Toyota"`    |
| `model`        | string  | Vehicle model name       | `"Corolla"`   |
| `year`         | integer | Manufacturing year       | `2018`        |
| `engine_size`  | float   | Engine displacement (L)  | `1.8`         |
| `fuel_type`    | string  | Fuel type                | `"Petrol"`    |
| `transmission` | string  | Transmission type        | `"Automatic"` |
| `mileage`      | integer | Odometer reading (km)    | `45000`       |
| `car_type`     | string  | Vehicle body type        | `"Sedan"`     |
| `drive_type`   | string  | Drivetrain configuration | `"FWD"`       |

#### Sample Request

```bash
curl -X POST "http://localhost:8000/predict" \
  -H "Content-Type: application/json" \
  -d '{
    "brand": "Toyota",
    "model": "Corolla",
    "year": 2018,
    "engine_size": 1.8,
    "fuel_type": "Petrol",
    "transmission": "Automatic",
    "mileage": 45000,
    "car_type": "Sedan",
    "drive_type": "FWD"
  }'
```

#### Sample Response

```json
{
  "estimated_price_usd": 12847.32,
  "currency": "USD",
  "formatted_price": "$12,847.32"
}
```

---

## Tech Stack

| Layer                | Technology                   |
| -------------------- | ---------------------------- |
| **ML Framework**     | scikit-learn (Random Forest) |
| **API Framework**    | FastAPI                      |
| **Serialization**    | joblib                       |
| **Data Processing**  | pandas                       |
| **Containerization** | Docker                       |
| **ASGI Server**      | Uvicorn                      |

---

## Future Roadmap

- [ ] Model retraining pipeline with MLflow tracking
- [ ] Prometheus metrics endpoint for monitoring
- [ ] A/B testing infrastructure for model comparison
- [ ] Multi-currency support via exchange rate API
- [ ] Kubernetes Helm chart for orchestration

---

## Contributing

Contributions are welcome! Please feel free to submit a Pull Request.

---

## License

This project is licensed under the MIT License - see the [LICENSE](LICENSE) file for details.

---

<div align="center">

**Built for the MLOps community**

---

**Author:** [Mohammed Cherkaoui](https://github.com/mohammed761-dl)

</div>
