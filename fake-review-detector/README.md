# Fake Review Detector

A machine learning project to detect fake reviews from Amazon and Flipkart product links.  
Paste a product link, and the system will analyze reviews to help you decide whether to buy the product.

---

## Features

- Downloads and uses the Amazon Polarity dataset for training.
- Cleans and processes review text.
- Trains a robust ML model (TF-IDF + Logistic Regression).
- REST API for predicting if a review is fake or genuine.
- Easily extensible for scraping reviews from Amazon/Flipkart.

---

## Project Structure

```
fake-review-detector/
│
├── backend/
│   ├── app.py                  # Flask API
│   ├── requirements.txt        # Python dependencies
│   ├── dataset/
│   │   └── download_dataset.py # Download Amazon Polarity dataset
│   └── model/
│       ├── train_model.py      # Train ML model
│       ├── config.json         # Model/data configuration
│       └── ...                 # Model files
└── README.md
```

---

## Getting Started

### 1. Install dependencies

```bash
cd backend
pip install -r requirements.txt
```

### 2. Download the dataset

```bash
python dataset/download_dataset.py
```

### 3. Train the model

```bash
python model/train_model.py
```

### 4. Run the API

```bash
python app.py
```

### 5. Test the API

Send a POST request to `http://127.0.0.1:5000/predict` with JSON:
```json
{
  "review": "This product is amazing!"
}
```

---

## Example API Response

```json
{
  "prediction": "Genuine",
  "confidence": 0.92
}
```

---

## Future Work

- Scrape reviews directly from Amazon/Flipkart product links.
- Build a web frontend for easy use.
- Improve model with more advanced NLP techniques.

---

## License

For educational use only.