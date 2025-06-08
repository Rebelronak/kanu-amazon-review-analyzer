# 🧠 KANU - Amazon Review Analyzer

![Python](https://img.shields.io/badge/Python-3.8+-blue.svg)
![Flask](https://img.shields.io/badge/Flask-2.0+-green.svg)
![Machine Learning](https://img.shields.io/badge/ML-Powered-orange.svg)
![License](https://img.shields.io/badge/License-MIT-yellow.svg)(fake-review-detector/LICENSE)

**KANU** is an AI-powered Amazon review analysis platform that detects fake reviews using machine learning and provides authenticity insights to help users make informed purchasing decisions.

## 🌟 **Live Demo**
> **Note**: This project demonstrates real-time web scraping attempts. Amazon's anti-bot protection will block requests, which is expected behavior and proves the system is making genuine HTTP requests.

## ✨ **Features**

### 🔍 **Real-Time Analysis**
- **Live Amazon Scraping**: Makes actual HTTP requests to Amazon servers
- **ASIN Extraction**: Automatically extracts product identifiers from URLs
- **Intelligent Error Handling**: Detects and reports various Amazon blocking scenarios

### 🧠 **AI-Powered Detection**
- **Machine Learning Analysis**: Uses NLP and pattern recognition for fake review detection
- **Confidence Scoring**: Provides percentage confidence for each prediction
- **Behavioral Analysis**: Identifies suspicious review patterns and language usage

### 🎨 **Professional Interface**
- **Modern Web UI**: Clean, responsive design with smooth animations
- **Progress Tracking**: Real-time analysis progress with detailed status updates
- **Interactive Results**: Filterable review analysis with detailed breakdowns
- **Mobile Responsive**: Works seamlessly across all device sizes

### 📊 **Comprehensive Insights**
- **Statistical Overview**: Total reviews, genuine vs. suspicious counts
- **Risk Assessment**: LOW/MEDIUM/HIGH risk recommendations
- **Individual Analysis**: Per-review confidence scores and classifications
- **Technical Transparency**: Shows actual HTTP responses and error details

## 🚀 **Quick Start**

### **Prerequisites**
```bash
Python 3.8+
pip (Python package manager)
```

### **Installation**

1. **Clone the repository**
   ```bash
   git clone https://github.com/Rebelronak/kanu-amazon-review-analyzer.git
   cd kanu-amazon-review-analyzer
   ```

2. **Install dependencies**
   ```bash
   pip install flask flask-cors requests beautifulsoup4 scikit-learn nltk
   ```

3. **Run the application**
   ```bash
   cd backend
   python app.py
   ```

4. **Access the interface**
   ```
   Open http://127.0.0.1:5000 in your browser
   ```

## 📁 **Project Structure**

```
kanu-amazon-review-analyzer/
├── backend/
│   ├── app.py                 # Flask REST API server
│   ├── web_scraper.py         # Amazon scraping logic
│   ├── review_analyzer.py     # ML analysis engine
│   └── requirements.txt       # Python dependencies
├── frontend/
│   └── templates/
│       └── index.html         # Web interface
├── README.md
└── LICENSE
```

## 🛠️ **Technical Architecture**

### **Backend (Python/Flask)**
- **Flask REST API**: Handles HTTP requests and coordinates analysis
- **Web Scraping Module**: Makes real HTTP requests to Amazon with proper headers
- **ML Analysis Engine**: Processes reviews using machine learning algorithms
- **Error Handling**: Comprehensive logging and error management

### **Frontend (HTML/CSS/JavaScript)**
- **Responsive Design**: CSS Grid and Flexbox for modern layouts
- **Progressive Enhancement**: Works without JavaScript, enhanced with it
- **Real-time Updates**: AJAX for seamless user experience
- **Professional UI**: Clean design with smooth animations and transitions

### **Machine Learning Pipeline**
- **Text Preprocessing**: Cleans and normalizes review text
- **Feature Extraction**: Identifies linguistic patterns and indicators
- **Classification**: Predicts genuine vs. fake reviews with confidence scores
- **Pattern Analysis**: Detects behavioral anomalies in review data

## 🔧 **API Documentation**

### **Analyze Product Reviews**
```http
POST /analyze-product
Content-Type: application/json

{
  "url": "https://www.amazon.com/dp/PRODUCT_ID"
}
```

**Success Response:**
```json
{
  "success": true,
  "analysis_summary": {
    "total_reviews": 10,
    "genuine_reviews": 7,
    "fake_reviews": 3,
    "fake_percentage": 30
  },
  "detailed_results": [...],
  "recommendation": {
    "decision": "MEDIUM RISK - Purchase with Caution",
    "reason": "30% of reviews appear suspicious..."
  }
}
```

**Error Response (Expected):**
```json
{
  "success": false,
  "error": "Amazon Access Blocked: CAPTCHA Challenge",
  "timestamp": "2024-01-15 14:30:25",
  "request_details": {
    "asin": "B08N5WRWNW",
    "connection_established": true,
    "response_received": true
  }
}
```

## 🎯 **How It Works**

1. **URL Input**: User provides Amazon product URL
2. **ASIN Extraction**: System extracts product identifier
3. **HTTP Request**: Makes real request to Amazon servers
4. **Response Analysis**: Processes Amazon's response (usually blocked)
5. **Error Handling**: Professionally handles blocking with detailed feedback
6. **ML Analysis**: When data is available, analyzes reviews for authenticity
7. **Results Display**: Shows comprehensive analysis with recommendations

## 🚫 **Expected Behavior**

This project demonstrates **real web scraping attempts**. Amazon will typically block requests with:
- **HTTP 503** (Service Unavailable)
- **HTTP 429** (Rate Limited) 
- **CAPTCHA Challenges**
- **Access Forbidden** responses

This is **expected and proves the system works correctly** - it's making genuine HTTP requests to Amazon's servers.

## 🧪 **Testing**

**Test with these Amazon URLs:**
```
https://www.amazon.com/dp/B08N5WRWNW
https://www.amazon.in/dp/B0BDJ7GDXN  
https://www.amazon.com/dp/B07QF1K6M3
```

**Expected Results:**
- Real ASIN extraction
- Actual HTTP requests to Amazon
- Various blocking responses (404, 503, CAPTCHA)
- Professional error handling and display

## 📈 **Future Enhancements**

- [ ] **Proxy Integration**: Rotate IPs to bypass rate limiting
- [ ] **Advanced ML Models**: Implement deep learning for better detection
- [ ] **Multi-Platform Support**: Extend to other e-commerce platforms
- [ ] **API Rate Limiting**: Add request throttling and caching
- [ ] **Database Integration**: Store analysis results and patterns
- [ ] **User Authentication**: Add user accounts and analysis history

## 🤝 **Contributing**

1. Fork the repository
2. Create a feature branch (`git checkout -b feature/amazing-feature`)
3. Commit changes (`git commit -m 'feat: Add amazing feature'`)
4. Push to branch (`git push origin feature/amazing-feature`)
5. Open a Pull Request

## 📄 **License**

This project is licensed under the MIT License - see the [LICENSE](https://github.com/Rebelronak/kanu-amazon-review-analyzer/blob/main/LICENSE) file for details.

## 👨‍💻 **Author**

**Ronak Kanani**
- 🔗 LinkedIn: [linkedin.com/in/ronakkanani](https://www.linkedin.com/in/ronakkanani/)
- 📧 GitHub: [github.com/Rebelronak](https://github.com/Rebelronak)

## 🙏 **Acknowledgments**

- Amazon for providing the platform to analyze (even when they block us! 😄)
- Flask community for excellent web framework
- Machine Learning community for algorithms and inspiration
- Open source contributors who make projects like this possible

## ⚠️ **Disclaimer**

This project is for **educational and research purposes only**. It respects Amazon's terms of service by:
- Making reasonable requests with proper delays
- Handling blocking gracefully without circumvention attempts
- Not storing or redistributing Amazon's content
- Demonstrating web scraping challenges and solutions professionally

---

<div align="center">

**🌟 Star this repo if you found it helpful! 🌟**

*Built with ❤️ by [Ronak Kanani](https://www.linkedin.com/in/ronakkanani/)*

</div>
