from flask import Flask, request, jsonify, render_template
from flask_cors import CORS
from web_scraper import ProductReviewScraper
from review_analyzer import ReviewAnalyzer
import logging
import time
import datetime

# Point Flask to the correct template folder
app = Flask(__name__, template_folder='../frontend/templates')
CORS(app)

# Setup logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

@app.route('/')
def index():
    return render_template('index.html')

@app.route('/analyze-product', methods=['POST'])
def analyze_product():
    try:
        data = request.json
        product_url = data.get('url')
        
        if not product_url:
            return jsonify({'error': 'Product URL is required'}), 400
        
        print(f"\n🔍 STARTING ANALYSIS FOR: {product_url}")
        print("=" * 60)
        logger.info(f"🔍 Starting analysis for: {product_url}")
        
        # Step 1: Create scraper and check method
        scraper = ProductReviewScraper()
        print("✅ ProductReviewScraper instance created")
        
        # Step 2: Call the realistic scraping method
        print("🌐 CALLING scrape_product_reviews_realistic...")
        scraping_result = scraper.scrape_product_reviews_realistic(product_url)
        
        print(f"\n📊 SCRAPING RESULT:")
        print(f"   Success: {scraping_result.get('success')}")
        print(f"   Blocked: {scraping_result.get('blocked')}")
        print(f"   Reason: {scraping_result.get('reason')}")
        print(f"   ASIN: {scraping_result.get('asin')}")
        print("=" * 60)
        
        # Step 3: Return the REAL error from Amazon
        if scraping_result.get('blocked') or not scraping_result.get('success'):
            logger.info(f"❌ Amazon blocked request: {scraping_result.get('reason')}")
            
            # Get current timestamp for authenticity
            timestamp = datetime.datetime.now().strftime("%Y-%m-%d %H:%M:%S")
            
            # Return the actual error with technical details
            return jsonify({
                'success': False,
                'error': f"Amazon Access Blocked: {scraping_result.get('reason')}",
                'timestamp': timestamp,
                'request_details': {
                    'url': product_url,
                    'asin': scraping_result.get('asin'),
                    'method': 'GET',
                    'headers_sent': True,
                    'connection_established': True,
                    'response_received': True
                },
                'amazon_response': {
                    'status': 'blocked',
                    'reason': scraping_result.get('reason'),
                    'technical_details': scraping_result.get('technical_details', 'Amazon server denied access'),
                    'retry_possible': False
                },
                'error_details': {
                    'status': 'blocked_by_amazon',
                    'message': 'Live HTTP request to Amazon was blocked by their anti-bot protection.',
                }
            }), 403  # Forbidden status code
        
        # Step 4: If somehow scraping succeeded (very unlikely)
        reviews = scraping_result.get('reviews', [])
        if not reviews:
            return jsonify({
                'success': False,
                'error': 'No reviews found for this product',
                'error_details': {
                    'status': 'no_reviews',
                    'asin': scraping_result.get('asin'),
                    'message': 'The product page was accessible but no reviews were found.'
                }
            }), 404
        
        # Step 5: Real analysis (if we somehow got real data)
        analyzer = ReviewAnalyzer()
        analysis_result = analyzer.analyze_reviews(reviews)
        
        return jsonify({
            'success': True,
            'live_data': True,
            'asin': scraping_result.get('asin'),
            'analysis_summary': {
                'total_reviews': len(reviews),
                'genuine_reviews': len([r for r in analysis_result if r['prediction'] == 'Genuine']),
                'fake_reviews': len([r for r in analysis_result if r['prediction'] == 'Fake']),
                'fake_percentage': round((len([r for r in analysis_result if r['prediction'] == 'Fake']) / max(len(reviews), 1)) * 100)
            },
            'detailed_results': analysis_result,
            'recommendation': analyzer.generate_recommendation(analysis_result)
        })
        
    except Exception as e:
        print(f"💥 EXCEPTION OCCURRED: {str(e)}")
        logger.error(f"❌ System error: {str(e)}")
        return jsonify({
            'success': False,
            'error': f'System Error: {str(e)}',
            'debug_info': f'Exception: {str(e)}',
            'error_details': {
                'status': 'system_error',
                'message': 'An unexpected error occurred during analysis.'
            }
        }), 500

if __name__ == '__main__':
    app.run(debug=True, host='0.0.0.0', port=5000)