/**
 * Fake Review Detector - Frontend JavaScript
 * Created by Ronak Kanani
 * Advanced AI-powered review analysis system
 */

class FakeReviewDetector {
    constructor() {
        this.apiBaseUrl = 'http://127.0.0.1:5000';
        this.currentMethod = 'text';
        this.analysisCount = 0;
        this.init();
    }

    init() {
        this.bindEvents();
        this.initializeUI();
        this.loadStoredStats();
    }

    bindEvents() {
        // Method selection buttons
        document.querySelectorAll('.method-btn').forEach(btn => {
            btn.addEventListener('click', (e) => this.switchMethod(e.target.closest('.method-btn')));
        });

        // Form submission
        document.getElementById('analysis-form').addEventListener('submit', (e) => this.handleFormSubmit(e));

        // Input character counting
        document.getElementById('review-text').addEventListener('input', (e) => this.updateCharCount(e.target));
        document.getElementById('batch-reviews').addEventListener('input', (e) => this.updateReviewCount(e.target));

        // Filter buttons for batch results
        document.addEventListener('click', (e) => {
            if (e.target.classList.contains('filter-btn')) {
                this.filterResults(e.target.dataset.filter);
            }
        });

        // Export buttons
        document.getElementById('export-json')?.addEventListener('click', () => this.exportResults('json'));
        document.getElementById('export-csv')?.addEventListener('click', () => this.exportResults('csv'));

        // Retry button
        document.getElementById('retry-btn')?.addEventListener('click', () => this.hideError());

        // Real-time input validation
        this.setupInputValidation();
    }

    initializeUI() {
        // Initialize tooltips, animations, etc.
        this.updateStats();
        this.setupProgressiveEnhancement();
    }

    setupInputValidation() {
        const urlInput = document.getElementById('product-url');
        const textInput = document.getElementById('review-text');
        const batchInput = document.getElementById('batch-reviews');

        urlInput.addEventListener('input', (e) => this.validateUrl(e.target));
        textInput.addEventListener('input', (e) => this.validateText(e.target));
        batchInput.addEventListener('input', (e) => this.validateBatch(e.target));
    }

    validateUrl(input) {
        const url = input.value.trim();
        const isValid = this.isValidProductUrl(url);
        
        input.classList.toggle('invalid', url && !isValid);
        input.classList.toggle('valid', url && isValid);
        
        return isValid;
    }

    validateText(input) {
        const text = input.value.trim();
        const isValid = text.length >= 10 && text.length <= 5000;
        
        input.classList.toggle('invalid', text && !isValid);
        input.classList.toggle('valid', text && isValid);
        
        return isValid;
    }

    validateBatch(input) {
        const lines = input.value.trim().split('\n').filter(line => line.trim());
        const isValid = lines.length > 0 && lines.length <= 50;
        
        input.classList.toggle('invalid', lines.length > 0 && !isValid);
        input.classList.toggle('valid', lines.length > 0 && isValid);
        
        return isValid;
    }

    isValidProductUrl(url) {
        if (!url) return false;
        
        try {
            const urlObj = new URL(url);
            const domain = urlObj.hostname.toLowerCase();
            
            // Check for supported platforms
            return domain.includes('amazon') || 
                   domain.includes('flipkart') ||
                   domain.includes('myntra') ||
                   domain.includes('ajio');
        } catch {
            return false;
        }
    }

    switchMethod(button) {
        // Update active button
        document.querySelectorAll('.method-btn').forEach(btn => btn.classList.remove('active'));
        button.classList.add('active');

        // Get method type
        this.currentMethod = button.dataset.method;

        // Hide all input sections
        document.querySelectorAll('.input-section').forEach(section => section.classList.add('hidden'));

        // Show selected input section
        document.getElementById(`${this.currentMethod}-input-section`).classList.remove('hidden');

        // Update button text
        const btnText = document.querySelector('.btn-text');
        switch(this.currentMethod) {
            case 'text':
                btnText.textContent = 'Analyze Review';
                break;
            case 'url':
                btnText.textContent = 'Analyze Product Reviews';
                break;
            case 'batch':
                btnText.textContent = 'Analyze All Reviews';
                break;
        }

        // Clear previous results
        this.hideResults();
        this.hideError();
    }

    updateCharCount(textarea) {
        const count = textarea.value.length;
        const counter = document.getElementById('char-count');
        const maxLength = 5000;
        
        counter.textContent = count;
        counter.parentElement.classList.toggle('warning', count > maxLength * 0.9);
        counter.parentElement.classList.toggle('error', count > maxLength);
    }

    updateReviewCount(textarea) {
        const lines = textarea.value.trim().split('\n').filter(line => line.trim());
        const counter = document.getElementById('review-count');
        const maxReviews = 50;
        
        counter.textContent = lines.length;
        counter.parentElement.classList.toggle('warning', lines.length > maxReviews * 0.8);
        counter.parentElement.classList.toggle('error', lines.length > maxReviews);
    }

    async handleFormSubmit(e) {
        e.preventDefault();
        
        // Validate input based on current method
        if (!this.validateCurrentInput()) {
            return;
        }

        // Show loading state
        this.setLoadingState(true);
        this.hideResults();
        this.hideError();

        const startTime = Date.now();

        try {
            let result;
            
            switch(this.currentMethod) {
                case 'text':
                    result = await this.analyzeSingleReview();
                    break;
                case 'url':
                    result = await this.analyzeProductUrl();
                    break;
                case 'batch':
                    result = await this.analyzeBatchReviews();
                    break;
            }

            const endTime = Date.now();
            const analysisTime = ((endTime - startTime) / 1000).toFixed(1);

            // Update stats
            this.updateAnalysisStats(analysisTime);

            // Show results
            this.displayResults(result, analysisTime);
            
        } catch (error) {
            console.error('Analysis error:', error);
            this.showError(error.message || 'Analysis failed. Please try again.');
        } finally {
            this.setLoadingState(false);
        }
    }

    validateCurrentInput() {
        switch(this.currentMethod) {
            case 'text':
                const text = document.getElementById('review-text').value.trim();
                if (!text) {
                    this.showError('Please enter a review to analyze.');
                    return false;
                }
                if (text.length < 10) {
                    this.showError('Review text is too short. Please enter at least 10 characters.');
                    return false;
                }
                return true;

            case 'url':
                const url = document.getElementById('product-url').value.trim();
                if (!url) {
                    this.showError('Please enter a product URL.');
                    return false;
                }
                if (!this.isValidProductUrl(url)) {
                    this.showError('Please enter a valid product URL from supported platforms.');
                    return false;
                }
                return true;

            case 'batch':
                const batchText = document.getElementById('batch-reviews').value.trim();
                const lines = batchText.split('\n').filter(line => line.trim());
                if (lines.length === 0) {
                    this.showError('Please enter at least one review.');
                    return false;
                }
                if (lines.length > 50) {
                    this.showError('Maximum 50 reviews allowed. Please reduce the number of reviews.');
                    return false;
                }
                return true;

            default:
                return false;
        }
    }

    async analyzeSingleReview() {
        const review = document.getElementById('review-text').value.trim();
        
        const response = await fetch(`${this.apiBaseUrl}/predict`, {
            method: 'POST',
            headers: {
                'Content-Type': 'application/json',
            },
            body: JSON.stringify({ review })
        });

        if (!response.ok) {
            throw new Error(`Server error: ${response.status}`);
        }

        const data = await response.json();
        
        if (data.error) {
            throw new Error(data.error);
        }

        return {
            type: 'single',
            data: data
        };
    }

    async analyzeProductUrl() {
        const url = document.getElementById('product-url').value.trim();
        

        // Real URL analysis (not simulation anymore!)
        const response = await fetch(`${this.apiBaseUrl}/analyze-product-url`, {
            method: 'POST',
            headers: {
                'Content-Type': 'application/json',
            },
            body: JSON.stringify({ url: url })
        });

        if (!response.ok) {
            const errorData = await response.json();
            throw new Error(errorData.error || `Server error: ${response.status}`);
        }

        const data = await response.json();
        
        if (data.error) {
            throw new Error(data.error);
        }

        return {
            type: 'url',
            data: data,
            url: url,
            platform: data.platform
        };
    }

    async analyzeBatchReviews() {
        const batchText = document.getElementById('batch-reviews').value.trim();
        const reviews = batchText.split('\n')
            .map(line => line.trim())
            .filter(line => line.length > 0);

        const response = await fetch(`${this.apiBaseUrl}/batch-predict`, {
            method: 'POST',
            headers: {
                'Content-Type': 'application/json',
            },
            body: JSON.stringify({ reviews })
        });

        if (!response.ok) {
            throw new Error(`Server error: ${response.status}`);
        }

        const data = await response.json();
        
        if (data.error) {
            throw new Error(data.error);
        }

        return {
            type: 'batch',
            data: data
        };
    }

    getPlatformFromUrl(url) {
        const domain = new URL(url).hostname.toLowerCase();
        
        if (domain.includes('amazon')) return 'Amazon';
        if (domain.includes('flipkart')) return 'Flipkart';
        if (domain.includes('myntra')) return 'Myntra';
        if (domain.includes('ajio')) return 'Ajio';
        
        return 'Unknown';
    }

    displayResults(result, analysisTime) {
        // Hide all result sections
        document.querySelectorAll('#results-section .result-card').forEach(card => {
            card.style.display = 'none';
        });

        // Update analysis time
        document.getElementById('analysis-time').textContent = `${analysisTime}s`;

        // Show results section
        document.getElementById('results-section').style.display = 'block';
        document.getElementById('results-section').classList.add('fade-in');

        switch(result.type) {
            case 'single':
                this.displaySingleResult(result.data);
                break;
            case 'batch':
                this.displayBatchResult(result.data);
                break;
            case 'url':
                this.displayUrlResult(result.data, result.url, result.platform);
                break;
        }

        // Scroll to results
        document.getElementById('results-section').scrollIntoView({ 
            behavior: 'smooth', 
            block: 'start' 
        });
    }

    displaySingleResult(data) {
        const resultCard = document.getElementById('single-result');
        
        // Update prediction badge
        const predictionBadge = document.getElementById('prediction-badge');
        const badgeIcon = predictionBadge.querySelector('.badge-icon');
        const badgeText = predictionBadge.querySelector('.badge-text');
        
        if (data.prediction === 'Fake') {
            predictionBadge.className = 'prediction-badge fake';
            badgeIcon.textContent = '❌';
            badgeText.textContent = 'Fake Review Detected';
        } else {
            predictionBadge.className = 'prediction-badge genuine';
            badgeIcon.textContent = '✅';
            badgeText.textContent = 'Genuine Review';
        }

        // Update confidence meter
        const confidence = data.confidence || 0;
        document.getElementById('confidence-fill').style.width = `${confidence}%`;
        document.getElementById('confidence-percentage').textContent = `${confidence}%`;
        
        // Set confidence color
        const confidenceFill = document.getElementById('confidence-fill');
        if (confidence >= 80) {
            confidenceFill.className = 'confidence-fill high';
        } else if (confidence >= 60) {
            confidenceFill.className = 'confidence-fill medium';
        } else {
            confidenceFill.className = 'confidence-fill low';
        }

        // Update recommendation
        const recommendationBadge = document.getElementById('recommendation-badge');
        const recIcon = recommendationBadge.querySelector('.rec-icon');
        const recText = recommendationBadge.querySelector('.rec-text');
        
        if (data.buy_recommendation) {
            if (data.buy_recommendation.includes('NOT')) {
                recommendationBadge.className = 'recommendation-badge danger';
                recIcon.textContent = '🚫';
                recText.textContent = 'Do Not Buy';
            } else if (data.buy_recommendation.includes('Cautious')) {
                recommendationBadge.className = 'recommendation-badge warning';
                recIcon.textContent = '⚠️';
                recText.textContent = 'Be Cautious';
            } else {
                recommendationBadge.className = 'recommendation-badge safe';
                recIcon.textContent = '✅';
                recText.textContent = 'Safe to Buy';
            }
        }

        document.getElementById('recommendation-reason').textContent = 
            data.recommendation_reason || 'Analysis completed successfully.';

        resultCard.style.display = 'block';
    }

    displayBatchResult(data) {
        const resultCard = document.getElementById('batch-result');
        const summary = data.summary;

        // Update summary stats
        document.getElementById('fake-count').textContent = summary.fake_reviews || 0;
        document.getElementById('genuine-count').textContent = summary.genuine_reviews || 0;
        document.getElementById('total-analyzed').textContent = summary.total_reviews || 0;

        // Update risk meter
        const fakePercentage = summary.fake_percentage || 0;
        document.getElementById('risk-fill').style.width = `${Math.min(fakePercentage * 2, 100)}%`;
        
        let riskLevel, riskClass;
        if (fakePercentage > 50) {
            riskLevel = 'High Risk';
            riskClass = 'high';
        } else if (fakePercentage > 25) {
            riskLevel = 'Medium Risk';
            riskClass = 'medium';
        } else {
            riskLevel = 'Low Risk';
            riskClass = 'low';
        }
        
        document.getElementById('risk-level').textContent = riskLevel;
        document.getElementById('risk-fill').className = `risk-fill ${riskClass}`;

        // Update overall recommendation
        const overallBadge = document.getElementById('overall-badge');
        const overallIcon = overallBadge.querySelector('.overall-icon');
        const overallText = overallBadge.querySelector('.overall-text');
        
        if (fakePercentage > 50) {
            overallBadge.className = 'overall-badge danger';
            overallIcon.textContent = '🚫';
            overallText.textContent = 'High Risk - Many Fake Reviews';
        } else if (fakePercentage > 25) {
            overallBadge.className = 'overall-badge warning';
            overallIcon.textContent = '⚠️';
            overallText.textContent = 'Medium Risk - Some Concerns';
        } else {
            overallBadge.className = 'overall-badge safe';
            overallIcon.textContent = '✅';
            overallText.textContent = 'Low Risk - Mostly Genuine';
        }

        // Display detailed results
        this.displayDetailedResults(data.results);

        resultCard.style.display = 'block';
    }

    displayUrlResult(data, url, platform) {
        // Set platform badge
        document.getElementById('platform-badge').textContent = platform;
        document.getElementById('product-link').href = url;

        // Use the same batch result display but with URL-specific header
        this.displayBatchResult(data);
        
        document.getElementById('url-result').style.display = 'block';
    }

    displayDetailedResults(results) {
        const resultsList = document.getElementById('detailed-results-list');
        resultsList.innerHTML = '';

        results.forEach((result, index) => {
            if (result.error) return; // Skip errored results

            const resultItem = document.createElement('div');
            resultItem.className = `result-item ${result.prediction?.toLowerCase() || 'unknown'}`;
            resultItem.dataset.filter = result.prediction?.toLowerCase() || 'unknown';

            resultItem.innerHTML = `
                <div class="result-item-header">
                    <span class="result-index">#${index + 1}</span>
                    <span class="result-prediction ${result.prediction?.toLowerCase()}">
                        ${result.prediction === 'Fake' ? '❌' : '✅'} ${result.prediction || 'Unknown'}
                    </span>
                    <span class="result-confidence">${result.confidence || 0}%</span>
                </div>
                <div class="result-item-content">
                    ${(result.review || 'No review text').substring(0, 150)}${result.review?.length > 150 ? '...' : ''}
                </div>
            `;

            resultsList.appendChild(resultItem);
        });

        // Store results for filtering and export
        this.currentResults = results;
    }

    filterResults(filter) {
        // Update active filter button
        document.querySelectorAll('.filter-btn').forEach(btn => {
            btn.classList.toggle('active', btn.dataset.filter === filter);
        });

        // Filter result items
        document.querySelectorAll('.result-item').forEach(item => {
            if (filter === 'all') {
                item.style.display = 'block';
            } else {
                item.style.display = item.dataset.filter === filter ? 'block' : 'none';
            }
        });
    }

    exportResults(format) {
        if (!this.currentResults) return;

        const data = this.currentResults.map((result, index) => ({
            index: index + 1,
            review: result.review || '',
            prediction: result.prediction || 'Unknown',
            confidence: result.confidence || 0,
            timestamp: new Date().toISOString()
        }));

        if (format === 'json') {
            this.downloadFile(
                JSON.stringify(data, null, 2),
                'fake-review-analysis.json',
                'application/json'
            );
        } else if (format === 'csv') {
            const csv = this.convertToCSV(data);
            this.downloadFile(
                csv,
                'fake-review-analysis.csv',
                'text/csv'
            );
        }
    }

    convertToCSV(data) {
        const headers = ['Index', 'Review', 'Prediction', 'Confidence', 'Timestamp'];
        const rows = data.map(item => [
            item.index,
            `"${item.review.replace(/"/g, '""')}"`, // Escape quotes
            item.prediction,
            item.confidence,
            item.timestamp
        ]);

        return [headers, ...rows].map(row => row.join(',')).join('\n');
    }

    downloadFile(content, filename, contentType) {
        const blob = new Blob([content], { type: contentType });
        const url = window.URL.createObjectURL(blob);
        const link = document.createElement('a');
        
        link.href = url;
        link.download = filename;
        document.body.appendChild(link);
        link.click();
        document.body.removeChild(link);
        window.URL.revokeObjectURL(url);
    }

    setLoadingState(isLoading) {
        const analyzeBtn = document.getElementById('analyze-btn');
        const btnText = analyzeBtn.querySelector('.btn-text');
        const btnLoading = analyzeBtn.querySelector('.btn-loading');

        if (isLoading) {
            analyzeBtn.classList.add('loading');
            analyzeBtn.disabled = true;
            btnText.style.display = 'none';
            btnLoading.style.display = 'flex';
        } else {
            analyzeBtn.classList.remove('loading');
            analyzeBtn.disabled = false;
            btnText.style.display = 'inline';
            btnLoading.style.display = 'none';
        }
    }

    showError(message) {
        document.getElementById('error-message').textContent = message;
        document.getElementById('error-section').style.display = 'block';
        document.getElementById('error-section').classList.add('fade-in');
        
        // Scroll to error
        document.getElementById('error-section').scrollIntoView({ 
            behavior: 'smooth', 
            block: 'center' 
        });
    }

    hideError() {
        document.getElementById('error-section').style.display = 'none';
    }

    hideResults() {
        document.getElementById('results-section').style.display = 'none';
    }

    updateAnalysisStats(analysisTime) {
        this.analysisCount++;
        
        // Update stats display
        document.getElementById('reviews-analyzed').textContent = 
            this.analysisCount.toLocaleString();
        document.getElementById('response-time').textContent = `${analysisTime}s`;
        
        // Store in localStorage
        localStorage.setItem('frd_analysis_count', this.analysisCount.toString());
        localStorage.setItem('frd_last_analysis_time', analysisTime);
    }

    loadStoredStats() {
        // Load stored statistics
        const storedCount = localStorage.getItem('frd_analysis_count');
        if (storedCount) {
            this.analysisCount = parseInt(storedCount);
            document.getElementById('reviews-analyzed').textContent = 
                this.analysisCount.toLocaleString();
        }

        const storedTime = localStorage.getItem('frd_last_analysis_time');
        if (storedTime) {
            document.getElementById('response-time').textContent = `${storedTime}s`;
        }
    }

    updateStats() {
        // Animate stats on load
        const stats = document.querySelectorAll('.stat-number');
        stats.forEach(stat => {
            const finalValue = stat.textContent;
            if (!isNaN(parseFloat(finalValue))) {
                this.animateNumber(stat, 0, parseFloat(finalValue), 1000);
            }
        });
    }

    animateNumber(element, start, end, duration) {
        const startTime = performance.now();
        const isPercentage = element.textContent.includes('%');
        const isTime = element.textContent.includes('s');
        
        const animate = (currentTime) => {
            const elapsed = currentTime - startTime;
            const progress = Math.min(elapsed / duration, 1);
            
            const current = start + (end - start) * this.easeOutCubic(progress);
            
            if (isPercentage) {
                element.textContent = `${current.toFixed(1)}%`;
            } else if (isTime) {
                element.textContent = `${current.toFixed(1)}s`;
            } else {
                element.textContent = Math.floor(current).toLocaleString();
            }
            
            if (progress < 1) {
                requestAnimationFrame(animate);
            }
        };
        
        requestAnimationFrame(animate);
    }

    easeOutCubic(t) {
        return 1 - Math.pow(1 - t, 3);
    }

    setupProgressiveEnhancement() {
        // Add progressive enhancement features
        
        // Keyboard shortcuts
        document.addEventListener('keydown', (e) => {
            if (e.ctrlKey || e.metaKey) {
                switch(e.key) {
                    case 'Enter':
                        e.preventDefault();
                        document.getElementById('analysis-form').dispatchEvent(new Event('submit'));
                        break;
                    case '1':
                        e.preventDefault();
                        document.querySelector('[data-method="text"]').click();
                        break;
                    case '2':
                        e.preventDefault();
                        document.querySelector('[data-method="url"]').click();
                        break;
                    case '3':
                        e.preventDefault();
                        document.querySelector('[data-method="batch"]').click();
                        break;
                }
            }
        });

        // Add tooltips for better UX
        this.addTooltips();
        
        // Add auto-save for drafts
        this.setupAutoSave();
    }

    addTooltips() {
        const tooltips = [
            { selector: '#text-method', text: 'Analyze a single review text (Ctrl+1)' },
            { selector: '#url-method', text: 'Analyze all reviews from a product URL (Ctrl+2)' },
            { selector: '#batch-method', text: 'Analyze multiple reviews at once (Ctrl+3)' },
            { selector: '#analyze-btn', text: 'Start AI analysis (Ctrl+Enter)' }
        ];

        tooltips.forEach(tooltip => {
            const element = document.querySelector(tooltip.selector);
            if (element) {
                element.title = tooltip.text;
            }
        });
    }

    setupAutoSave() {
        // Auto-save drafts every 30 seconds
        setInterval(() => {
            const textContent = document.getElementById('review-text').value;
            const urlContent = document.getElementById('product-url').value;
            const batchContent = document.getElementById('batch-reviews').value;

            if (textContent) localStorage.setItem('frd_draft_text', textContent);
            if (urlContent) localStorage.setItem('frd_draft_url', urlContent);
            if (batchContent) localStorage.setItem('frd_draft_batch', batchContent);
        }, 30000);

        // Restore drafts on load
        const draftText = localStorage.getItem('frd_draft_text');
        const draftUrl = localStorage.getItem('frd_draft_url');
        const draftBatch = localStorage.getItem('frd_draft_batch');

        if (draftText) document.getElementById('review-text').value = draftText;
        if (draftUrl) document.getElementById('product-url').value = draftUrl;
        if (draftBatch) document.getElementById('batch-reviews').value = draftBatch;
    }
}

// Initialize the application
document.addEventListener('DOMContentLoaded', function() {
    // Create global instance
    window.fakeReviewDetector = new FakeReviewDetector();
    
    // Add smooth scrolling for all anchor links
    document.querySelectorAll('a[href^="#"]').forEach(anchor => {
        anchor.addEventListener('click', function (e) {
            e.preventDefault();
            const target = document.querySelector(this.getAttribute('href'));
            if (target) {
                target.scrollIntoView({
                    behavior: 'smooth',
                    block: 'start'
                });
            }
        });
    });

    // Add intersection observer for animations
    const observerOptions = {
        threshold: 0.1,
        rootMargin: '0px 0px -50px 0px'
    };

    const observer = new IntersectionObserver((entries) => {
        entries.forEach(entry => {
            if (entry.isIntersecting) {
                entry.target.classList.add('animate-in');
            }
        });
    }, observerOptions);

    // Observe elements for animation
    document.querySelectorAll('.feature-card, .stat-card').forEach(el => {
        observer.observe(el);
    });
});

// Export for potential external use
if (typeof module !== 'undefined' && module.exports) {
    module.exports = FakeReviewDetector;
}