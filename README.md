# GenAI Data Sanitization Demo 🔒

A comprehensive Streamlit demo for detecting and sanitizing Personally Identifiable Information (PII) in text data, with advanced AI guardrails for content validation.

## Features

- **Text Analysis**: Advanced PII detection using Presidio, spaCy, and regex patterns
- **AI Guardrails**: Multi-layered content validation including:
  - Ethical compliance (toxic language detection)
  - Legal compliance (medical/legal advice detection)
  - Technical validation (hallucination detection)
  - Data compliance (PII detection)
  - Brand protection (competitor mention detection)
- **Performance Metrics**: Real-time analytics and processing statistics
- **Multiple Detection Methods**: Presidio NLP, spaCy NER, and fallback regex patterns

## Prerequisites

- Python 3.8 or higher
- pip package manager

## Installation

### 1. Clone or Download the Project
```bash
git clone <repository-url>
cd InputOutputGuardrail
```

### 2. Install Required Dependencies
```bash
# Core dependencies
pip install streamlit presidio-analyzer presidio-anonymizer
pip install spacy transformers sentence-transformers
pip install scikit-learn numpy pandas python-dotenv
pip install guardrails-ai nltk

# Download spaCy English model
python -m spacy download en_core_web_sm

# Download NLTK data (required for guardrails)
python -c "import nltk; nltk.download('punkt_tab'); nltk.download('popular')"
```

### 3. Environment Setup
Create a `.env` file in the project root:
```bash
# Optional: OpenAI API key for advanced LLM features
OPENAI_API_KEY=your_openai_api_key_here
```

## Running the Demo

### Start the Streamlit Application
```bash
streamlit run app.py
```

The application will open in your default web browser at `http://localhost:8501`

## Usage Guide

### 1. Text Analysis Tab
- Enter text in the input area
- Select detection methods (Presidio, spaCy, Regex)
- Choose anonymization method (Replace, Redact, Hash)
- Click "Analyze Text" to process
- View detected PII entities and anonymized results

### 2. Performance Metrics Tab
- Monitor real-time processing statistics
- View detection method performance
- Analyze processing times and accuracy

### 3. AI Guardrails Demo Tab
- Select guardrail types to enable
- Adjust sensitivity levels
- Use quick test examples or enter custom content
- Run guardrail analysis to detect policy violations

## Quick Test Examples

The application includes pre-built examples for testing:

- **🤖 Ethical Test**: Toxic language detection
- **⚖️ Legal Test**: Medical advice detection
- **🔍 Technical Test**: Hallucination detection
- **🔒 PII Test**: Personal information detection
- **🏢 Brand Test**: Competitor mention detection


## Project Structure

InputOutputGuardrail/
├── app.py                 # Main Streamlit application
├── presidio_backend.py    # PII detection engine
├── guardrail_backend.py   # AI guardrails implementation
├── guardrail_server/
│   └── config.py         # Configuration settings
├── .env                  # Environment variables (create this)
└── README.md             # Project documentation


## Dependencies Overview

### Core Libraries
- **Streamlit**: Web application framework
- **Presidio**: Microsoft's PII detection library
- **spaCy**: Natural language processing
- **Guardrails AI**: Content validation framework

### Optional Dependencies
- **OpenAI API**: For advanced LLM-based features
- **Transformers**: For additional NLP models
- **NLTK**: Natural language toolkit

## Troubleshooting

### Common Issues

1. **"Resource punkt_tab not found"**
   ```bash
   python -c "import nltk; nltk.download('punkt_tab')"
   python -c "import nltk; nltk.download('popular')"
   ```

2. **"Model 'en_core_web_sm' not found"**
   ```bash
   python -m spacy download en_core_web_sm
   ```

3. **"No module named 'transformers'"**
   ```bash
   pip install transformers
   ```
4. **"OpenAI API key not found"**
- Add your OpenAI API key to the .env file

## Features in Detail
### PII Detection Methods
1. Presidio : Advanced NLP-based detection
2. spaCy : Named Entity Recognition
3. Regex : Pattern-based fallback detection
### Supported PII Types
- Email addresses
- Phone numbers
- Social Security Numbers (SSN)
- Credit card numbers
- IP addresses
- Person names
- And more...
### AI Guardrails
- Ethical : Detects harmful, toxic, or inappropriate content
- Legal : Identifies potential legal issues (medical/legal advice)
- Technical : Validates factual accuracy and detects hallucinations
- Data Compliance : Ensures PII protection
- Brand Protection : Monitors competitor mentions
