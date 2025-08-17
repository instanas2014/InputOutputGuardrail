import warnings
warnings.filterwarnings("ignore")

from typing import Optional, List, Dict, Any
import re
import time
from datetime import datetime
import os
from dotenv import load_dotenv

# Load environment variables
load_dotenv()

# Guardrails imports
from guardrails import Guard, OnFailAction, install
from guardrails.validator_base import (
    FailResult,
    PassResult,
    ValidationResult,
    Validator,
    register_validator,
)

# ML and NLP imports
try:
    from transformers import pipeline, AutoTokenizer, AutoModelForTokenClassification
    from sentence_transformers import SentenceTransformer
    from sklearn.metrics.pairwise import cosine_similarity
    import numpy as np
    import nltk
    TRANSFORMERS_AVAILABLE = True
except ImportError:
    TRANSFORMERS_AVAILABLE = False

# Presidio imports for PII detection
try:
    from presidio_analyzer import AnalyzerEngine
    from presidio_anonymizer import AnonymizerEngine
    PRESIDIO_AVAILABLE = True
except ImportError:
    PRESIDIO_AVAILABLE = False

# Add Hub validator imports at the top
from guardrails.hub import ToxicLanguage

class GuardrailBackend:
    """Comprehensive Guardrail Backend for AI Safety"""
    
    def __init__(self):
        # Check for required environment variables
        self.openai_api_key = os.getenv('OPENAI_API_KEY')
        if not self.openai_api_key:
            print("Warning: OPENAI_API_KEY not found in environment variables")
        
        self.guards = {}
        self.metrics = {
            'total_validations': 0,
            'failed_validations': 0,
            'validation_times': [],
            'guardrail_triggers': {
                'ethical': 0,
                'legal': 0,
                'technical': 0,
                'data_compliance': 0,
                'brand': 0
            }
        }
        self._initialize_models()
        self._setup_guards()
    
    def _initialize_models(self):
        """Initialize ML models for guardrails"""
        if not TRANSFORMERS_AVAILABLE:
            print("Warning: Transformers not available. Some guardrails will be limited.")
            return
            
        try:
            # Remove toxicity_classifier since we're using Hub validator
            # self.toxicity_classifier = pipeline(...) # REMOVE THIS
            
            # Topic classification for legal/brand guardrails
            self.topic_classifier = pipeline(
                "zero-shot-classification",
                model='facebook/bart-large-mnli',
                hypothesis_template="This sentence contains discussions of: {}.",
                multi_label=True,
                device=-1
            )
            
            # NER for competitor detection
            tokenizer = AutoTokenizer.from_pretrained("dslim/bert-base-NER")
            model = AutoModelForTokenClassification.from_pretrained("dslim/bert-base-NER")
            self.ner_pipeline = pipeline("ner", model=model, tokenizer=tokenizer, device=-1)
            
            # Sentence transformer for semantic similarity
            self.sentence_transformer = SentenceTransformer('all-MiniLM-L6-v2')
            
            print("✓ ML models initialized successfully")
            
        except Exception as e:
            print(f"Warning: Could not initialize some ML models: {e}")
            # Remove toxicity_classifier reference
            self.topic_classifier = None
            self.ner_pipeline = None
            self.sentence_transformer = None
    
    def _setup_guards(self):
        """Setup all guardrail guards"""
        # 1. Ethical Guardrail - Using only ToxicLanguage validator
        self.guards['ethical'] = Guard(name='ethical_guard').use(
            ToxicLanguage(
                threshold=0.5,
                validation_method="sentence",
                on_fail=OnFailAction.EXCEPTION
            )
        )
        
        # 2. Legal/Regulatory Compliance Guardrail
        self.guards['legal'] = Guard(name='legal_guard').use(
            LegalComplianceValidator(on_fail=OnFailAction.EXCEPTION)
        )
        
        # 3. Technical Guardrail (Hallucination)
        self.guards['technical'] = Guard(name='technical_guard').use(
            HallucinationValidator(on_fail=OnFailAction.EXCEPTION)
        )
        
        # 4. Data Compliance Guardrail (PII)
        self.guards['data_compliance'] = Guard(name='data_compliance_guard').use(
            PIIValidator(on_fail=OnFailAction.EXCEPTION)
        )
        
        # 5. Brand Guardrail (Competitor mentions)
        self.guards['brand'] = Guard(name='brand_guard').use(
            CompetitorValidator(
                competitors=["Pizza Hut", "Domino's", "Papa John's", "Little Caesars", "Pizza by Alfredo"],
                on_fail=OnFailAction.EXCEPTION
            )
        )
        
        print("✓ All guardrails initialized")
    
    def validate_content(self, content: str, guardrail_types: List[str] = None) -> Dict[str, Any]:
        """Validate content against specified guardrails"""
        if guardrail_types is None:
            guardrail_types = ['ethical', 'legal', 'technical', 'data_compliance', 'brand']
        
        start_time = time.time()
        results = {
            'passed': True,
            'violations': [],
            'guardrail_results': {},
            'processing_time': 0
        }
        
        self.metrics['total_validations'] += 1
        
        for guardrail_type in guardrail_types:
            if guardrail_type not in self.guards:
                continue
                
            try:
                guard_result = self.guards[guardrail_type].validate(content)
                results['guardrail_results'][guardrail_type] = {
                    'passed': True,
                    'message': 'Validation passed'
                }
            except Exception as e:
                results['passed'] = False
                results['violations'].append({
                    'type': guardrail_type,
                    'message': str(e)
                })
                results['guardrail_results'][guardrail_type] = {
                    'passed': False,
                    'message': str(e)
                }
                self.metrics['guardrail_triggers'][guardrail_type] += 1
        
        if not results['passed']:
            self.metrics['failed_validations'] += 1
        
        processing_time = time.time() - start_time
        results['processing_time'] = processing_time
        self.metrics['validation_times'].append(processing_time)
        
        return results
    
    def get_metrics(self) -> Dict[str, Any]:
        """Get guardrail performance metrics"""
        avg_time = np.mean(self.metrics['validation_times']) if self.metrics['validation_times'] else 0
        
        return {
            'total_validations': self.metrics['total_validations'],
            'failed_validations': self.metrics['failed_validations'],
            'success_rate': (self.metrics['total_validations'] - self.metrics['failed_validations']) / max(1, self.metrics['total_validations']) * 100,
            'average_processing_time': avg_time,
            'guardrail_triggers': self.metrics['guardrail_triggers'],
            'models_available': TRANSFORMERS_AVAILABLE and PRESIDIO_AVAILABLE
        }

# Custom Validators

# REMOVE the entire EthicalValidator class (lines 191-238)
# @register_validator(name="ethical_validator", data_type="string")
# class EthicalValidator(Validator):
#     """Validates content for ethical concerns: toxicity, bias, harmful content"""
#     
#     def __init__(self, toxicity_threshold: float = 0.7, **kwargs):
#         self.toxicity_threshold = toxicity_threshold
#         super().__init__(**kwargs)
#     
#     def validate(self, value: str, metadata: Optional[Dict] = None) -> ValidationResult:
#         issues = []
#         
#         # Check for explicit harmful patterns
#         harmful_patterns = [
#             r'\b(kill|murder|suicide|harm)\s+(yourself|others)\b',
#             r'\b(hate|discriminat\w*)\s+(against|towards)\b',
#             r'\b(racist|sexist|homophobic|transphobic)\b'
#         ]
#         
#         for pattern in harmful_patterns:
#             if re.search(pattern, value.lower()):
#                 issues.append(f"Contains potentially harmful language: {pattern}")
#         
#         # Use toxicity classifier if available
#         backend = GuardrailBackend()
#         if hasattr(backend, 'toxicity_classifier') and backend.toxicity_classifier:
#             try:
#                 result = backend.toxicity_classifier(value)
#                 if result[0]['label'] == 'TOXIC' and result[0]['score'] > self.toxicity_threshold:
#                     issues.append(f"High toxicity detected (score: {result[0]['score']:.2f})")
#             except Exception as e:
#                 pass  # Fallback to pattern matching
#         
#         if issues:
#             return FailResult(error_message=f"Ethical violations detected: {'; '.join(issues)}")
#         
#         return PassResult()

@register_validator(name="legal_compliance_validator", data_type="string")
class LegalComplianceValidator(Validator):
    """Validates content for legal and regulatory compliance"""
    
    def validate(self, value: str, metadata: Optional[Dict] = None) -> ValidationResult:
        issues = []
        
        # Check for potential legal issues
        legal_patterns = [
            (r'\b(copyright|©)\s+\d{4}\b', "Potential copyright infringement"),
            (r'\b(trademark|™|®)\b', "Trademark usage detected"),
            (r'\b(medical advice|diagnosis|treatment)\b', "Medical advice detected - may require disclaimer"),
            (r'\b(financial advice|investment|trading)\b', "Financial advice detected - may require disclaimer"),
            (r'\b(illegal|unlawful|criminal)\s+(activity|behavior)\b', "References to illegal activities")
        ]
        
        for pattern, description in legal_patterns:
            if re.search(pattern, value.lower()):
                issues.append(description)
        
        # Check for regulated topics
        regulated_topics = ['medical advice', 'financial advice', 'legal advice', 'pharmaceutical']
        backend = GuardrailBackend()
        if hasattr(backend, 'topic_classifier') and backend.topic_classifier:
            try:
                result = backend.topic_classifier(value, regulated_topics)
                high_confidence_topics = [
                    topic for topic, score in zip(result['labels'], result['scores']) 
                    if score > 0.8
                ]
                if high_confidence_topics:
                    issues.append(f"Regulated topics detected: {', '.join(high_confidence_topics)}")
            except Exception as e:
                pass
        
        if issues:
            return FailResult(error_message=f"Legal compliance issues: {'; '.join(issues)}")
        
        return PassResult()

@register_validator(name="hallucination_validator", data_type="string")
class HallucinationValidator(Validator):
    """Validates content for potential hallucinations using source verification"""
    
    def __init__(self, sources: Optional[List[str]] = None, similarity_threshold: float = 0.8, **kwargs):
        self.sources = sources or [
            "Alfredo's Pizza Cafe serves authentic Italian pizzas with fresh ingredients.",
            "We offer delivery and pickup services with average delivery time of 30-45 minutes.",
            "Our menu includes pizzas, salads, appetizers, and desserts.",
            "We accept online orders through our website and mobile app."
        ]
        self.similarity_threshold = similarity_threshold
        super().__init__(**kwargs)
    
    def validate(self, value: str, metadata: Optional[Dict] = None) -> ValidationResult:
        # Simple fact-checking against known sources
        if not TRANSFORMERS_AVAILABLE:
            # Fallback to basic keyword matching
            return self._basic_fact_check(value)
        
        try:
            backend = GuardrailBackend()
            if not hasattr(backend, 'sentence_transformer') or not backend.sentence_transformer:
                return self._basic_fact_check(value)
            
            # Split into sentences
            sentences = nltk.sent_tokenize(value) if 'nltk' in globals() else [value]
            
            suspicious_sentences = []
            for sentence in sentences:
                if not self._verify_against_sources(sentence, backend.sentence_transformer):
                    suspicious_sentences.append(sentence)
            
            if suspicious_sentences:
                return FailResult(
                    error_message=f"Potential hallucination detected in: {'; '.join(suspicious_sentences[:2])}..."
                )
            
        except Exception as e:
            return self._basic_fact_check(value)
        
        return PassResult()
    
    def _basic_fact_check(self, value: str) -> ValidationResult:
        """Basic fact checking using keyword patterns"""
        suspicious_patterns = [
            r'\b(definitely|certainly|absolutely)\s+(true|false|correct)\b',
            r'\b(studies show|research proves|scientists say)\b',
            r'\b(\d+)%\s+(of people|accuracy|success rate)\b'
        ]
        
        for pattern in suspicious_patterns:
            if re.search(pattern, value.lower()):
                return FailResult(error_message="Potentially unverified claims detected")
        
        return PassResult()
    
    def _verify_against_sources(self, sentence: str, model) -> bool:
        """Verify sentence against known sources using semantic similarity"""
        try:
            sentence_embedding = model.encode([sentence])
            source_embeddings = model.encode(self.sources)
            
            similarities = cosine_similarity(sentence_embedding, source_embeddings)[0]
            max_similarity = np.max(similarities)
            
            return max_similarity > self.similarity_threshold
        except:
            return True  # Default to pass if verification fails

@register_validator(name="pii_validator", data_type="string")
class PIIValidator(Validator):
    """Validates content for PII leakage - prioritized over presidio_backend"""
    
    def __init__(self, **kwargs):
        super().__init__(**kwargs)
        if PRESIDIO_AVAILABLE:
            self.analyzer = AnalyzerEngine()
        else:
            self.analyzer = None
    
    def validate(self, value: str, metadata: Optional[Dict] = None) -> ValidationResult:
        pii_detected = []
        
        # Pattern-based PII detection
        pii_patterns = {
            'phone': r'\b\d{3}[-.]?\d{3}[-.]?\d{4}\b',
            'email': r'\b[A-Za-z0-9._%+-]+@[A-Za-z0-9.-]+\.[A-Z|a-z]{2,}\b',
            'ssn': r'\b\d{3}-\d{2}-\d{4}\b',
            'credit_card': r'\b\d{4}[\s-]?\d{4}[\s-]?\d{4}[\s-]?\d{4}\b',
            'name_pattern': r'\bmy name is\s+([A-Z][a-z]+\s+[A-Z][a-z]+)\b'
        }
        
        for pii_type, pattern in pii_patterns.items():
            matches = re.findall(pattern, value, re.IGNORECASE)
            if matches:
                pii_detected.append(f"{pii_type}: {len(matches)} instances")
        
        # Use Presidio if available
        if self.analyzer:
            try:
                results = self.analyzer.analyze(text=value, language='en')
                for result in results:
                    pii_detected.append(f"{result.entity_type} (confidence: {result.score:.2f})")
            except Exception as e:
                pass  # Fallback to pattern matching
        
        if pii_detected:
            return FailResult(error_message=f"PII detected: {'; '.join(pii_detected)}")
        
        return PassResult()

@register_validator(name="competitor_validator", data_type="string")
class CompetitorValidator(Validator):
    """Validates content for competitor mentions"""
    
    def __init__(self, competitors: List[str] = None, **kwargs):
        self.competitors = competitors or [
            "Pizza Hut", "Domino's", "Papa John's", "Little Caesars", 
            "Pizza by Alfredo", "Papa Murphy's", "Casey's"
        ]
        super().__init__(**kwargs)
    
    def validate(self, value: str, metadata: Optional[Dict] = None) -> ValidationResult:
        mentioned_competitors = []
        
        # Direct name matching
        for competitor in self.competitors:
            if competitor.lower() in value.lower():
                mentioned_competitors.append(competitor)
        
        # Pattern-based detection for variations
        competitor_patterns = [
            r'\b(pizza\s+hut|dominos?|papa\s+johns?)\b',
            r'\b(little\s+caesars?|pizza\s+by\s+alfredo)\b',
            r'\b(other\s+pizza\s+(place|restaurant|chain))\b'
        ]
        
        for pattern in competitor_patterns:
            if re.search(pattern, value.lower()):
                mentioned_competitors.append("competitor reference")
        
        if mentioned_competitors:
            return FailResult(
                error_message=f"Competitor mentions detected: {', '.join(set(mentioned_competitors))}"
            )
        
        return PassResult()

# Global instance
guardrail_backend = GuardrailBackend()

# REMOVE the duplicate guard setup at the end (lines 423-433)
# Remove these lines:
# self.ethical_guard = Guard().use_many(
#     ToxicLanguage(...),
#     BiasCheck(...)
# )