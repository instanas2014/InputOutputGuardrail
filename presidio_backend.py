import asyncio
import time
from typing import Dict, List, Tuple, Optional
import re
import hashlib
from concurrent.futures import ThreadPoolExecutor
from PIL import Image

# Core libraries for PII detection
try:
    from presidio_analyzer import AnalyzerEngine
    from presidio_anonymizer import AnonymizerEngine
    from presidio_image_redactor import ImageRedactorEngine
    PRESIDIO_AVAILABLE = True
except ImportError:
    PRESIDIO_AVAILABLE = False

try:
    import spacy
    # Try to load English model
    try:
        nlp = spacy.load("en_core_web_sm")
        SPACY_AVAILABLE = True
    except OSError:
        SPACY_AVAILABLE = False
except ImportError:
    SPACY_AVAILABLE = False

class DataSanitizer:
    """High-performance data sanitization engine with multiple detection methods"""
    
    def __init__(self):
        self.thread_pool = ThreadPoolExecutor(max_workers=4)
        self.cache = {}
        self.anonymization_map = {}
        
        # Initialize engines if available
        if PRESIDIO_AVAILABLE:
            self.analyzer = AnalyzerEngine()
            self.anonymizer = AnonymizerEngine()
            self.image_redactor = ImageRedactorEngine()
        
        # Fallback regex patterns for common PII
        self.pii_patterns = {
            'email': r'\b[A-Za-z0-9._%+-]+@[A-Za-z0-9.-]+\.[A-Z|a-z]{2,}\b',
            'phone': r'(?:\+?1[-.\ s]?)?\(?[0-9]{3}\)?[-.\ s]?[0-9]{3}[-.\ s]?[0-9]{4}',
            'ssn': r'\b\d{3}-?\d{2}-?\d{4}\b',
            'credit_card': r'\b(?:\d{4}[-\s]?){3}\d{4}\b',
            'ip_address': r'\b(?:[0-9]{1,3}\.){3}[0-9]{1,3}\b',
        }
        
    def _generate_cache_key(self, text: str) -> str:
        """Generate cache key for input text"""
        return hashlib.md5(text.encode()).hexdigest()
    
    def _create_anonymous_replacement(self, entity_type: str, original: str) -> str:
        """Create consistent anonymous replacement for detected entities"""
        key = f"{entity_type}_{original}"
        if key not in self.anonymization_map:
            if entity_type.lower() == 'person':
                self.anonymization_map[key] = f"[PERSON_{len([k for k in self.anonymization_map if 'PERSON' in k]) + 1}]"
            elif entity_type.lower() == 'email':
                self.anonymization_map[key] = f"[EMAIL_{len([k for k in self.anonymization_map if 'EMAIL' in k]) + 1}]"
            elif entity_type.lower() == 'phone':
                self.anonymization_map[key] = f"[PHONE_{len([k for k in self.anonymization_map if 'PHONE' in k]) + 1}]"
            else:
                self.anonymization_map[key] = f"[{entity_type.upper()}_{len([k for k in self.anonymization_map if entity_type.upper() in k]) + 1}]"
        return self.anonymization_map[key]
    
    async def analyze_text_presidio(self, text: str) -> List[Dict]:
        """Analyze text using Presidio (if available)"""
        if not PRESIDIO_AVAILABLE:
            return []
        
        loop = asyncio.get_event_loop()
        try:
            results = await loop.run_in_executor(
                self.thread_pool,
                lambda: self.analyzer.analyze(text=text, language='en')
            )
            return [
                {
                    'entity_type': result.entity_type,
                    'start': result.start,
                    'end': result.end,
                    'text': text[result.start:result.end],
                    'confidence': result.score,
                    'detection_method': 'presidio'
                }
                for result in results
            ]
        except Exception as e:
            print(f"Presidio analysis error: {e}")
            return []
    
    async def analyze_text_spacy(self, text: str) -> List[Dict]:
        """Analyze text using spaCy NER (if available)"""
        if not SPACY_AVAILABLE:
            return []
        
        loop = asyncio.get_event_loop()
        try:
            doc = await loop.run_in_executor(self.thread_pool, nlp, text)
            return [
                {
                    'entity_type': ent.label_,
                    'start': ent.start_char,
                    'end': ent.end_char,
                    'text': ent.text,
                    'confidence': 0.8,  # spaCy doesn't provide confidence scores
                    'detection_method': 'spacy'
                }
                for ent in doc.ents
                if ent.label_ in ['PERSON', 'ORG', 'GPE', 'DATE', 'MONEY']
            ]
        except Exception as e:
            print(f"spaCy analysis error: {e}")
            return []
    
    async def analyze_text_regex(self, text: str) -> List[Dict]:
        """Analyze text using regex patterns"""
        entities = []
        for entity_type, pattern in self.pii_patterns.items():
            matches = re.finditer(pattern, text, re.IGNORECASE)
            for match in matches:
                entities.append({
                    'entity_type': entity_type,
                    'start': match.start(),
                    'end': match.end(),
                    'text': match.group(),
                    'confidence': 0.7,
                    'detection_method': 'regex'
                })
        return entities
    
    async def analyze_text(self, text: str) -> Dict:
        """Comprehensive text analysis using all available methods"""
        cache_key = self._generate_cache_key(text)
        if cache_key in self.cache:
            return self.cache[cache_key]
        
        start_time = time.time()
        
        # Run all analyzers concurrently
        tasks = [
            self.analyze_text_presidio(text),
            self.analyze_text_spacy(text),
            self.analyze_text_regex(text)
        ]
        
        results = await asyncio.gather(*tasks, return_exceptions=True)
        
        # Combine results and deduplicate
        all_entities = []
        for result in results:
            if isinstance(result, list):
                all_entities.extend(result)
        
        # Remove duplicates based on text overlap
        unique_entities = self._deduplicate_entities(all_entities)
        
        # Calculate risk score
        risk_score = self._calculate_risk_score(unique_entities)
        
        analysis_result = {
            'entities': unique_entities,
            'risk_score': risk_score,
            'processing_time': time.time() - start_time,
            'methods_used': [task.__name__.split('_')[-1] for task in [self.analyze_text_presidio, self.analyze_text_spacy, self.analyze_text_regex]]
        }
        
        # Cache result
        self.cache[cache_key] = analysis_result
        return analysis_result
    
    def _deduplicate_entities(self, entities: List[Dict]) -> List[Dict]:
        """Remove duplicate entities based on text overlap"""
        if not entities:
            return []
        
        # Sort by start position
        entities = sorted(entities, key=lambda x: x['start'])
        
        deduplicated = []
        for entity in entities:
            # Check for overlap with existing entities
            overlaps = False
            for existing in deduplicated:
                if (entity['start'] < existing['end'] and entity['end'] > existing['start']):
                    # Keep the one with higher confidence
                    if entity['confidence'] > existing['confidence']:
                        deduplicated.remove(existing)
                        deduplicated.append(entity)
                    overlaps = True
                    break
            
            if not overlaps:
                deduplicated.append(entity)
        
        return sorted(deduplicated, key=lambda x: x['start'])
    
    def _calculate_risk_score(self, entities: List[Dict]) -> float:
        """Calculate risk score based on detected entities"""
        if not entities:
            return 0.0
        
        risk_weights = {
            'ssn': 10.0,
            'credit_card': 9.0,
            'phone': 7.0,
            'email': 6.0,
            'person': 5.0,
            'date': 3.0,
            'ip_address': 4.0,
            'org': 2.0
        }
        
        total_risk = 0.0
        for entity in entities:
            weight = risk_weights.get(entity['entity_type'].lower(), 1.0)
            total_risk += weight * entity['confidence']
        
        # Normalize to 0-100 scale
        return min(100.0, total_risk)
    
    def anonymize_text(self, text: str, entities: List[Dict], method: str = 'replace') -> Tuple[str, Dict]:
        """Anonymize text based on detected entities"""
        if not entities:
            return text, {}
        
        # Sort entities by start position in reverse order to maintain indices
        entities = sorted(entities, key=lambda x: x['start'], reverse=True)
        
        anonymized_text = text
        anonymization_mapping = {}
        
        for entity in entities:
            original = entity['text']
            if method == 'replace':
                replacement = self._create_anonymous_replacement(entity['entity_type'], original)
            elif method == 'mask':
                replacement = '*' * len(original)
            elif method == 'redact':
                replacement = '[REDACTED]'
            else:
                replacement = original  # No anonymization
            
            anonymized_text = (
                anonymized_text[:entity['start']] + 
                replacement + 
                anonymized_text[entity['end']:]
            )
            
            anonymization_mapping[original] = replacement
        
        return anonymized_text, anonymization_mapping
    
    async def analyze_image(self, image: Image.Image) -> Dict:
        """Analyze image for sensitive content (placeholder implementation)"""
        # This is a simplified implementation
        # In production, you would use OCR + PII detection or specialized image analysis
        
        start_time = time.time()
        
        # Convert image to text using OCR (placeholder)
        # In real implementation, you'd use pytesseract or similar
        extracted_text = "Sample extracted text from image"  # Placeholder
        
        # Analyze extracted text
        text_analysis = await self.analyze_text(extracted_text)
        
        return {
            'extracted_text': extracted_text,
            'text_analysis': text_analysis,
            'processing_time': time.time() - start_time,
            'image_size': image.size
        }

# Availability flags for import
__all__ = ['DataSanitizer', 'PRESIDIO_AVAILABLE', 'SPACY_AVAILABLE']