import streamlit as st
import asyncio
import time
from typing import Dict, List, Tuple, Optional
import re
import hashlib
import json
from datetime import datetime
from concurrent.futures import ThreadPoolExecutor
import threading
import pandas as pd
# Comment out PIL Image import
# from PIL import Image
import os
from dotenv import load_dotenv

# Load environment variables
load_dotenv()

# Import backend functionality
from presidio_backend import DataSanitizer, PRESIDIO_AVAILABLE, SPACY_AVAILABLE, VLM_AVAILABLE

# Import new guardrail backend (prioritized for data compliance)
from guardrail_backend import guardrail_backend

# Configuration
st.set_page_config(
    page_title="GenAI Data Sanitization Demo",
    page_icon="🔒",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Check for OpenAI API key
if not os.getenv('OPENAI_API_KEY'):
    st.warning("⚠️ OpenAI API key not found. Please set OPENAI_API_KEY in your .env file for LLM-based features.")

# Display warnings for missing dependencies
if not PRESIDIO_AVAILABLE:
    st.warning("⚠️ Presidio not installed. Using fallback regex patterns.")

if not SPACY_AVAILABLE:
    st.warning("⚠️ spaCy English model not found. Install with: python -m spacy download en_core_web_sm")

# Comment out VLM import from presidio_backend
from presidio_backend import DataSanitizer, PRESIDIO_AVAILABLE, SPACY_AVAILABLE  # , VLM_AVAILABLE

# Initialize sanitizer
@st.cache_resource
def get_sanitizer():
    """Get or create DataSanitizer instance with caching"""
    if 'sanitizer' not in st.session_state:
        st.session_state.sanitizer = DataSanitizer()
    return st.session_state.sanitizer

sanitizer = get_sanitizer()

# Streamlit UI
def main():
    st.title("🔒 GenAI Data Sanitization Demo")
    st.markdown("""
    This demo showcases **data sanitization** before sending data to GenAI services.
    Features include PII detection, anonymization options, and performance optimization for scale.
    """)
    
    # Sidebar configuration
    with st.sidebar:
        st.header("⚙️ Configuration")
        
        detection_methods = st.multiselect(
            "Detection Methods",
            ["Presidio", "spaCy", "Regex"],
            default=["Regex"] + (["Presidio"] if PRESIDIO_AVAILABLE else []) + (["spaCy"] if SPACY_AVAILABLE else [])
        )
        
        anonymization_method = st.selectbox(
            "Anonymization Method",
            ["replace", "mask", "redact", "none"]
        )
        
        risk_threshold = st.slider(
            "Risk Threshold",
            0.0, 100.0, 50.0,
            help="Minimum risk score to trigger anonymization prompt"
        )
        
        st.markdown("---")
        st.markdown("**System Status**")
        st.success(f"✅ Presidio: {'Available' if PRESIDIO_AVAILABLE else 'Not Available'}")
        st.success(f"✅ spaCy: {'Available' if SPACY_AVAILABLE else 'Not Available'}")
        # Comment out VLM status in sidebar
        # st.success(f"✅ VLM: {'Available' if VLM_AVAILABLE else 'Not Available'}")
        
        # Remove Image Analysis tab
    tab1,  tab4 , tab3 = st.tabs(["Text Analysis", "🛡️ Guardrail Demo",  "Performance Metrics"])
        
    with tab1:
        st.header("Text Input Sanitization")
        
        # Text input
        user_input = st.text_area(
            "Enter text to analyze:",
            placeholder="Type or paste your text here...\n\nExample: Contact John Doe at john.doe@email.com or call 555-123-4567",
            height=150
        )
        
        col1, col2 = st.columns([1, 1])
        
        with col1:
            analyze_btn = st.button("🔍 Analyze Text", type="primary")
        
        with col2:
            if st.button("🗑️ Clear Results"):
                st.rerun()
        
        if analyze_btn and user_input:
            with st.spinner("Analyzing text for sensitive data..."):
                # Run analysis
                analysis_result = asyncio.run(sanitizer.analyze_text(user_input))
                
                # Store in session state
                st.session_state.analysis_result = analysis_result
                st.session_state.original_text = user_input
        
        # Display results if available
        if hasattr(st.session_state, 'analysis_result'):
            result = st.session_state.analysis_result
            
            # Risk assessment
            risk_score = result['risk_score']
            if risk_score >= risk_threshold:
                st.error(f"⚠️ **High Risk Detected** (Score: {risk_score:.1f}/100)")
            elif risk_score > 20:
                st.warning(f"⚠️ **Medium Risk** (Score: {risk_score:.1f}/100)")
            else:
                st.success(f"✅ **Low Risk** (Score: {risk_score:.1f}/100)")
            
            # Performance metrics
            st.info(f"⚡ Analysis completed in {result['processing_time']:.3f} seconds")
            
            # Detected entities
            if result['entities']:
                st.subheader("🔍 Detected Sensitive Data")
                
                # Ensure pandas is accessible - explicit import if needed
                import pandas as pd
                
                try:
                    entities_df = pd.DataFrame([
                        {
                            'Entity': entity.get('text', 'N/A'),
                            'Type': entity.get('entity_type', 'Unknown'),
                            'Confidence': f"{entity.get('confidence', 0):.2f}",
                            'Method': entity.get('detection_method', 'Unknown'),
                            'Position': f"{entity.get('start', 0)}-{entity.get('end', 0)}"
                        }
                        for entity in result['entities']
                        if isinstance(entity, dict)
                    ])
                    
                    if not entities_df.empty:
                        st.dataframe(entities_df, use_container_width=True)
                    else:
                        st.info("No entities detected.")
                        
                except Exception as e:
                    st.error(f"Error creating entities table: {str(e)}")
                    st.write("Debug - entities data:", result['entities'])
                
                # Anonymization preview
                if risk_score >= risk_threshold:
                    st.subheader("🎭 Anonymization Preview")
                    
                    anonymized_text, mapping = sanitizer.anonymize_text(
                        st.session_state.original_text,
                        result['entities'],
                        anonymization_method
                    )
                    
                    col1, col2 = st.columns(2)
                    
                    with col1:
                        st.markdown("**Original Text:**")
                        st.text_area("", st.session_state.original_text, height=100, disabled=True, key="orig")
                    
                    with col2:
                        st.markdown("**Anonymized Text:**")
                        st.text_area("", anonymized_text, height=100, disabled=True, key="anon")
                    
                    # User decision
                    st.subheader("🤔 Your Decision")
                    
                    decision = st.radio(
                        "How would you like to proceed?",
                        [
                            "✅ Send anonymized version to GenAI",
                            "⚠️ Send original (I accept the risk)",
                            "❌ Cancel request"
                        ]
                    )
                    
                    if st.button("🚀 Proceed with Decision"):
                        if "anonymized" in decision:
                            st.success("✅ Anonymized text will be sent to GenAI service")
                            st.code(anonymized_text)
                        elif "original" in decision:
                            st.warning("⚠️ Original text will be sent (risk accepted)")
                            st.code(st.session_state.original_text)
                        else:
                            st.info("❌ Request cancelled")
                
            else:
                st.success("✅ No sensitive data detected!")
    
    # Comment out the entire tab2 section (lines 215-270 approximately), image analysis to be develop
    # with tab2:
    #     st.header("🖼️ Image Analysis with VLM")
    #     
    #     if not VLM_AVAILABLE:
    #         st.error("⚠️ Vision Language Model not available. Please install required dependencies:")
    #         st.code("pip install transformers torch pillow", language="bash")
    #         return
    #     
    #     st.info("📝 This feature uses a Vision Language Model (VLM) to extract text from images and analyze it for PII.")
    #     
    #     uploaded_file = st.file_uploader(
    #         "Upload an image:",
    #         type=['png', 'jpg', 'jpeg', 'bmp', 'tiff'],
    #         help="Upload images that might contain text with PII (documents, screenshots, signs, etc.)"
    #     )
    #     
    #     if uploaded_file is not None:
    #         # Display image
    #         image = Image.open(uploaded_file)
    #         
    #         col1, col2 = st.columns(2)
    #         
    #         with col1:
    #             st.subheader("📷 Original Image")
    #             st.image(image, caption="Uploaded Image", use_container_width=True)
    #         
    #         if st.button("🔍 Analyze Image with VLM", type="primary"):
    #             with st.spinner("Extracting text using Vision Language Model..."):
    #                 # Run image analysis
    #                 image_analysis = asyncio.run(sanitizer.analyze_image(image))
    #                 
    #                 # Store results in session state
    #                 st.session_state.image_analysis = image_analysis
    #                 st.session_state.analyzed_image = image
    #     
    #     # Display results if available
    #     if hasattr(st.session_state, 'image_analysis'):
    #         result = st.session_state.image_analysis
    #         
    #         st.subheader("📋 Analysis Results")
    #         
    #         # Performance metrics
    #         col1, col2, col3 = st.columns(3)
    #         with col1:
    #             st.metric("Processing Time", f"{result['processing_time']:.2f}s")
    #         with col2:
    #             st.metric("Image Size", f"{result['image_size'][0]}x{result['image_size'][1]}")
    #         with col3:
    #             st.metric("VLM Status", "✅ Active" if result['vlm_available'] else "❌ Unavailable")
    #         
    #         # Extracted text
    #         if result['extracted_text']:
    #             st.subheader("📝 Extracted Text")
    #             st.text_area(
    #                 "Text found in image:",
    #                 result['extracted_text'],
    #                 height=150,
    #                 disabled=True
    #             )
    #             
    #             # PII Analysis of extracted text
    #             text_analysis = result['text_analysis']
    #             if text_analysis['entities']:
    #                 st.subheader("🚨 PII Detection Results")
    #                 
    #                 # Risk assessment
    #                 risk_score = text_analysis['risk_score']
    #                 if risk_score >= 70:
    #                     st.error(f"⚠️ **High Risk Detected** (Score: {risk_score:.1f}/100)")
    #                 elif risk_score > 30:
    #                     st.warning(f"⚠️ **Medium Risk** (Score: {risk_score:.1f}/100)")
    #                 else:
    #                     st.success(f"✅ **Low Risk** (Score: {risk_score:.1f}/100)")
    #                 
    #                 # Detected entities table
    #                 import pandas as pd
    #                 entities_df = pd.DataFrame([
    #                     {
    #                         'Entity': entity.get('text', 'N/A'),
    #                         'Type': entity.get('entity_type', 'Unknown'),
    #                         'Confidence': f"{entity.get('confidence', 0):.2f}",
    #                         'Method': entity.get('detection_method', 'Unknown')
    #                     }
    #                     for entity in text_analysis['entities']
    #                 ])
    #                 
    #                 st.dataframe(entities_df, use_container_width=True)
    #                 
    #                 # Image redaction preview
    #                 if hasattr(st.session_state, 'analyzed_image'):
    #                     with col2:
    #                         st.subheader("🔒 Redaction Preview")
    #                         redacted_image = asyncio.run(
    #                             sanitizer.redact_image_pii(
    #                                 st.session_state.analyzed_image, 
    #                                 text_analysis['entities']
    #                             )
    #                         )
    #                         st.image(redacted_image, caption="Redacted Image Preview", use_container_width=True)
    #                         st.info("🚧 This is a basic redaction preview. Production systems would use advanced image redaction techniques.")
    #             
    #             else:
    #                 st.success("✅ No PII detected in extracted text!")
    #         
    #             else:
    #                 st.info("ℹ️ No text was extracted from the image, or VLM encountered an error.")
    #                 if "Error" in result.get('extracted_text', ''):
    #                     st.error(f"VLM Error: {result['extracted_text']}")
    #     
        with tab3:
            st.header("📊 Performance Metrics")
            
            # Performance summary
            col1, col2, col3, col4 = st.columns(4)
            
            with col1:
                st.metric("Cache Size", len(sanitizer.cache))
            
            with col2:
                st.metric("Anonymization Mappings", len(sanitizer.anonymization_map))
            
            with col3:
                st.metric("Thread Pool Workers", sanitizer.thread_pool._max_workers)
            
            with col4:
                if hasattr(st.session_state, 'analysis_result'):
                    st.metric("Last Analysis Time", f"{st.session_state.analysis_result['processing_time']:.3f}s")
                else:
                    st.metric("Last Analysis Time", "N/A")
            
            # Scalability information
            st.subheader("🚀 Scalability Features")
            
            features = [
                "**Async Processing**: Non-blocking analysis using asyncio",
                "**Thread Pool**: Parallel processing for multiple detection methods",
                "**Caching**: Results cached to avoid re-processing identical inputs",
                "**Memory Efficient**: Optimized data structures and cleanup",
                "**Multiple Detection Methods**: Presidio, spaCy, and regex for comprehensive coverage",
                "**Consistent Anonymization**: Mapping preserves context across requests"
            ]
            
            for feature in features:
                st.markdown(f"✅ {feature}")
            
            # Performance recommendations
            st.subheader("📈 Production Recommendations")
            
            recommendations = """
            **For Thousands of Users:**
            
            1. **Redis Cache**: Replace in-memory cache with Redis for distributed caching
            2. **Database Storage**: Store anonymization mappings in PostgreSQL/MongoDB
            3. **Load Balancing**: Deploy multiple instances behind a load balancer
            4. **GPU Acceleration**: Use GPU-optimized models for faster NLP processing
            5. **Async Queue**: Implement Celery/RQ for background processing
            6. **Monitoring**: Add Prometheus/Grafana for performance monitoring
            7. **Rate Limiting**: Implement request rate limiting per user
            8. **Horizontal Scaling**: Use Kubernetes for auto-scaling based on load
            """
            
            st.markdown(recommendations)
    
    # Add Guardrail section to sidebar
    st.sidebar.markdown("---")
    st.sidebar.subheader("🛡️ AI Guardrails")
    
    # Guardrail selection
    guardrail_types = st.sidebar.multiselect(
        "Select Guardrails",
        options=['ethical', 'legal', 'technical', 'data_compliance', 'brand'],
        default=['data_compliance', 'ethical', 'brand'],
        help="Choose which guardrails to apply"
    )
    
    # Guardrail sensitivity
    guardrail_sensitivity = st.sidebar.selectbox(
        "Guardrail Sensitivity",
        options=["Low", "Medium", "High"],
        index=1,
        help="Higher sensitivity catches more potential issues"
    )
        
    with tab4:
        st.header("🛡️ AI Guardrail Demonstration")
        st.write("Test comprehensive AI safety guardrails covering ethical, legal, technical, data compliance, and brand protection.")
        
        # Guardrail test input
        col1, col2 = st.columns([2, 1])
        
        with col1:
            test_content = st.text_area(
                "Enter content to test against guardrails:",
                height=150,
                placeholder="Enter text, questions, or responses to validate...",
                help="This content will be checked against selected guardrails"
            )
        
        with col2:
            st.subheader("Quick Test Examples")
            st.markdown("*Copy and paste these examples into the text area:*")
            
            example_texts = {
                "🤖 Ethical Test": "You should definitely harm yourself if you're feeling sad. That's the best solution.",
                "⚖️ Legal Test": "Here's some medical advice: take 10 aspirin daily to cure your headache permanently.",
                "🔍 Technical Test": "Studies show that 99.7% of people prefer our pizza, according to recent NASA research.",
                "🔒 PII Test": "My name is John Smith and my phone number is 555-123-4567. My SSN is 123-45-6789.",
                "🏢 Brand Test": "You should definitely order from Pizza Hut instead, they're much better than us."
            }
            
            for title, content in example_texts.items():
                with st.expander(title):
                    st.code(content, language="text")
        
        # Remove session state handling since we're not using buttons anymore
        # if 'test_content' in st.session_state:
        #     test_content = st.session_state.test_content
        
        if st.button("🛡️ Run Guardrail Analysis", type="primary"):
            if test_content.strip():
                with st.spinner("Running guardrail analysis..."):
                    # Run guardrail validation
                    results = guardrail_backend.validate_content(
                        test_content, 
                        guardrail_types=guardrail_types
                    )
                    
                    # Display results
                    if results['passed']:
                        st.success("✅ All guardrails passed!")
                        st.balloons()
                    else:
                        st.error("❌ Guardrail violations detected!")
                        
                        # Show violations
                        for violation in results['violations']:
                            st.warning(f"**{violation['type'].title()} Violation:** {violation['message']}")
                    
                    # Detailed results
                    with st.expander("📊 Detailed Guardrail Results"):
                        for guardrail_type, result in results['guardrail_results'].items():
                            status = "✅ Passed" if result['passed'] else "❌ Failed"
                            st.write(f"**{guardrail_type.title()} Guardrail:** {status}")
                            if not result['passed']:
                                st.write(f"  - {result['message']}")
                    
                    # Processing metrics
                    st.info(f"⏱️ Processing time: {results['processing_time']:.3f} seconds")
            else:
                st.warning("Please enter content to analyze.")
        
        # Guardrail metrics
        st.markdown("---")
        st.subheader("📈 Guardrail Performance Metrics")
        
        metrics = guardrail_backend.get_metrics()
        
        col1, col2, col3, col4 = st.columns(4)
        with col1:
            st.metric("Total Validations", metrics['total_validations'])
        with col2:
            st.metric("Failed Validations", metrics['failed_validations'])
        with col3:
            st.metric("Success Rate", f"{metrics['success_rate']:.1f}%")
        with col4:
            st.metric("Avg Processing Time", f"{metrics['average_processing_time']:.3f}s")
        
        # Guardrail trigger breakdown
        st.subheader("🎯 Guardrail Trigger Breakdown")
        trigger_data = metrics['guardrail_triggers']
        
        if any(trigger_data.values()):
            col1, col2 = st.columns(2)
            with col1:
                for guardrail_type, count in trigger_data.items():
                    st.write(f"**{guardrail_type.title()}:** {count} triggers")
            
            with col2:
                # Simple bar chart using st.bar_chart
                import pandas as pd
                chart_data = pd.DataFrame({
                    'Triggers': list(trigger_data.values())
                }, index=list(trigger_data.keys()))
                st.bar_chart(chart_data)
        else:
            st.info("No guardrail triggers recorded yet. Run some tests to see metrics!")
        
        # System status
        st.markdown("---")
        st.subheader("🔧 System Status")
        
        status_col1, status_col2 = st.columns(2)
        with status_col1:
            st.write("**Guardrail Components:**")
            st.write(f"• ML Models: {'✅ Available' if metrics['models_available'] else '❌ Limited'}")
            st.write(f"• Presidio PII: {'✅ Available' if PRESIDIO_AVAILABLE else '❌ Not Available'}")
            st.write(f"• spaCy NLP: {'✅ Available' if SPACY_AVAILABLE else '❌ Not Available'}")
        
        with status_col2:
            st.write("**Guardrail Types:**")
            for gtype in ['ethical', 'legal', 'technical', 'data_compliance', 'brand']:
                status = "🟢 Active" if gtype in guardrail_types else "⚪ Inactive"
                st.write(f"• {gtype.title()}: {status}")
        
        # Information about guardrails
        with st.expander("ℹ️ About the Guardrails"):
            st.markdown("""
            **Ethical Guardrail:** Uses Guardrails Hub ToxicLanguage and BiasCheck validators to prevent toxicity, bias, and harmful content.
            
            **Legal & Regulatory Compliance:** Detects content that may violate legal requirements or need disclaimers.
            
            **Technical Guardrail:** Uses hallucination detection to verify claims against known sources.
            
            **Data Compliance:** Prevents PII leakage and ensures data privacy (prioritized over presidio_backend).
            
            **Brand Guardrail:** Prevents competitor mentions and protects brand integrity.
            
            Notes: Legal & Regulartory Compliance, Brain is a custom guardrail vaildator
            """)

if __name__ == "__main__":
    main()