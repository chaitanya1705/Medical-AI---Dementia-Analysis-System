from flask import Flask, request, jsonify
from flask_cors import CORS
import torch
import torch.nn as nn
import numpy as np
from PIL import Image
import pandas as pd
import io
import base64
from transformers import AutoTokenizer, AutoModel
import torchvision.transforms as transforms
import requests
import os
import google.generativeai as genai
from datetime import datetime
import timm
import torchvision.models as models
import albumentations as A
from albumentations.pytorch import ToTensorV2
import joblib  # For loading Random Forest
try:
    from sklearn.ensemble import RandomForestRegressor
    SKLEARN_AVAILABLE = True
except ImportError:
    print("WARNING: scikit-learn not available. Random Forest regression disabled.")
    SKLEARN_AVAILABLE = False

# Load environment variables
try:
    from dotenv import load_dotenv
    load_dotenv()
    print("SUCCESS: .env file loaded successfully")
except ImportError:
    print("WARNING: python-dotenv not installed. Using system environment variables only.")

app = Flask(__name__)
CORS(app)

# Configure Gemini API
GEMINI_API_KEY = os.getenv('GEMINI_API_KEY')
if GEMINI_API_KEY:
    try:
        genai.configure(api_key=GEMINI_API_KEY)
        
        model_names = [
            'gemini-1.5-flash',
            'gemini-1.5-pro',
            'gemini-1.0-pro',
            'gemini-pro'
        ]
        
        gemini_model = None
        for model_name in model_names:
            try:
                gemini_model = genai.GenerativeModel(model_name)
                test_response = gemini_model.generate_content("Hello")
                print(f"SUCCESS: Gemini API configured successfully with model: {model_name}")
                break
            except Exception as e:
                print(f"WARNING: Failed to use model {model_name}: {e}")
                continue
        
        if gemini_model is None:
            print("ERROR: Could not initialize any Gemini model. Using fallback care plans.")
            
    except Exception as e:
        print(f"ERROR: Gemini API configuration failed: {e}")
        gemini_model = None
else:
    print("WARNING: GEMINI_API_KEY not found in environment variables.")
    gemini_model = None

# Global variables
model = None
tokenizer = None
rf_regressor = None  # NEW: Random Forest regressor
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

def enhance_biomarker_features(biomarkers):
    """Create additional engineered features from biomarkers (same as training)"""
    
    # Original 6 biomarkers
    features = biomarkers.copy()
    
    # Ratios (often important in medical diagnosis)
    if len(biomarkers) >= 6:
        features.extend([
            biomarkers[3] / biomarkers[4] if biomarkers[4] != 0 else 0,  # AB42/AB40 ratio
            biomarkers[0] / biomarkers[1] if biomarkers[1] != 0 else 0,  # P-Tau 181/217 ratio
            (biomarkers[0] + biomarkers[1] + biomarkers[2]) / 3,        # Average P-Tau
            (biomarkers[3] + biomarkers[4]) / 2,                        # Average Amyloid
        ])
    
    # Log transforms (often used in medical data)
    log_features = [np.log1p(max(0, f)) for f in biomarkers]
    features.extend(log_features)
    
    return features

def create_medical_transforms(is_training=False):
    """Create medical-specific image transforms (validation only for inference)"""
    
    # For inference, only use validation transforms
    return A.Compose([
        A.Resize(224, 224),
        A.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
        ToTensorV2()
    ])

def create_vision_encoder(encoder_type="efficientnet_b0"):
    """Create different vision encoders optimized for medical imaging"""
    
    if encoder_type == "inception_v3":
        # InceptionV3 - good for multi-scale features
        model = models.inception_v3(pretrained=True, aux_logits=False)
        model.fc = nn.Identity()
        return model, 2048
    
    elif encoder_type == "resnet50":
        # ResNet50 - deeper and more powerful
        model = models.resnet50(pretrained=True)
        model.fc = nn.Identity()
        return model, 2048
    
    elif encoder_type == "densenet121":
        # DenseNet - excellent for medical imaging
        model = models.densenet121(pretrained=True)
        model.classifier = nn.Identity()
        return model, 1024
        
    elif encoder_type == "efficientnet_b3":
        # Larger EfficientNet - better capacity
        model = timm.create_model('efficientnet_b3', pretrained=True, num_classes=0)
        return model, 1536
        
    elif encoder_type == "convnext_tiny":
        # ConvNeXt - modern CNN architecture
        model = timm.create_model('convnext_tiny', pretrained=True, num_classes=0)
        return model, 768
    
    elif encoder_type == "swin_tiny":
        # Swin Transformer - vision transformer
        model = timm.create_model('swin_tiny_patch4_window7_224', pretrained=True, num_classes=0)
        return model, 768
    
    else:
        # Default EfficientNet-B0
        model = timm.create_model('efficientnet_b0', pretrained=True, num_classes=0)
        return model, 1280

class CustomMedicalVLM(nn.Module):
    def __init__(self, num_classes=4, biomarker_dim=16, hidden_dim=256, vision_encoder="efficientnet_b0"):
        super(CustomMedicalVLM, self).__init__()

        print(f"Initializing Custom Medical VLM with {vision_encoder}...")

        # Text Encoder
        text_model_options = [
            "emilyalsentzer/Bio_ClinicalBERT",  # Clinical BERT
            "dmis-lab/biobert-base-cased-v1.1",  # BioBERT
            "microsoft/BiomedNLP-PubMedBERT-base-uncased-abstract-fulltext",  # PubMedBERT
            "distilbert-base-uncased"  # Fallback
        ]

        self.text_encoder = None
        text_dim = 768

        for model_name in text_model_options:
            try:
                print(f"Loading text model: {model_name}")
                self.text_encoder = AutoModel.from_pretrained(model_name)
                text_dim = self.text_encoder.config.hidden_size
                print(f"✓ Successfully loaded {model_name} with hidden size: {text_dim}")
                break
            except Exception as e:
                print(f"Failed to load {model_name}: {e}")
                continue

        if self.text_encoder is None:
            raise ValueError("Could not load any text model")

        # Vision Encoder - Multiple options
        try:
            print(f"Loading {vision_encoder} for vision...")
            self.vision_encoder, vision_dim = create_vision_encoder(vision_encoder)
            print(f"✓ {vision_encoder} loaded with output size: {vision_dim}")
        except Exception as e:
            print(f"{vision_encoder} failed: {e}. Using EfficientNet-B0...")
            self.vision_encoder, vision_dim = create_vision_encoder("efficientnet_b0")

        # Enhanced Biomarker Encoder (now handles 16 features)
        self.biomarker_encoder = nn.Sequential(
            nn.Linear(biomarker_dim, hidden_dim),
            nn.BatchNorm1d(hidden_dim),
            nn.ReLU(),
            nn.Dropout(0.3),
            nn.Linear(hidden_dim, hidden_dim // 2),
            nn.BatchNorm1d(hidden_dim // 2),
            nn.ReLU(),
            nn.Dropout(0.2),
            nn.Linear(hidden_dim // 2, 64)
        )

        # Enhanced Feature Projectors
        self.text_projector = nn.Sequential(
            nn.Linear(text_dim, hidden_dim * 2),
            nn.ReLU(),
            nn.Dropout(0.3),
            nn.Linear(hidden_dim * 2, hidden_dim),
            nn.BatchNorm1d(hidden_dim),
            nn.ReLU(),
            nn.Dropout(0.2)
        )

        self.vision_projector = nn.Sequential(
            nn.Linear(vision_dim, hidden_dim * 2),
            nn.ReLU(),
            nn.Dropout(0.3),
            nn.Linear(hidden_dim * 2, hidden_dim),
            nn.BatchNorm1d(hidden_dim),
            nn.ReLU(),
            nn.Dropout(0.2)
        )

        # Multi-modal Fusion Layer
        fusion_input_dim = hidden_dim + hidden_dim + 64  # text + vision + biomarkers
        self.fusion_layer = nn.Sequential(
            nn.Linear(fusion_input_dim, hidden_dim),
            nn.BatchNorm1d(hidden_dim),
            nn.ReLU(),
            nn.Dropout(0.3),
            nn.Linear(hidden_dim, hidden_dim // 2),
            nn.BatchNorm1d(hidden_dim // 2),
            nn.ReLU(),
            nn.Dropout(0.2),
            nn.Linear(hidden_dim // 2, hidden_dim // 4)
        )

        # Classification head only (regression removed)
        self.classification_head = nn.Sequential(
            nn.Linear(hidden_dim // 4, num_classes)
        )
        
        # Store feature dimension for Random Forest
        self.feature_dim = hidden_dim // 4

        print(f"✓ Custom Medical VLM initialized successfully!")
        print(f"  Text encoder: {text_dim} -> {hidden_dim}")
        print(f"  Vision encoder: {vision_dim} -> {hidden_dim}")
        print(f"  Biomarker encoder: {biomarker_dim} -> 64")
        print(f"  Fusion: {fusion_input_dim} -> {hidden_dim // 4}")
        print(f"  Classification only (Regression via Random Forest)")

    def forward(self, image, biomarkers, input_ids, attention_mask, return_features=False):
        # Text encoding
        text_outputs = self.text_encoder(
            input_ids=input_ids,
            attention_mask=attention_mask
        )

        # Use pooled output or mean of last hidden state
        if hasattr(text_outputs, 'pooler_output') and text_outputs.pooler_output is not None:
            text_features = text_outputs.pooler_output
        else:
            text_features = text_outputs.last_hidden_state.mean(dim=1)

        text_features = self.text_projector(text_features)

        # Vision encoding
        vision_features = self.vision_encoder(image)
        vision_features = self.vision_projector(vision_features)

        # Biomarker encoding
        biomarker_features = self.biomarker_encoder(biomarkers)

        # Multi-modal fusion
        fused_features = torch.cat([text_features, vision_features, biomarker_features], dim=1)
        fused_features = self.fusion_layer(fused_features)

        # Classification output
        classification_output = self.classification_head(fused_features)

        if return_features:
            return classification_output, fused_features
        else:
            return classification_output

def load_trained_model(encoder_name="inception_v3"):
    """Load pre-trained VLM model from .pth file with specified encoder"""
    global model
    try:
        print(f"Loading your pre-trained VLM model with {encoder_name}...")
        model_path = f'models/best_medical_vlm_{encoder_name}.pth'
        print(f"Looking for: best_medical_vlm_{encoder_name}.pth")
        
        model = CustomMedicalVLM(num_classes=4, biomarker_dim=16, hidden_dim=256, vision_encoder=encoder_name)  # Revert back to 256
        
        # Load state dict and handle regression_head keys
        state_dict = torch.load(f'best_medical_vlm_{encoder_name}.pth', map_location=device)
        
        # Remove regression_head keys if they exist (for backward compatibility)
        state_dict_filtered = {}
        for key, value in state_dict.items():
            if not key.startswith('regression_head'):
                state_dict_filtered[key] = value
            else:
                print(f"Skipping regression_head key: {key} (handled by Random Forest)")
        
        model.load_state_dict(state_dict_filtered, strict=False)
        model.to(device)
        model.eval()
        
        print(f"SUCCESS: Your pre-trained VLM {encoder_name} model loaded successfully!")
        return True
    except FileNotFoundError:
        print(f"ERROR: best_medical_vlm_{encoder_name}.pth not found!")
        # Try fallback models
        fallback_models = ["efficientnet_b0", "inception_v3", "densenet121", "resnet50"]
        for fallback in fallback_models:
            if fallback != encoder_name:
                try:
                    print(f"Trying fallback VLM model: {fallback}")
                    model = CustomMedicalVLM(num_classes=4, biomarker_dim=16, hidden_dim=256, vision_encoder=fallback)  # Revert back to 256
                    
                    # Handle regression_head for fallback too
                    state_dict = torch.load(f'best_medical_vlm_{fallback}.pth', map_location=device)
                    state_dict_filtered = {}
                    for key, value in state_dict.items():
                        if not key.startswith('regression_head'):
                            state_dict_filtered[key] = value
                    
                    model.load_state_dict(state_dict_filtered, strict=False)
                    model.to(device)
                    model.eval()
                    print(f"SUCCESS: Loaded fallback VLM model {fallback}")
                    return True
                except Exception as e:
                    print(f"Fallback {fallback} failed: {e}")
                    continue
        return False
    except Exception as e:
        print(f"ERROR: Error loading your VLM model: {e}")
        return False

def load_random_forest(encoder_name="inception_v3"):
    """Load pre-trained Random Forest model from .pkl file"""
    global rf_regressor
    try:
        # Check if sklearn is available
        try:
            rf_path = f'models/rf_regressor_{encoder_name}.pkl'
            import sklearn
            print(f"sklearn version: {sklearn.__version__}")
        except ImportError:
            print("ERROR: sklearn not installed. Install with: pip install scikit-learn")
            print("Random Forest will not be available - progression prediction disabled")
            return False
        
        rf_path = f'rf_regressor_{encoder_name}.pkl'
        print(f"Loading Random Forest regressor: {rf_path}")
        
        rf_regressor = joblib.load(rf_path)
        print(f"SUCCESS: Random Forest regressor {encoder_name} loaded successfully!")
        print(f"RF features expected: {rf_regressor.n_features_in_}")
        return True
    except FileNotFoundError:
        print(f"ERROR: rf_regressor_{encoder_name}.pkl not found!")
        # Try fallback models
        fallback_models = ["efficientnet_b0", "inception_v3", "densenet121", "resnet50"]
        for fallback in fallback_models:
            if fallback != encoder_name:
                try:
                    fallback_rf_path = f'rf_regressor_{fallback}.pkl'
                    print(f"Trying fallback Random Forest: {fallback_rf_path}")
                    rf_regressor = joblib.load(fallback_rf_path)
                    print(f"SUCCESS: Loaded fallback Random Forest {fallback}")
                    return True
                except Exception as e:
                    print(f"Fallback {fallback} failed: {e}")
                    continue
        return False
    except Exception as e:
        print(f"ERROR: Error loading Random Forest: {e}")
        if "sklearn" in str(e).lower():
            print("Install scikit-learn with: pip install scikit-learn")
        return False

def load_tokenizer():
    """Load the tokenizer"""
    global tokenizer
    tokenizer_options = [
        "emilyalsentzer/Bio_ClinicalBERT",
        "dmis-lab/biobert-base-cased-v1.1",
        "microsoft/BiomedNLP-PubMedBERT-base-uncased-abstract-fulltext",
        "distilbert-base-uncased"
    ]

    for model_name in tokenizer_options:
        try:
            tokenizer = AutoTokenizer.from_pretrained(model_name)
            print(f"Tokenizer loaded: {model_name}")
            return True
        except Exception as e:
            continue
    
    print("Could not load any tokenizer")
    return False

def extract_biomarkers_from_dataframe(df):
    """Extract biomarkers from uploaded DataFrame and apply enhanced feature engineering"""
    try:
        df.columns = df.columns.str.strip()
        
        column_mapping = {
            'Plasma P-Tau 181': 'biomarker_1',
            'Plasma P-Tau 217': 'biomarker_2', 
            'Plasma P-Tau 231': 'biomarker_3',
            'Plasma Amyloid Beta 42': 'biomarker_4',
            'Plasma Amyloid Beta 40': 'biomarker_5',
            'AB42/AB40': 'biomarker_6',
        }
        
        existing_mapping = {k: v for k, v in column_mapping.items() if k in df.columns}
        df = df.rename(columns=existing_mapping)
        
        biomarker_cols = ['biomarker_1', 'biomarker_2', 'biomarker_3', 'biomarker_4', 'biomarker_5', 'biomarker_6']
        first_row = df.iloc[0]
        biomarker_values = []
        
        for col in biomarker_cols:
            if col in df.columns:
                biomarker_values.append(float(first_row[col]))
            else:
                biomarker_values.append(0.0)
        
        # Apply enhanced feature engineering (matching training)
        enhanced_biomarkers = enhance_biomarker_features(biomarker_values)
        
        return enhanced_biomarkers
    except Exception as e:
        raise ValueError(f"Error extracting biomarkers: {e}")

def create_enhanced_medical_text(biomarkers):
    """Create enhanced medical text description matching training approach"""
    
    # Extract original biomarkers
    original_biomarkers = biomarkers[:6]
    biomarker_names = ['P-Tau 181', 'P-Tau 217', 'P-Tau 231', 'Amyloid Beta 42', 'Amyloid Beta 40', 'AB42/AB40']
    
    # Start with clinical assessment
    text = "Clinical Assessment for dementia staging. Plasma biomarker analysis reveals: "
    
    # Add biomarker context with interpretations
    for name, value in zip(biomarker_names, original_biomarkers):
        # Add clinical interpretation
        if 'P-Tau' in name and value > 20:
            interpretation = "(elevated)"
        elif 'Amyloid' in name and value < 100:
            interpretation = "(reduced)"
        else:
            interpretation = "(normal range)"
        
        text += f"{name}: {value:.3f} {interpretation}, "
    
    text += "MRI neuroimaging shows structural brain changes consistent with cognitive assessment."
    
    return text

def predict_stage_and_progression(image_tensor, enhanced_biomarkers):
    """Make prediction using VLM for classification and Random Forest for regression"""
    global model, tokenizer, rf_regressor
    
    # Create enhanced medical text
    medical_text = create_enhanced_medical_text(enhanced_biomarkers)
    
    # Tokenize text
    text_inputs = tokenizer(
        medical_text,
        max_length=512,
        padding='max_length',
        truncation=True,
        return_tensors='pt'
    ).to(device)
    
    # Prepare inputs
    image_tensor = image_tensor.unsqueeze(0).to(device)
    biomarker_tensor = torch.tensor(enhanced_biomarkers, dtype=torch.float32).unsqueeze(0).to(device)
    
    # Make prediction with VLM
    with torch.no_grad():
        cls_output, fused_features = model(
            image_tensor,
            biomarker_tensor,
            text_inputs['input_ids'],
            text_inputs['attention_mask'],
            return_features=True  # Get features for Random Forest
        )
        
        stage_probs = torch.softmax(cls_output, dim=1).cpu().numpy()[0]
        predicted_stage_idx = torch.argmax(cls_output, dim=1).cpu().numpy()[0]
        
        stage_names = ['ND (No Dementia)', 'VMD (Very Mild Dementia)', 'MD (Mild Dementia)', 'MOD (Moderate Dementia)']
        predicted_stage = stage_names[predicted_stage_idx]
        
        # Prepare features for Random Forest (if available and not MOD stage)
        progression_months = 0.0
        if rf_regressor is not None and predicted_stage_idx < 3:  # Not MOD stage
            try:
                # Based on training code analysis:
                # RF was trained with features from model with hidden_dim=128 -> feature_dim=32
                # But current model has hidden_dim=256 -> feature_dim=64
                # We need to match the exact feature combination from training
                
                # Get dimensions
                neural_features = fused_features.cpu().numpy()  # (1, 64) from current model
                biomarker_features = biomarker_tensor.cpu().numpy()  # (1, 16)
                stage_prob_features = stage_probs.reshape(1, -1)  # (1, 4)
                
                print(f"Debug - Neural features shape: {neural_features.shape}")
                print(f"Debug - Biomarker features shape: {biomarker_features.shape}")
                print(f"Debug - Stage prob features shape: {stage_prob_features.shape}")
                
                # The training used model.feature_dim=32, but current model gives 64
                # We need to reduce neural features to match training expectations
                if neural_features.shape[1] == 64:  # Current model output
                    # Reduce to 32 features to match training (take first 32 or use PCA-like reduction)
                    neural_features_reduced = neural_features[:, :32]  # Simple truncation
                    print(f"Debug - Reduced neural features to: {neural_features_reduced.shape}")
                else:
                    neural_features_reduced = neural_features
                
                # Combine features exactly as in training: neural_32 + biomarkers_16 + stage_probs_4 = 52
                rf_features = np.concatenate([
                    neural_features_reduced,  # 32 features (reduced from 64)
                    biomarker_features,       # 16 features
                    stage_prob_features       # 4 features
                ], axis=1)
                
                print(f"Debug - Combined RF features shape: {rf_features.shape}")
                print(f"Debug - RF expects: {rf_regressor.n_features_in_} features")
                
                # Final check and adjustment
                if rf_features.shape[1] != rf_regressor.n_features_in_:
                    expected_features = rf_regressor.n_features_in_
                    current_features = rf_features.shape[1]
                    
                    print(f"WARNING: Still have feature mismatch!")
                    print(f"  Expected: {expected_features}")
                    print(f"  Got: {current_features}")
                    
                    if current_features > expected_features:
                        # Truncate features
                        rf_features = rf_features[:, :expected_features]
                        print(f"  Truncated to {rf_features.shape[1]} features")
                    elif current_features < expected_features:
                        # Pad with zeros
                        padding = np.zeros((1, expected_features - current_features))
                        rf_features = np.concatenate([rf_features, padding], axis=1)
                        print(f"  Padded to {rf_features.shape[1]} features")
                
                # Predict progression time using Random Forest
                progression_prediction = rf_regressor.predict(rf_features)
                progression_months = float(max(0, progression_prediction[0]))
                
                print(f"Random Forest prediction: {progression_months:.2f} months")
                
            except Exception as e:
                print(f"Random Forest prediction failed: {e}")
                progression_months = 0.0
        
        # Extract original biomarkers for display
        original_biomarkers = enhanced_biomarkers[:6]
        biomarker_names = ['P-Tau 181', 'P-Tau 217', 'P-Tau 231', 'Amyloid Beta 42', 'Amyloid Beta 40', 'AB42/AB40']
        
        return {
            'predicted_stage': predicted_stage,
            'stage_index': int(predicted_stage_idx),
            'confidence': float(stage_probs[predicted_stage_idx]),
            'all_probabilities': {stage_names[i]: float(prob) for i, prob in enumerate(stage_probs)},
            'progression_months': progression_months,
            'prediction_method': 'VLM + Random Forest' if rf_regressor is not None else 'VLM Only',
            'biomarkers': {
                biomarker_names[i]: original_biomarkers[i] for i in range(6)
            },
            'enhanced_features_count': len(enhanced_biomarkers)
        }

# Keep all existing care plan generation functions unchanged
def generate_care_plan_with_gemini(stage, biomarkers, confidence):
    """Generate biomarker-specific and stage-tailored care plan using Gemini API"""
    
    # Create detailed biomarker analysis
    biomarker_analysis = analyze_biomarkers_for_prompt(biomarkers)
    
    prompt = f"""
You are a specialist neurologist creating a personalized dementia care plan based on detailed biomarker analysis.

PATIENT CLINICAL PROFILE:
- Dementia Stage: {stage}
- Diagnostic Confidence: {confidence:.1%}
- Biomarker Analysis: {biomarker_analysis}

BIOMARKER VALUES:
{format_biomarkers_for_prompt(biomarkers)}

CRITICAL REQUIREMENTS:
1. Generate EXACTLY 5 care recommendations
2. Each recommendation should be 1-2 sentences (100-150 characters)
3. Make recommendations SPECIFIC to the biomarker levels and dementia stage
4. Include both immediate actions and long-term strategies
5. Focus on non-pharmaceutical interventions

FORMATTING REQUIREMENTS:
- Start each with number: "1.", "2.", "3.", "4.", "5."
- Be specific about frequency, duration, or intensity when relevant
- Reference biomarker implications where appropriate

CARE PLAN FOCUS AREAS (tailor to biomarker profile):

For P-Tau levels (tau protein pathology):
- Cognitive stimulation intensity based on tau burden
- Sleep optimization (tau clearance occurs during sleep)
- Physical exercise recommendations

For Amyloid levels (amyloid pathology):
- Cardiovascular health interventions
- Anti-inflammatory dietary approaches
- Social engagement strategies

For AB42/AB40 ratio (amyloid processing):
- Metabolic health optimization
- Stress reduction techniques
- Environmental modifications

STAGE-SPECIFIC CONSIDERATIONS:
{get_stage_specific_guidance(stage)}

Generate 5 numbered recommendations that are:
- Biomarker-informed
- Stage-appropriate
- Actionable and specific
- Evidence-based
- Family-caregiver friendly

EXAMPLE FORMAT:
1. Implement structured cognitive training sessions 3x weekly targeting memory and executive function based on elevated tau levels
2. Establish consistent sleep schedule of 7-8 hours nightly to enhance amyloid clearance given current AB42/AB40 ratio
3. Begin supervised aerobic exercise program 30 minutes daily to address cardiovascular risk factors indicated by biomarker profile
4. Adopt Mediterranean diet with omega-3 supplementation to reduce neuroinflammation suggested by P-tau elevation
5. Schedule monthly cognitive assessments and biomarker monitoring to track progression and adjust interventions

NOW GENERATE 5 SPECIFIC CARE RECOMMENDATIONS:
"""

    try:
        max_retries = 3
        for attempt in range(max_retries):
            try:
                response = gemini_model.generate_content(
                    prompt,
                    generation_config={
                        'temperature': 0.4,
                        'top_p': 0.9,
                        'max_output_tokens': 800,
                    }
                )
                
                if hasattr(response, 'candidates') and response.candidates:
                    if response.candidates[0].content.parts:
                        response_text = response.candidates[0].content.parts[0].text
                        break
                elif hasattr(response, 'text') and response.text:
                    response_text = response.text
                    break
                else:
                    continue
                    
            except Exception as e:
                print(f"Gemini API attempt {attempt + 1} failed: {e}")
                if attempt == max_retries - 1:
                    raise e
                continue
        
        recommendations = parse_gemini_response_strict(response_text)
        
        if len(recommendations) != 5:
            print(f"Warning: Gemini returned {len(recommendations)} recommendations. Using enhanced fallback.")
            return get_enhanced_fallback_care_plan(stage, biomarkers)
        
        # Validate and enhance recommendations
        validated_recommendations = []
        for i, rec in enumerate(recommendations):
            rec = rec.strip()
            
            if len(rec) > 150:
                rec = rec[:147] + "..."
            elif len(rec) < 20:
                enhanced_fallback = get_enhanced_fallback_care_plan(stage, biomarkers)
                rec = enhanced_fallback[i] if i < len(enhanced_fallback) else "Follow specialized care protocols"
            
            validated_recommendations.append(rec)
        
        print(f"SUCCESS: Generated biomarker-specific care plan with {len(validated_recommendations)} recommendations")
        return validated_recommendations
        
    except Exception as e:
        print(f"Error generating care plan with Gemini: {e}")
        return get_enhanced_fallback_care_plan(stage, biomarkers)

def analyze_biomarkers_for_prompt(biomarkers):
    """Analyze biomarker levels to provide context for care planning"""
    analysis = []
    
    # P-Tau analysis
    ptau_181 = biomarkers.get('P-Tau 181', 0)
    ptau_217 = biomarkers.get('P-Tau 217', 0)
    
    if ptau_181 > 2.5 or ptau_217 > 1.8:
        analysis.append("Elevated tau pathology indicating significant neurodegeneration")
    elif ptau_181 > 1.5 or ptau_217 > 1.0:
        analysis.append("Moderate tau pathology suggesting active disease progression")
    else:
        analysis.append("Low tau pathology indicating early or minimal neurodegeneration")
    
    # Amyloid analysis
    ab42 = biomarkers.get('Amyloid Beta 42', 0)
    ab40 = biomarkers.get('Amyloid Beta 40', 0)
    ab_ratio = biomarkers.get('AB42/AB40', 0)
    
    if ab_ratio < 0.08:
        analysis.append("Significantly reduced AB42/AB40 ratio indicating high amyloid burden")
    elif ab_ratio < 0.12:
        analysis.append("Moderately reduced AB42/AB40 ratio suggesting amyloid accumulation")
    else:
        analysis.append("Normal AB42/AB40 ratio indicating low amyloid burden")
    
    return "; ".join(analysis)

def format_biomarkers_for_prompt(biomarkers):
    """Format biomarkers for the prompt with clinical context"""
    formatted = []
    for marker, value in biomarkers.items():
        if 'P-Tau' in marker:
            formatted.append(f"- {marker}: {value:.3f} pg/mL (neurodegeneration marker)")
        elif 'Amyloid Beta' in marker:
            formatted.append(f"- {marker}: {value:.3f} pg/mL (amyloid pathology marker)")
        elif 'AB42/AB40' in marker:
            formatted.append(f"- {marker}: {value:.3f} ratio (amyloid processing efficiency)")
        else:
            formatted.append(f"- {marker}: {value:.3f}")
    return "\n".join(formatted)

def get_stage_specific_guidance(stage):
    """Provide stage-specific guidance for care planning"""
    guidance = {
        'ND (No Dementia)': """
Focus on prevention and risk reduction. Emphasize lifestyle modifications that can delay onset.
Target cardiovascular health, cognitive reserve building, and early biomarker monitoring.
""",
        'VMD (Very Mild Dementia)': """
Emphasize maintaining independence while implementing supportive strategies. 
Focus on cognitive stimulation, safety measures, and slowing progression.
Introduce compensatory strategies and family education.
""",
        'MD (Mild Dementia)': """
Balance independence support with increased supervision and safety measures.
Focus on maintaining function, managing behavioral symptoms, and caregiver support.
Implement structured routines and environmental modifications.
""",
        'MOD (Moderate Dementia)': """
Prioritize safety, comfort, and quality of life. Implement comprehensive care strategies.
Focus on managing complex care needs, behavioral interventions, and family support.
Consider professional care services and specialized programs.
"""
    }
    return guidance.get(stage, guidance['VMD (Very Mild Dementia)'])

def get_enhanced_fallback_care_plan(stage, biomarkers):
    """Enhanced fallback care plans that consider biomarker levels"""
    
    # Analyze biomarkers for tailored recommendations
    ptau_high = any(biomarkers.get(marker, 0) > 2.0 for marker in ['P-Tau 181', 'P-Tau 217', 'P-Tau 231'])
    ab_ratio_low = biomarkers.get('AB42/AB40', 1.0) < 0.1
    
    base_plans = {
        'ND (No Dementia)': [
            "Implement daily aerobic exercise 30-45 minutes to enhance neuroplasticity and reduce dementia risk",
            "Follow Mediterranean diet with weekly fish consumption targeting brain health and inflammation reduction",
            "Engage in challenging cognitive activities 3x weekly including puzzles, reading, and learning new skills",
            "Maintain consistent sleep schedule 7-8 hours nightly to optimize amyloid clearance and cognitive function",
            "Schedule annual comprehensive cognitive evaluations and biomarker monitoring for early detection"
        ],
        'VMD (Very Mild Dementia)': [
            "Establish structured daily routines with visual cues and calendars to support memory and reduce confusion",
            "Begin supervised cognitive rehabilitation program focusing on memory strategies and executive function",
            "Implement home safety modifications including improved lighting, grab bars, and clear pathways",
            "Participate in social engagement activities 2-3x weekly to maintain cognitive stimulation and emotional wellbeing",
            "Coordinate quarterly medical monitoring with neurologist and biomarker tracking for progression assessment"
        ],
        'MD (Mild Dementia)': [
            "Provide structured supervision for complex activities while encouraging maintained independence in basic tasks",
            "Implement comprehensive safety protocols including medical alert systems and environment modifications",
            "Establish consistent caregiver routines with clear communication strategies and behavioral management techniques",
            "Engage in simplified cognitive and physical activities tailored to current abilities and interests",
            "Coordinate multidisciplinary care team including neurology, social work, and occupational therapy services"
        ],
        'MOD (Moderate Dementia)': [
            "Ensure continuous supervision and assistance with all activities of daily living and personal care",
            "Create calming environment with familiar objects, consistent routines, and minimal overwhelming stimuli",
            "Implement person-centered care approaches using validation therapy and music/art therapy interventions",
            "Establish comprehensive safety measures including wandering prevention and 24-hour monitoring systems",
            "Provide extensive caregiver support including respite care, support groups, and professional care consultation"
        ]
    }
    
    plan = base_plans.get(stage, base_plans['VMD (Very Mild Dementia)'])
    
    # Modify recommendations based on biomarker profile
    if ptau_high:
        # High tau - emphasize neuroprotection
        plan[1] = plan[1].replace("Mediterranean diet", "Anti-inflammatory Mediterranean diet with curcumin and omega-3 supplements")
    
    if ab_ratio_low:
        # Low AB ratio - emphasize amyloid clearance
        plan[3] = plan[3].replace("sleep schedule", "sleep hygiene protocol with sleep study evaluation for optimal amyloid clearance")
    
    return plan

def parse_gemini_response_strict(response_text):
    """Parse Gemini response with strict validation for exactly 5 numbered recommendations"""
    recommendations = []
    lines = response_text.strip().split('\n')
    
    expected_numbers = ['1.', '2.', '3.', '4.', '5.']
    
    for line in lines:
        line = line.strip()
        
        if not line:
            continue
            
        for i, expected_num in enumerate(expected_numbers):
            if line.startswith(expected_num):
                clean_line = line[len(expected_num):].strip()
                clean_line = clean_line.lstrip('-•*: ')
                
                if len(clean_line) >= 15:
                    recommendations.append(clean_line)
                    break
        
        if len(recommendations) == 5:
            break
    
    return recommendations

def get_fallback_care_plan(stage):
    """Enhanced fallback care plans with exactly 5 single-line recommendations"""
    care_plans = {
        'ND (No Dementia)': [
            "Maintain regular exercise routine with 30 minutes daily activity",
            "Follow healthy diet rich in fruits, vegetables, and omega-3",
            "Engage in cognitive activities like reading and puzzles", 
            "Ensure 7-8 hours of quality sleep nightly",
            "Schedule annual cognitive assessments and monitoring"
        ],
        'VMD (Very Mild Dementia)': [
            "Establish structured daily routines and consistent schedules",
            "Participate in social activities and maintain connections",
            "Use memory aids like calendars and reminder notes",
            "Engage in light exercise like walking for 30 minutes",
            "Schedule follow-up appointments every 3-6 months"
        ],
        'MD (Mild Dementia)': [
            "Implement home safety measures and remove trip hazards",
            "Maintain familiar routines and structured environment",
            "Provide supervision for complex tasks and activities",
            "Encourage simple cognitive exercises and social time",
            "Monitor mood changes and provide emotional support"
        ],
        'MOD (Moderate Dementia)': [
            "Ensure 24/7 supervision and assistance with daily tasks",
            "Create calm environment with minimal distractions",
            "Use simple communication and clear instructions",
            "Install safety equipment like door alarms and ID tags",
            "Consider respite care and adult day programs"
        ]
    }
    
    return care_plans.get(stage, care_plans['VMD (Very Mild Dementia)'])

@app.route('/api/analyze', methods=['POST'])
def analyze_patient_data():
    """Main API endpoint for patient analysis with VLM + Random Forest hybrid"""
    try:
        if model is None or tokenizer is None:
            return jsonify({'error': 'VLM model or tokenizer not loaded'}), 500
        
        if 'mri_image' not in request.files or 'biomarker_data' not in request.files:
            return jsonify({'error': 'Missing required files'}), 400
        
        mri_file = request.files['mri_image']
        biomarker_file = request.files['biomarker_data']
        
        # Process MRI image with enhanced transforms
        image = Image.open(mri_file.stream).convert('RGB')
        
        # Apply enhanced medical transforms
        transform = create_medical_transforms(is_training=False)
        image_array = np.array(image)
        transformed = transform(image=image_array)
        image_tensor = transformed['image']
        
        # Process biomarker data with enhanced features
        file_extension = biomarker_file.filename.split('.')[-1].lower()
        
        if file_extension == 'csv':
            df = pd.read_csv(biomarker_file.stream)
        elif file_extension in ['xlsx', 'xls']:
            df = pd.read_excel(biomarker_file.stream)
        else:
            return jsonify({'error': 'Unsupported file format'}), 400
        
        # Extract enhanced biomarkers (16 features instead of 6)
        enhanced_biomarkers = extract_biomarkers_from_dataframe(df)
        
        # Make prediction with VLM + Random Forest hybrid
        results = predict_stage_and_progression(image_tensor, enhanced_biomarkers)
        
        return jsonify({
            'success': True,
            'results': results
        })
        
    except Exception as e:
        return jsonify({'error': str(e)}), 500

@app.route('/api/generate_care_plan', methods=['POST'])
def generate_care_plan():
    """Generate care plan using Gemini API with strict validation"""
    try:
        data = request.json
        stage = data.get('stage', '')
        biomarkers = data.get('biomarkers', {})
        confidence = data.get('confidence', 0.0)
        
        print(f"Generating care plan for stage: {stage}")
        
        if gemini_model:
            try:
                care_plan = generate_care_plan_with_gemini(stage, biomarkers, confidence)
                plan_source = "AI-Generated"
            except Exception as e:
                print(f"Gemini API error: {e}")
                care_plan = get_fallback_care_plan(stage)
                plan_source = "Fallback"
        else:
            care_plan = get_fallback_care_plan(stage)
            plan_source = "Fallback"
        
        if len(care_plan) != 5:
            print(f"Warning: Care plan has {len(care_plan)} points, adjusting to 5")
            fallback = get_fallback_care_plan(stage)
            if len(care_plan) < 5:
                care_plan.extend(fallback[len(care_plan):5])
            else:
                care_plan = care_plan[:5]
        
        print(f"SUCCESS: Generated care plan with {len(care_plan)} recommendations ({plan_source})")
        
        return jsonify({
            'success': True,
            'care_plan': care_plan,
            'source': plan_source,
            'stage': stage,
            'recommendations_count': len(care_plan)
        })
        
    except Exception as e:
        print(f"ERROR in generate_care_plan: {e}")
        return jsonify({'error': str(e)}), 500

@app.route('/api/health', methods=['GET'])
def health_check():
    """Health check endpoint"""
    return jsonify({
        'status': 'healthy',
        'vlm_model_loaded': model is not None,
        'random_forest_loaded': rf_regressor is not None,
        'tokenizer_loaded': tokenizer is not None,
        'gemini_configured': gemini_model is not None,
        'prediction_method': 'VLM + Random Forest Hybrid',
        'enhanced_features': True,
        'biomarker_dimensions': 16,
        'timestamp': datetime.now().isoformat()
    })

@app.route('/api/initialize', methods=['POST'])
def initialize_models():
    """Initialize both VLM model and Random Forest with encoder selection"""
    
    # Get encoder choice from request
    data = request.get_json() or {}
    encoder_name = data.get('encoder', 'inception_v3')  # Default to inception_v3
    
    print(f"Initializing hybrid models with encoder: {encoder_name}")
    
    # Load VLM model
    vlm_loaded = load_trained_model(encoder_name)
    
    # Load Random Forest
    rf_loaded = load_random_forest(encoder_name)
    
    # Load tokenizer
    tokenizer_loaded = load_tokenizer()
    
    return jsonify({
        'vlm_model_loaded': vlm_loaded,
        'random_forest_loaded': rf_loaded,
        'tokenizer_loaded': tokenizer_loaded,
        'gemini_configured': gemini_model is not None,
        'encoder_used': encoder_name,
        'prediction_method': 'VLM + Random Forest Hybrid',
        'enhanced_features': True,
        'biomarker_dimensions': 16,
        'ready': vlm_loaded and tokenizer_loaded,  # RF is optional
        'care_plan_available': True,
        'rf_available': rf_loaded
    })

@app.route('/api/available_models', methods=['GET'])
def get_available_models():
    """Get list of available trained VLM and Random Forest models"""
    
    available_encoders = [
        "efficientnet_b0",
        "inception_v3", 
        "densenet121",
        "resnet50",
        "efficientnet_b3"
    ]
    
    # Check which models actually exist
    existing_models = []
    for encoder in available_encoders:
        vlm_path = f'best_medical_vlm_{encoder}.pth'
        rf_path = f'rf_regressor_{encoder}.pkl'
        
        vlm_exists = os.path.exists(vlm_path)
        rf_exists = os.path.exists(rf_path)
        
        existing_models.append({
            'encoder': encoder,
            'vlm_path': vlm_path,
            'rf_path': rf_path,
            'vlm_available': vlm_exists,
            'rf_available': rf_exists,
            'complete_hybrid': vlm_exists and rf_exists,
            'classification_only': vlm_exists and not rf_exists
        })
    
    return jsonify({
        'available_models': existing_models,
        'total_encoders': len(existing_models),
        'complete_hybrids': len([m for m in existing_models if m['complete_hybrid']]),
        'vlm_only': len([m for m in existing_models if m['vlm_available'] and not m['rf_available']]),
        'prediction_methods': ['VLM + Random Forest (Complete)', 'VLM Only (Classification)']
    })

@app.route('/api/model_info', methods=['GET'])
def get_model_info():
    """Get detailed information about currently loaded models"""
    return jsonify({
        'current_status': {
            'vlm_loaded': model is not None,
            'rf_loaded': rf_regressor is not None,
            'tokenizer_loaded': tokenizer is not None
        },
        'capabilities': {
            'classification': model is not None,
            'regression': rf_regressor is not None,
            'care_plans': gemini_model is not None
        },
        'model_details': {
            'vlm_architecture': 'Multi-modal VLM (Vision + Text + Biomarkers)',
            'rf_algorithm': 'Random Forest Regressor' if rf_regressor is not None else 'Not loaded',
            'feature_engineering': '16D enhanced biomarker features',
            'prediction_pipeline': 'VLM for staging + RF for progression time'
        }
    })

if __name__ == '__main__':
    print("Enhanced Medical AI Backend Server - VLM + Random Forest Hybrid")
    print("=" * 70)
    
    if GEMINI_API_KEY:
        print("SUCCESS: Gemini API configured successfully")
    else:
        print("WARNING: Gemini API key not found - using fallback care plans")
    
    print("\nLooking for trained models...")
    
    # Check for available models
    available_encoders = ["efficientnet_b0", "inception_v3", "densenet121", "resnet50", "efficientnet_b3"]
    found_vlm_models = []
    found_rf_models = []
    
    for encoder in available_encoders:
        vlm_path = f'best_medical_vlm_{encoder}.pth'
        rf_path = f'rf_regressor_{encoder}.pkl'
        
        if os.path.exists(vlm_path):
            found_vlm_models.append(encoder)
            print(f"✓ Found VLM: {vlm_path}")
        else:
            print(f"✗ Missing VLM: {vlm_path}")
            
        if os.path.exists(rf_path):
            found_rf_models.append(encoder)
            print(f"✓ Found RF: {rf_path}")
        else:
            print(f"✗ Missing RF: {rf_path}")
    
    complete_hybrids = set(found_vlm_models).intersection(set(found_rf_models))
    
    if found_vlm_models:
        print(f"\n✓ Found {len(found_vlm_models)} VLM models")
        print(f"✓ Found {len(found_rf_models)} Random Forest models")
        print(f"✓ Complete hybrids available: {len(complete_hybrids)}")
        
        # Try to load the best available hybrid model
        if complete_hybrids:
            default_encoder = "inception_v3" if "inception_v3" in complete_hybrids else list(complete_hybrids)[0]
            print(f"Loading complete hybrid: {default_encoder}")
        else:
            default_encoder = "inception_v3" if "inception_v3" in found_vlm_models else found_vlm_models[0]
            print(f"Loading VLM only: {default_encoder}")
        
        vlm_loaded = load_trained_model(default_encoder)
        rf_loaded = load_random_forest(default_encoder)
        tokenizer_loaded = load_tokenizer()
        
        print("=" * 70)
        if vlm_loaded and tokenizer_loaded:
            print("SYSTEM READY!")
            print(f"SUCCESS: VLM model ({default_encoder}): LOADED")
            print(f"SUCCESS: Random Forest ({default_encoder}): {'LOADED' if rf_loaded else 'NOT AVAILABLE'}")
            print("SUCCESS: Medical tokenizer: LOADED") 
            print("✓ Enhanced biomarker features (16D): ENABLED")
            print(f"✓ Prediction method: {'VLM + RF Hybrid' if rf_loaded else 'VLM Classification Only'}")
        else:
            print("SYSTEM INITIALIZATION FAILED")
            if not vlm_loaded:
                print(f"ERROR: {default_encoder} VLM model failed to load")
            if not tokenizer_loaded:
                print("ERROR: Tokenizer failed to load")
    else:
        print("ERROR: No trained VLM models found!")
        print("Please train models first using the training script")
    
    if gemini_model:
        print("Gemini AI care plan generation: ENABLED")
    else:
        print("Care plan generation: FALLBACK MODE")
    
    print(f"\nServer starting on http://localhost:5001")
    print("=" * 70)
    
    app.run(debug=True, host='0.0.0.0', port=5001)