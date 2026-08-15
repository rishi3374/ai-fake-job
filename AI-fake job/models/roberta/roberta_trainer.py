"""
RoBERTa Fine-tuning Module
Fine-tunes RoBERTa model for fake job detection with attention masks and schedulers
"""

import pandas as pd
import numpy as np
from pathlib import Path
import logging
from typing import Dict, Tuple, Optional, List
import json
import os

# Optional tqdm import
try:
    from tqdm import tqdm
    TQDM_AVAILABLE = True
except ImportError:
    TQDM_AVAILABLE = False

# PyTorch and Transformers imports
try:
    import torch
    import torch.nn as nn
    from torch.utils.data import Dataset, DataLoader
    from transformers import (RobertaTokenizer, RobertaForSequenceClassification,
                              AdamW, get_linear_schedule_with_warmup,
                              EarlyStoppingCallback)
    PYTORCH_AVAILABLE = True
except ImportError:
    PYTORCH_AVAILABLE = False

# Configure logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)


class JobDataset:
    """Custom Dataset for job descriptions"""
    
    def __init__(self, texts: List[str], labels: List[int], tokenizer, max_length: int = 512):
        """
        Initialize dataset
        
        Args:
            texts: List of job descriptions
            labels: List of labels (0=real, 1=fake)
            tokenizer: RoBERTa tokenizer
            max_length: Maximum sequence length
        """
        if not PYTORCH_AVAILABLE:
            raise ImportError("PyTorch is required for RoBERTa training")
            
        self.texts = texts
        self.labels = labels
        self.tokenizer = tokenizer
        self.max_length = max_length
    
    def __len__(self):
        return len(self.texts)
    
    def __getitem__(self, idx):
        text = str(self.texts[idx])
        label = self.labels[idx]
        
        # Tokenize with attention mask
        encoding = self.tokenizer.encode_plus(
            text,
            add_special_tokens=True,
            max_length=self.max_length,
            padding='max_length',
            truncation=True,
            return_attention_mask=True,
            return_tensors='pt'
        )
        
        return {
            'input_ids': encoding['input_ids'].flatten(),
            'attention_mask': encoding['attention_mask'].flatten(),
            'labels': torch.tensor(label, dtype=torch.long)
        }


class RoBERTaTrainer:
    """Fine-tune RoBERTa for fake job detection"""
    
    def __init__(self, 
                 model_name: str = 'roberta-base',
                 model_dir: str = "data/models",
                 max_length: int = 512,
                 batch_size: int = 16,
                 learning_rate: float = 2e-5,
                 epochs: int = 3,
                 warmup_steps: int = 500,
                 device: Optional[str] = None):
        """
        Initialize RoBERTa trainer
        
        Args:
            model_name: Hugging Face model name
            model_dir: Directory to save models
            max_length: Maximum sequence length
            batch_size: Training batch size
            learning_rate: Learning rate
            epochs: Number of training epochs
            warmup_steps: Number of warmup steps
            device: Device to use (cuda/cpu)
        """
        if not PYTORCH_AVAILABLE:
            raise ImportError("PyTorch and Transformers are required for RoBERTa training")
        
        self.model_name = model_name
        self.model_dir = Path(model_dir)
        self.model_dir.mkdir(parents=True, exist_ok=True)
        
        self.max_length = max_length
        self.batch_size = batch_size
        self.learning_rate = learning_rate
        self.epochs = epochs
        self.warmup_steps = warmup_steps
        
        # Set device
        if device is None:
            self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        else:
            self.device = torch.device(device)
        
        logger.info(f"Using device: {self.device}")
        
        # Initialize tokenizer and model
        self.tokenizer = RobertaTokenizer.from_pretrained(model_name)
        self.model = RobertaForSequenceClassification.from_pretrained(
            model_name,
            num_labels=2,
            problem_type="single_label_classification"
        )
        self.model.to(self.device)
        
        # Training history
        self.training_history = {
            'train_loss': [],
            'val_loss': [],
            'train_acc': [],
            'val_acc': []
        }
    
    def prepare_data(self, df: pd.DataFrame, text_column: str = 'description') -> Tuple:
        """
        Prepare data for training
        
        Args:
            df: Input dataframe
            text_column: Name of text column
            
        Returns:
            Tuple of (train_loader, val_loader, test_loader)
        """
        logger.info("Preparing data for RoBERTa training...")
        
        # Extract texts and labels
        texts = df[text_column].fillna('').tolist()
        labels = df['fraudulent'].tolist()
        
        # Split data
        from sklearn.model_selection import train_test_split
        
        train_texts, temp_texts, train_labels, temp_labels = train_test_split(
            texts, labels, test_size=0.3, random_state=42, stratify=labels
        )
        
        val_texts, test_texts, val_labels, test_labels = train_test_split(
            temp_texts, temp_labels, test_size=0.5, random_state=42, stratify=temp_labels
        )
        
        logger.info(f"Train samples: {len(train_texts)}, Val samples: {len(val_texts)}, Test samples: {len(test_texts)}")
        
        # Create datasets
        train_dataset = JobDataset(train_texts, train_labels, self.tokenizer, self.max_length)
        val_dataset = JobDataset(val_texts, val_labels, self.tokenizer, self.max_length)
        test_dataset = JobDataset(test_texts, test_labels, self.tokenizer, self.max_length)
        
        # Create dataloaders
        train_loader = DataLoader(train_dataset, batch_size=self.batch_size, shuffle=True)
        val_loader = DataLoader(val_dataset, batch_size=self.batch_size, shuffle=False)
        test_loader = DataLoader(test_dataset, batch_size=self.batch_size, shuffle=False)
        
        return train_loader, val_loader, test_loader
    
    def train_epoch(self, train_loader, optimizer, scheduler) -> Tuple[float, float]:
        """
        Train for one epoch
        
        Args:
            train_loader: Training data loader
            optimizer: Optimizer
            scheduler: Learning rate scheduler
            
        Returns:
            Tuple of (average loss, accuracy)
        """
        self.model.train()
        total_loss = 0
        correct_predictions = 0
        total_predictions = 0
        
        progress_bar = tqdm(train_loader, desc="Training") if TQDM_AVAILABLE else train_loader
        
        for batch in progress_bar:
            # Move batch to device
            input_ids = batch['input_ids'].to(self.device)
            attention_mask = batch['attention_mask'].to(self.device)
            labels = batch['labels'].to(self.device)
            
            # Forward pass
            self.model.zero_grad()
            outputs = self.model(
                input_ids=input_ids,
                attention_mask=attention_mask,
                labels=labels
            )
            
            loss = outputs.loss
            logits = outputs.logits
            
            # Backward pass
            loss.backward()
            torch.nn.utils.clip_grad_norm_(self.model.parameters(), max_norm=1.0)
            optimizer.step()
            scheduler.step()
            
            # Calculate metrics
            total_loss += loss.item()
            predictions = torch.argmax(logits, dim=1)
            correct_predictions += (predictions == labels).sum().item()
            total_predictions += labels.size(0)
            
            # Update progress bar
            if TQDM_AVAILABLE:
                progress_bar.set_postfix({
                    'loss': f'{loss.item():.4f}',
                    'acc': f'{correct_predictions/total_predictions:.4f}'
                })
        
        avg_loss = total_loss / len(train_loader)
        accuracy = correct_predictions / total_predictions
        
        return avg_loss, accuracy
    
    def validate(self, val_loader) -> Tuple[float, float]:
        """
        Validate the model
        
        Args:
            val_loader: Validation data loader
            
        Returns:
            Tuple of (average loss, accuracy)
        """
        self.model.eval()
        total_loss = 0
        correct_predictions = 0
        total_predictions = 0
        
        with torch.no_grad():
            progress_bar = tqdm(val_loader, desc="Validating") if TQDM_AVAILABLE else val_loader
            
            for batch in progress_bar:
                # Move batch to device
                input_ids = batch['input_ids'].to(self.device)
                attention_mask = batch['attention_mask'].to(self.device)
                labels = batch['labels'].to(self.device)
                
                # Forward pass
                outputs = self.model(
                    input_ids=input_ids,
                    attention_mask=attention_mask,
                    labels=labels
                )
                
                loss = outputs.loss
                logits = outputs.logits
                
                # Calculate metrics
                total_loss += loss.item()
                predictions = torch.argmax(logits, dim=1)
                correct_predictions += (predictions == labels).sum().item()
                total_predictions += labels.size(0)
                
                # Update progress bar
                if TQDM_AVAILABLE:
                    progress_bar.set_postfix({
                        'loss': f'{loss.item():.4f}',
                        'acc': f'{correct_predictions/total_predictions:.4f}'
                    })
        
        avg_loss = total_loss / len(val_loader)
        accuracy = correct_predictions / total_predictions
        
        return avg_loss, accuracy
    
    def train(self, df: pd.DataFrame, text_column: str = 'description', early_stopping_patience: int = 3) -> Dict:
        """
        Train RoBERTa model
        
        Args:
            df: Input dataframe
            text_column: Name of text column
            early_stopping_patience: Patience for early stopping
            
        Returns:
            Dictionary with training results
        """
        logger.info("Starting RoBERTa fine-tuning...")
        
        # Prepare data
        train_loader, val_loader, test_loader = self.prepare_data(df, text_column)
        
        # Setup optimizer and scheduler
        optimizer = AdamW(self.model.parameters(), lr=self.learning_rate, correct_bias=False)
        total_steps = len(train_loader) * self.epochs
        scheduler = get_linear_schedule_with_warmup(
            optimizer,
            num_warmup_steps=self.warmup_steps,
            num_training_steps=total_steps
        )
        
        # Early stopping
        best_val_loss = float('inf')
        patience_counter = 0
        best_model_state = None
        
        # Training loop
        for epoch in range(self.epochs):
            logger.info(f"\nEpoch {epoch + 1}/{self.epochs}")
            
            # Train
            train_loss, train_acc = self.train_epoch(train_loader, optimizer, scheduler)
            
            # Validate
            val_loss, val_acc = self.validate(val_loader)
            
            # Store history
            self.training_history['train_loss'].append(train_loss)
            self.training_history['val_loss'].append(val_loss)
            self.training_history['train_acc'].append(train_acc)
            self.training_history['val_acc'].append(val_acc)
            
            logger.info(f"Train Loss: {train_loss:.4f}, Train Acc: {train_acc:.4f}")
            logger.info(f"Val Loss: {val_loss:.4f}, Val Acc: {val_acc:.4f}")
            
            # Early stopping
            if val_loss < best_val_loss:
                best_val_loss = val_loss
                patience_counter = 0
                best_model_state = self.model.state_dict().copy()
                logger.info("New best model saved!")
            else:
                patience_counter += 1
                logger.info(f"No improvement. Patience: {patience_counter}/{early_stopping_patience}")
                
                if patience_counter >= early_stopping_patience:
                    logger.info("Early stopping triggered!")
                    break
        
        # Load best model
        if best_model_state is not None:
            self.model.load_state_dict(best_model_state)
        
        # Final evaluation on test set
        test_loss, test_acc = self.validate(test_loader)
        logger.info(f"Test Loss: {test_loss:.4f}, Test Acc: {test_acc:.4f}")
        
        results = {
            'training_history': self.training_history,
            'test_loss': test_loss,
            'test_accuracy': test_acc,
            'best_val_loss': best_val_loss
        }
        
        return results
    
    def save_model(self, model_name: str = "roberta_fake_job_detector") -> Path:
        """
        Save trained model and tokenizer
        
        Args:
            model_name: Name for saved model
            
        Returns:
            Path to saved model directory
        """
        model_path = self.model_dir / model_name
        model_path.mkdir(exist_ok=True)
        
        # Save model
        self.model.save_pretrained(model_path)
        
        # Save tokenizer
        self.tokenizer.save_pretrained(model_path)
        
        # Save training history
        with open(model_path / "training_history.json", 'w') as f:
            json.dump(self.training_history, f, indent=2)
        
        # Save configuration
        config = {
            'model_name': self.model_name,
            'max_length': self.max_length,
            'batch_size': self.batch_size,
            'learning_rate': self.learning_rate,
            'epochs': self.epochs,
            'warmup_steps': self.warmup_steps
        }
        
        with open(model_path / "config.json", 'w') as f:
            json.dump(config, f, indent=2)
        
        logger.info(f"Model saved to {model_path}")
        
        return model_path
    
    def load_model(self, model_path: str):
        """
        Load trained model
        
        Args:
            model_path: Path to saved model directory
        """
        model_path = Path(model_path)
        
        # Load model
        self.model = RobertaForSequenceClassification.from_pretrained(model_path)
        self.model.to(self.device)
        
        # Load tokenizer
        self.tokenizer = RobertaTokenizer.from_pretrained(model_path)
        
        # Load configuration
        config_path = model_path / "config.json"
        if config_path.exists():
            with open(config_path, 'r') as f:
                config = json.load(f)
                self.max_length = config.get('max_length', 512)
        
        logger.info(f"Model loaded from {model_path}")
    
    def predict(self, text: str) -> Dict:
        """
        Make prediction on single text
        
        Args:
            text: Input text
            
        Returns:
            Dictionary with prediction results
        """
        self.model.eval()
        
        # Tokenize
        encoding = self.tokenizer.encode_plus(
            text,
            add_special_tokens=True,
            max_length=self.max_length,
            padding='max_length',
            truncation=True,
            return_attention_mask=True,
            return_tensors='pt'
        )
        
        # Move to device
        input_ids = encoding['input_ids'].to(self.device)
        attention_mask = encoding['attention_mask'].to(self.device)
        
        # Predict
        with torch.no_grad():
            outputs = self.model(input_ids=input_ids, attention_mask=attention_mask)
            logits = outputs.logits
            probabilities = torch.softmax(logits, dim=1)
            prediction = torch.argmax(probabilities, dim=1).item()
            confidence = probabilities[0][prediction].item()
        
        return {
            'prediction': prediction,
            'is_fake': bool(prediction == 1),
            'confidence': confidence,
            'probabilities': {
                'real': probabilities[0][0].item(),
                'fake': probabilities[0][1].item()
            }
        }
    
    def predict_batch(self, texts: List[str]) -> List[Dict]:
        """
        Make predictions on batch of texts
        
        Args:
            texts: List of input texts
            
        Returns:
            List of prediction results
        """
        results = []
        
        if TQDM_AVAILABLE:
            for text in tqdm(texts, desc="Predicting"):
                result = self.predict(text)
                results.append(result)
        else:
            for text in texts:
                result = self.predict(text)
                results.append(result)
        
        return results


def main():
    """Main execution function for testing"""
    if not PYTORCH_AVAILABLE:
        logger.warning("PyTorch and Transformers are required for RoBERTa training")
        logger.info("Skipping RoBERTa training - will be available after installing dependencies")
        return
    
    # Create sample data for testing
    sample_data = pd.DataFrame({
        'description': [
            'Senior Software Engineer position requiring 5+ years Python experience. Competitive salary and benefits.',
            'URGENT! Earn $5000 weekly working from home. No experience needed. Start today!',
            'Marketing Manager at TechCorp. MBA required. $80,000-$95,000 salary.',
            'Easy money! Work from anywhere and earn $10000 per month with zero investment.',
            'Data Analyst position with SQL, Python skills. 2+ years experience. $70,000 annually.',
            'Immediate hiring! High paying job with no qualifications required.',
            'DevOps Engineer with AWS, Kubernetes experience. $100,000-$125,000.',
            'Get rich overnight! Investment opportunity with 100% returns guaranteed.',
            'UX Designer position. Portfolio required. $75,000-$90,000.',
            'Work from home and earn $8000 weekly. Limited spots available!'
        ],
        'fraudulent': [0, 1, 0, 1, 0, 1, 0, 1, 0, 1]
    })
    
    # Train RoBERTa
    trainer = RoBERTaTrainer(
        epochs=2,  # Reduced for testing
        batch_size=4,  # Reduced for testing
        max_length=128  # Reduced for testing
    )
    
    results = trainer.train(sample_data)
    
    # Save model
    model_path = trainer.save_model("roberta_fake_job_detector")
    
    # Test prediction
    print("\n=== Testing Prediction ===")
    test_text = "Earn $3000 daily working from home. No experience needed. Apply now!"
    prediction = trainer.predict(test_text)
    print(f"Test: {test_text}")
    print(f"Prediction: {prediction}")


if __name__ == "__main__":
    main()
