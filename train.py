import os
import numpy as np
import matplotlib.pyplot as plt
from pathlib import Path
import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers
from tensorflow.keras.applications import MobileNetV2
from sklearn.metrics import classification_report, confusion_matrix
import seaborn as sns
from collections import Counter

# Set random seeds
np.random.seed(42)
tf.random.set_seed(42)

# Configuration
IMG_SIZE = (224, 224)
BATCH_SIZE = 32
EPOCHS = 20
LEARNING_RATE = 0.0001

# Dataset path
DATASET_PATH = r"C:\Users\ADITI\projects\Fruit_Ripeness\Train"

class FruitFreshnessClassifier:
    def __init__(self, dataset_path, img_size=IMG_SIZE, batch_size=BATCH_SIZE):
        self.dataset_path = dataset_path
        self.img_size = img_size
        self.batch_size = batch_size
        self.model = None
        self.history = None
        self.class_weights = None
        
    def reorganize_data(self):
        """Reorganize data into fresh/rotten structure for binary classification"""
        print("Checking data organization...")
        
        # Count images per category
        fresh_count = 0
        rotten_count = 0
        
        for folder in os.listdir(self.dataset_path):
            folder_path = os.path.join(self.dataset_path, folder)
            if os.path.isdir(folder_path):
                count = len([f for f in os.listdir(folder_path) 
                           if f.lower().endswith(('.jpg', '.jpeg', '.png'))])
                if folder.lower().startswith('fresh'):
                    fresh_count += count
                elif folder.lower().startswith('rotten'):
                    rotten_count += count
        
        print(f"Fresh images: {fresh_count}")
        print(f"Rotten images: {rotten_count}")
        
        # Calculate class weights for imbalance
        total = fresh_count + rotten_count
        if fresh_count > 0 and rotten_count > 0:
            self.class_weights = {
                0: total / (2 * fresh_count),
                1: total / (2 * rotten_count)
            }
            print(f"Class weights: {self.class_weights}")
        
        return fresh_count, rotten_count
    
    def prepare_data(self):
        """Prepare data generators"""
        print("Preparing data generators...")
        
        # Check data first
        self.reorganize_data()
        
        # Create a custom directory structure mapping
        # We'll map all fresh* folders to class 0, rotten* to class 1
        
        # Training data generator with augmentation
        train_datagen = keras.preprocessing.image.ImageDataGenerator(
            rescale=1./255,
            rotation_range=25,
            width_shift_range=0.2,
            height_shift_range=0.2,
            shear_range=0.15,
            zoom_range=0.2,
            horizontal_flip=True,
            brightness_range=[0.8, 1.2],
            fill_mode='nearest',
            validation_split=0.2
        )
        
        # Validation generator
        val_datagen = keras.preprocessing.image.ImageDataGenerator(
            rescale=1./255,
            validation_split=0.2
        )
        
        # Use subdirectory structure
        self.train_generator = train_datagen.flow_from_directory(
            self.dataset_path,
            target_size=self.img_size,
            batch_size=self.batch_size,
            class_mode='categorical',  # Changed to categorical for multiple fruit types
            subset='training',
            shuffle=True
        )
        
        self.val_generator = val_datagen.flow_from_directory(
            self.dataset_path,
            target_size=self.img_size,
            batch_size=self.batch_size,
            class_mode='categorical',
            subset='validation',
            shuffle=False
        )
        
        # Get number of classes
        self.num_classes = len(self.train_generator.class_indices)
        
        print(f"\nTraining samples: {self.train_generator.samples}")
        print(f"Validation samples: {self.val_generator.samples}")
        print(f"Number of classes: {self.num_classes}")
        print(f"Class mapping: {self.train_generator.class_indices}")
        
        # Calculate class distribution
        train_labels = self.train_generator.classes
        class_counts = Counter(train_labels)
        print(f"\nClass distribution in training:")
        for class_idx, count in sorted(class_counts.items()):
            class_name = list(self.train_generator.class_indices.keys())[class_idx]
            print(f"  {class_name}: {count} samples")
        
        return self.train_generator, self.val_generator
    
    def build_model(self):
        """Build model with correct output layer"""
        print("\nBuilding model...")
        
        # Load MobileNetV2
        base_model = MobileNetV2(
            input_shape=(*self.img_size, 3),
            include_top=False,
            weights='imagenet'
        )
        
        base_model.trainable = False
        
        # Build model
        inputs = keras.Input(shape=(*self.img_size, 3))
        x = base_model(inputs, training=False)
        x = layers.GlobalAveragePooling2D()(x)
        x = layers.BatchNormalization()(x)
        x = layers.Dropout(0.5)(x)
        x = layers.Dense(256, activation='relu', kernel_regularizer=keras.regularizers.l2(0.001))(x)
        x = layers.BatchNormalization()(x)
        x = layers.Dropout(0.4)(x)
        x = layers.Dense(128, activation='relu', kernel_regularizer=keras.regularizers.l2(0.001))(x)
        x = layers.Dropout(0.3)(x)
        outputs = layers.Dense(self.num_classes, activation='softmax')(x)
        
        self.model = keras.Model(inputs, outputs)
        
        # Compile
        self.model.compile(
            optimizer=keras.optimizers.Adam(learning_rate=LEARNING_RATE),
            loss='categorical_crossentropy',
            metrics=['accuracy',
                    keras.metrics.Precision(name='precision'),
                    keras.metrics.Recall(name='recall'),
                    keras.metrics.AUC(name='auc')]
        )
        
        print(f"\nModel built with {self.num_classes} output classes")
        print(f"Total parameters: {self.model.count_params():,}")
        
        return self.model
    
    def train(self, epochs=EPOCHS):
        """Train the model"""
        print(f"\nTraining for {epochs} epochs...")
        
        callbacks = [
            keras.callbacks.EarlyStopping(
                monitor='val_loss',
                patience=15,
                restore_best_weights=True,
                verbose=1
            ),
            keras.callbacks.ReduceLROnPlateau(
                monitor='val_loss',
                factor=0.5,
                patience=5,
                min_lr=1e-7,
                verbose=1
            ),
            keras.callbacks.ModelCheckpoint(
                'best_fruit_model.h5',
                monitor='val_accuracy',
                save_best_only=True,
                verbose=1
            ),
            keras.callbacks.TensorBoard(
                log_dir='./logs',
                histogram_freq=1
            )
        ]
        
        self.history = self.model.fit(
            self.train_generator,
            validation_data=self.val_generator,
            epochs=epochs,
            callbacks=callbacks,
            verbose=1
        )
        
        return self.history
    
    def fine_tune(self, epochs=30):
        """Fine-tune model"""
        print("\nFine-tuning model...")
        
        self.model.layers[1].trainable = True
        
        # Freeze early layers
        for layer in self.model.layers[1].layers[:-30]:
            layer.trainable = False
        
        # Compile with lower learning rate
        self.model.compile(
            optimizer=keras.optimizers.Adam(learning_rate=LEARNING_RATE/10),
            loss='categorical_crossentropy',
            metrics=['accuracy',
                    keras.metrics.Precision(name='precision'),
                    keras.metrics.Recall(name='recall'),
                    keras.metrics.AUC(name='auc')]
        )
        
        callbacks = [
            keras.callbacks.EarlyStopping(
                monitor='val_loss',
                patience=10,
                restore_best_weights=True,
                verbose=1
            ),
            keras.callbacks.ModelCheckpoint(
                'best_fruit_model_finetuned.h5',
                monitor='val_accuracy',
                save_best_only=True,
                verbose=1
            )
        ]
        
        history_fine = self.model.fit(
            self.train_generator,
            validation_data=self.val_generator,
            epochs=epochs,
            callbacks=callbacks,
            verbose=1
        )
        
        return history_fine
    
    def plot_training_history(self):
        """Plot training history"""
        if self.history is None:
            print("No training history available!")
            return
        
        fig, axes = plt.subplots(2, 2, figsize=(15, 10))
        
        # Accuracy
        axes[0, 0].plot(self.history.history['accuracy'], label='Train', linewidth=2)
        axes[0, 0].plot(self.history.history['val_accuracy'], label='Validation', linewidth=2)
        axes[0, 0].set_title('Model Accuracy', fontsize=14, fontweight='bold')
        axes[0, 0].set_xlabel('Epoch')
        axes[0, 0].set_ylabel('Accuracy')
        axes[0, 0].legend()
        axes[0, 0].grid(True, alpha=0.3)
        
        # Loss
        axes[0, 1].plot(self.history.history['loss'], label='Train', linewidth=2)
        axes[0, 1].plot(self.history.history['val_loss'], label='Validation', linewidth=2)
        axes[0, 1].set_title('Model Loss', fontsize=14, fontweight='bold')
        axes[0, 1].set_xlabel('Epoch')
        axes[0, 1].set_ylabel('Loss')
        axes[0, 1].legend()
        axes[0, 1].grid(True, alpha=0.3)
        
        # Precision
        axes[1, 0].plot(self.history.history['precision'], label='Train', linewidth=2)
        axes[1, 0].plot(self.history.history['val_precision'], label='Validation', linewidth=2)
        axes[1, 0].set_title('Precision', fontsize=14, fontweight='bold')
        axes[1, 0].set_xlabel('Epoch')
        axes[1, 0].set_ylabel('Precision')
        axes[1, 0].legend()
        axes[1, 0].grid(True, alpha=0.3)
        
        # Recall
        axes[1, 1].plot(self.history.history['recall'], label='Train', linewidth=2)
        axes[1, 1].plot(self.history.history['val_recall'], label='Validation', linewidth=2)
        axes[1, 1].set_title('Recall', fontsize=14, fontweight='bold')
        axes[1, 1].set_xlabel('Epoch')
        axes[1, 1].set_ylabel('Recall')
        axes[1, 1].legend()
        axes[1, 1].grid(True, alpha=0.3)
        
        plt.tight_layout()
        plt.savefig('training_history.png', dpi=300, bbox_inches='tight')
        plt.show()
        
        print("Training history plot saved!")
    
    def evaluate(self):
        """Evaluate model"""
        print("\nEvaluating model...")
        
        # Predictions
        self.val_generator.reset()
        predictions = self.model.predict(self.val_generator, verbose=1)
        y_pred = np.argmax(predictions, axis=1)
        y_true = self.val_generator.classes
        
        # Get class names
        class_names = list(self.val_generator.class_indices.keys())
        
        print("\n" + "=" * 60)
        print("Classification Report:")
        print("=" * 60)
        print(classification_report(y_true, y_pred, target_names=class_names, digits=4))
        
        # Confusion matrix
        cm = confusion_matrix(y_true, y_pred)
        
        plt.figure(figsize=(12, 10))
        sns.heatmap(cm, annot=True, fmt='d', cmap='Blues',
                   xticklabels=class_names, yticklabels=class_names,
                   cbar_kws={'label': 'Count'})
        plt.title('Confusion Matrix', fontsize=16, fontweight='bold', pad=20)
        plt.ylabel('True Label', fontsize=12)
        plt.xlabel('Predicted Label', fontsize=12)
        plt.xticks(rotation=45, ha='right')
        plt.yticks(rotation=0)
        plt.tight_layout()
        plt.savefig('confusion_matrix.png', dpi=300, bbox_inches='tight')
        plt.show()
        
        # Overall metrics
        accuracy = np.sum(y_pred == y_true) / len(y_true)
        print(f"\nOverall Accuracy: {accuracy:.4f} ({accuracy*100:.2f}%)")
        
        return y_pred, y_true
    
    def predict_image(self, image_path):
        """Predict single image"""
        img = keras.preprocessing.image.load_img(image_path, target_size=self.img_size)
        img_array = keras.preprocessing.image.img_to_array(img) / 255.0
        img_array = np.expand_dims(img_array, axis=0)
        
        predictions = self.model.predict(img_array, verbose=0)[0]
        predicted_class = np.argmax(predictions)
        confidence = predictions[predicted_class]
        
        class_names = list(self.train_generator.class_indices.keys())
        result = class_names[predicted_class]
        
        print(f"\nPrediction: {result}")
        print(f"Confidence: {confidence:.2%}")
        print("\nAll probabilities:")
        for i, prob in enumerate(predictions):
            print(f"  {class_names[i]}: {prob:.4f} ({prob*100:.2f}%)")
        
        # Display
        plt.figure(figsize=(8, 8))
        plt.imshow(img)
        plt.title(f"Predicted: {result}\nConfidence: {confidence:.2%}", 
                 fontsize=14, fontweight='bold')
        plt.axis('off')
        plt.tight_layout()
        plt.show()
        
        return result, confidence
    
    def save_model(self, filepath='fruit_model_final.h5'):
        """Save model"""
        self.model.save(filepath)
        print(f"\nModel saved to {filepath}")
    
    def load_model(self, filepath):
        """Load model"""
        self.model = keras.models.load_model(filepath)
        print(f"Model loaded from {filepath}")


def main():
    """Main pipeline"""
    print("=" * 60)
    print("FRUIT FRESHNESS CLASSIFICATION")
    print("=" * 60)
    
    # Initialize
    classifier = FruitFreshnessClassifier(DATASET_PATH)
    
    # Prepare data
    train_gen, val_gen = classifier.prepare_data()
    
    # Build model
    model = classifier.build_model()
    
    # Train
    print("\n" + "=" * 60)
    print("INITIAL TRAINING")
    print("=" * 60)
    classifier.train(epochs=EPOCHS)
    
    # Plot
    classifier.plot_training_history()
    
    # Evaluate
    print("\n" + "=" * 60)
    print("EVALUATION")
    print("=" * 60)
    classifier.evaluate()
    
    # Optional fine-tuning
    response = input("\nDo you want to fine-tune? (y/n): ")
    if response.lower() == 'y':
        print("\n" + "=" * 60)
        print("FINE-TUNING")
        print("=" * 60)
        classifier.fine_tune(epochs=30)
        classifier.evaluate()
    
    # Save
    classifier.save_model()
    
    print("\n" + "=" * 60)
    print("TRAINING COMPLETE!")
    print("=" * 60)


if __name__ == "__main__":
    main()