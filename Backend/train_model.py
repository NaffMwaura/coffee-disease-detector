import tensorflow as tf
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from tensorflow.keras.applications import MobileNetV2
from tensorflow.keras.layers import GlobalAveragePooling2D, Dense, Dropout
from tensorflow.keras.models import Model
from tensorflow.keras.callbacks import ModelCheckpoint, EarlyStopping
import os
import numpy as np
from sklearn.utils import class_weight
import math # Import the math library for ceil calculation

# --- Configuration ---
BATCH_SIZE = 32
IMG_HEIGHT = 224
IMG_WIDTH = 224
# Set to a higher number of epochs for a better chance of hitting max accuracy before EarlyStopping
EPOCHS = 50 
# FIX APPLIED: Ensure the model saves in the native Keras format (.keras) 
# to suppress the legacy file format warning and use the recommended standard.
MODEL_PATH = "./coffeescan1_model_final.keras" 
# FIX: Corrected path name to exactly match user's directory name ("coffee disease dataset")
# The '../' is necessary because the script is running inside the 'Backend' folder.
DATA_DIR = "../coffee disease dataset/train/" 

# The class names must match your directory names and the list in the backend
CLASS_NAMES = [
    'Cerscospora', 'Other_Non_Coffee_Leaf', 'coffee___healthy', 
    'coffee___red_spider_mite', 'coffee___rust', 'miner', 'phoma'
]
NUM_CLASSES = len(CLASS_NAMES)
# ---------------------

print("TensorFlow Version:", tf.__version__)

# --- 1. Data Preparation with Advanced Augmentation ---
# Create an ImageDataGenerator instance for training data with enhanced augmentation
train_datagen = ImageDataGenerator(
    rescale=1./255,
    # FIX: Added aggressive augmentation for better generalization, crucial for visual confusion
    rotation_range=20,
    width_shift_range=0.1,
    height_shift_range=0.1,
    shear_range=0.1,
    zoom_range=0.2,
    horizontal_flip=True, # Added flip
    fill_mode='nearest',
    validation_split=0.2 # Use 20% of data for validation
)

# Generator for training data
train_generator = train_datagen.flow_from_directory(
    DATA_DIR,
    target_size=(IMG_HEIGHT, IMG_WIDTH),
    batch_size=BATCH_SIZE,
    class_mode='categorical',
    subset='training',
    seed=42 # Ensure reproducibility
)

# Generator for validation data (only scaling and subset separation)
validation_generator = train_datagen.flow_from_directory(
    DATA_DIR,
    target_size=(IMG_HEIGHT, IMG_WIDTH),
    batch_size=BATCH_SIZE,
    class_mode='categorical',
    subset='validation',
    seed=42
)

# --- 2. Calculate Class Weights (Fixing Data Imbalance) ---
# This step automatically gives confusing/rare classes more weight in the loss function.
def calculate_class_weights(generator):
    """Calculates class weights based on the training data distribution."""
    # Get all class indices for the training data
    class_indices = generator.classes
    
    # Calculate balanced weights
    weights = class_weight.compute_class_weight(
        'balanced', 
        classes=np.unique(class_indices), 
        y=class_indices
    )
    # Map weights to a dictionary where keys are class indices
    return dict(enumerate(weights))

class_weights = calculate_class_weights(train_generator)
print("Calculated Class Weights:", class_weights)

# --- 3. Model Architecture and Fine-Tuning (Fixing Feature Extraction) ---

def build_fine_tuned_model(input_shape=(IMG_HEIGHT, IMG_WIDTH, 3), num_classes=NUM_CLASSES):
    """
    Builds the MobileNetV2 model with a custom head and prepares for fine-tuning.
    We are matching the architecture confirmed in the backend model_trainer.py.
    """
    # Load the base MobileNetV2 model pre-trained on ImageNet
    base_model = MobileNetV2(
        input_shape=input_shape,
        include_top=False, # We use our own classification head
        weights='imagenet'
    )
    
    # --- Transfer Learning Phase (Freeze All Base Layers) ---
    base_model.trainable = False 
    print("Initial phase: Base MobileNetV2 layers are frozen.")
    

    # Define the custom classification head using the Functional API
    x = base_model.output
    x = GlobalAveragePooling2D()(x) 
    x = Dropout(0.5)(x) 
    x = Dense(256, activation='relu')(x) # Dense layer as confirmed in the backend
    x = Dropout(0.5)(x) 
    predictions = Dense(num_classes, activation='softmax')(x) 

    model = Model(inputs=base_model.input, outputs=predictions)
    
    # Compile the model for the first (transfer learning) phase
    model.compile(
        optimizer=tf.keras.optimizers.Adam(learning_rate=0.001), 
        loss='categorical_crossentropy', 
        metrics=['accuracy']
    )
    
    # --- Fine-Tuning Setup Phase (Unfreeze the last block) ---
    # Unfreeze the last ~50 layers (to allow learning coffee-specific features)
    base_model.trainable = True
    
    # Freeze all layers *except* the ones in the last block (index -50)
    for layer in base_model.layers[:-50]:
        layer.trainable = False
        
    print(f"Fine-tuning phase: The last {len(base_model.layers) - 50} layers are unfrozen.")
    

    # Re-compile with a lower learning rate for fine-tuning
    model.compile(
        optimizer=tf.keras.optimizers.Adam(learning_rate=1e-5), # CRITICAL: Very low LR for fine-tuning
        loss='categorical_crossentropy',
        metrics=['accuracy']
    )
    
    return model

# Build the model
model = build_fine_tuned_model()
model.summary()


# --- 4. Callbacks and Training ---

# ModelCheckpoint: Save only the best model based on validation accuracy
checkpoint_callback = ModelCheckpoint(
    MODEL_PATH, 
    monitor='val_accuracy', 
    save_best_only=True, 
    mode='max', 
    verbose=1
)

# EarlyStopping: Stop training if validation accuracy doesn't improve for 10 epochs
early_stopping_callback = EarlyStopping(
    monitor='val_accuracy', 
    patience=10, 
    restore_best_weights=True, 
    mode='max'
)

# Calculate steps per epoch
# FIX: Use math.ceil to round up, ensuring the final partial batch is processed, 
# which avoids the "input ran out of data" warning.
step_size_train = math.ceil(train_generator.n / train_generator.batch_size)
step_size_valid = math.ceil(validation_generator.n / validation_generator.batch_size)

print("\n--- Starting Model Training (Fine-Tuning Phase) ---\n")

history = model.fit(
    train_generator,
    steps_per_epoch=step_size_train,
    epochs=EPOCHS,
    validation_data=validation_generator,
    validation_steps=step_size_valid,
    callbacks=[checkpoint_callback, early_stopping_callback],
    class_weight=class_weights, # FIX: Applying calculated weights here
    verbose=1
)

print(f"\nTraining Complete. Best model saved to {MODEL_PATH}")
print(f"Best Validation Accuracy achieved: {max(history.history['val_accuracy']):.4f}")