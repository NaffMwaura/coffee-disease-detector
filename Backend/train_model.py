import tensorflow as tf
import os
import matplotlib.pyplot as plt
# IMPORTANT: Since we pinned tensorflow==2.10.0, we must use the Keras API
# that comes bundled with it, which is often accessible via tf.keras.
from tensorflow.keras.applications.mobilenet_v2 import MobileNetV2, preprocess_input
from tensorflow.keras.layers import Dense, GlobalAveragePooling2D, Dropout
from tensorflow.keras.models import Model
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from tensorflow.keras.callbacks import EarlyStopping, ModelCheckpoint, ReduceLROnPlateau

# --- Configuration Constants ---
IMG_SIZE = (224, 224) # MobileNetV2 expects 224x224 input
BATCH_SIZE = 32
DATA_DIR = '../Coffee disease dataset/train' # Path confirmed in previous exchange
NUM_CLASSES = 7 # Assuming 7 classes (6 diseases + 1 healthy)
FINE_TUNE_LAYERS = 50 # Unfreeze the last 50 layers for fine-tuning
MODEL_SAVE_PATH = 'best_coffee_disease2_model.h5'

# --- 1. Data Loading and Augmentation ---
def load_and_augment_data():
    """
    Sets up data pipelines with augmentation for training and a generator
    for validation (using a split of the training data).
    """
    print("Loading data and setting up generators...")

    # Data Augmentation for Training: Crucial for preventing overfitting
    # Includes standard transformations for robustness
    train_datagen = ImageDataGenerator(
        preprocessing_function=preprocess_input,
        validation_split=0.2, # Use 20% of data for validation
        rotation_range=30,
        width_shift_range=0.2,
        height_shift_range=0.2,
        shear_range=0.2,
        zoom_range=0.2,
        horizontal_flip=True,
        fill_mode='nearest'
    )

    # Validation generator (only preprocessing, no augmentation)
    val_datagen = ImageDataGenerator(
        preprocessing_function=preprocess_input,
        validation_split=0.2
    )

    try:
        # Training generator
        # Note: flow_from_directory will try to ignore corrupt images during the flow 
        # but the UnidentifiedImageError often happens during initial file indexing.
        train_generator = train_datagen.flow_from_directory(
            DATA_DIR,
            target_size=IMG_SIZE,
            batch_size=BATCH_SIZE,
            class_mode='categorical',
            subset='training',
            shuffle=True
        )

        # Validation generator
        validation_generator = val_datagen.flow_from_directory(
            DATA_DIR,
            target_size=IMG_SIZE,
            batch_size=BATCH_SIZE,
            class_mode='categorical',
            subset='validation',
            shuffle=False
        )
    except Exception as e:
        print(f"An error occurred during file indexing: {e}")
        print("This usually means there is a corrupt or non-image file in your dataset directory.")
        print("Please check all files in the directory for integrity.")
        raise # Re-raise the exception to stop the process if the issue isn't resolved by the environment

    
    print(f"Detected {train_generator.num_classes} classes: {list(train_generator.class_indices.keys())}")
    
    return train_generator, validation_generator

# --- 2. Model Definition ---
def build_transfer_model(num_classes):
    """
    Builds the MobileNetV2 base model and adds a custom classification head.
    """
    # Load MobileNetV2, pre-trained on ImageNet, without the top classification layer
    # Ensured to use tf.keras.applications.mobilenet_v2.MobileNetV2
    base_model = MobileNetV2(
        weights='imagenet', 
        include_top=False, 
        input_shape=IMG_SIZE + (3,) # Adds the 3 color channels
    )

    # --- Add the custom classification head ---
    x = base_model.output
    x = GlobalAveragePooling2D()(x)
    x = Dropout(0.5)(x) # Add Dropout to further prevent overfitting
    x = Dense(256, activation='relu')(x)
    x = Dropout(0.5)(x) # Second Dropout layer
    predictions = Dense(num_classes, activation='softmax')(x) # Output layer

    # Final model combining base and new head
    model = Model(inputs=base_model.input, outputs=predictions)
    
    return model, base_model

# --- 3. Training Function ---
def train_model():
    """
    Implements the two-stage training process (Feature Extraction and Fine-Tuning).
    """
    train_generator, validation_generator = load_and_augment_data()
    
    # Check if the number of classes matches the generator output
    global NUM_CLASSES
    if train_generator.num_classes != NUM_CLASSES:
        print(f"WARNING: NUM_CLASSES constant ({NUM_CLASSES}) does not match generator output ({train_generator.num_classes}). Adjusting to {train_generator.num_classes}.")
        NUM_CLASSES = train_generator.num_classes

    model, base_model = build_transfer_model(NUM_CLASSES)

    # --- Phase 1: Feature Extraction (Training only the top layers) ---
    print("\n--- PHASE 1: FEATURE EXTRACTION (Training Top Layers) ---")
    
    # Freeze the base model layers
    for layer in base_model.layers:
        layer.trainable = False

    # Compile the model with a moderate learning rate
    base_learning_rate = 1e-3 # 0.001
    model.compile(
        optimizer=Adam(learning_rate=base_learning_rate),
        loss='categorical_crossentropy',
        metrics=['accuracy']
    )
    
    # Added a simple print to control output size for large summaries
    print("Model summary (Feature Extraction): Structure defined.")
    
    # Define Callbacks
    callbacks = [
    EarlyStopping(monitor='val_loss', patience=10, restore_best_weights=True),
    # Explicitly use the 'h5' format for maximum compatibility
    ModelCheckpoint(MODEL_SAVE_PATH, monitor='val_accuracy', save_best_only=True, mode='max', save_format='h5'), 
        ]

    initial_epochs = 10
    
    history_feature_extraction = model.fit(
        train_generator,
        epochs=initial_epochs,
        validation_data=validation_generator,
        callbacks=callbacks
    )

    # --- Phase 2: Fine-Tuning (Unfreezing top layers of base model) ---
    print("\n--- PHASE 2: FINE-TUNING (Unfreezing Top Layers of MobileNetV2) ---")

    # Unfreeze the base model
    base_model.trainable = True

    # Freeze all layers up to a specific point (e.g., last 50 layers)
    for layer in base_model.layers[:-FINE_TUNE_LAYERS]:
        layer.trainable = False

    # Ensure the frozen layers are actually non-trainable
    # NOTE: The print below is informational and can be commented out for clean output
    # for layer in base_model.layers:
    #     if layer.name.startswith('block') and layer.trainable:
    #         print(f"Layer {layer.name} is now trainable for fine-tuning.")


    # Compile the model again with a very low learning rate
    fine_tune_learning_rate = 1e-5 # 0.00001 - 100x smaller than the initial rate
    model.compile(
        optimizer=Adam(learning_rate=fine_tune_learning_rate),
        loss='categorical_crossentropy',
        metrics=['accuracy']
    )
    
    # Add a learning rate scheduler for fine-tuning
    callbacks_fine_tune = callbacks + [
        ReduceLROnPlateau(monitor='val_loss', factor=0.2, patience=5, min_lr=1e-7)
    ]

    fine_tune_epochs = 20
    total_epochs = initial_epochs + fine_tune_epochs

    history_fine_tune = model.fit(
        train_generator,
        epochs=total_epochs,
        initial_epoch=history_feature_extraction.epoch[-1],
        validation_data=validation_generator,
        callbacks=callbacks_fine_tune
    )

    print(f"\nTraining complete. The best model weights are saved to {MODEL_SAVE_PATH}")
    
    # Function to plot results (optional but highly recommended)
    plot_history(history_feature_extraction, history_fine_tune, total_epochs)


def plot_history(history_1, history_2, total_epochs):
    """Plots training and validation metrics for both phases."""
    acc = history_1.history['accuracy'] + history_2.history['accuracy']
    val_acc = history_1.history['val_accuracy'] + history_2.history['val_accuracy']

    loss = history_1.history['loss'] + history_2.history['loss']
    val_loss = history_1.history['val_loss'] + history_2.history['val_loss']

    plt.figure(figsize=(10, 8))
    plt.subplot(2, 1, 1)
    plt.plot(acc, label='Training Accuracy')
    plt.plot(val_acc, label='Validation Accuracy')
    plt.ylim([0.0, 1.0])
    plt.plot([len(history_1.epoch)-1, len(history_1.epoch)-1],
             plt.ylim(), label='Start Fine-Tuning', linestyle='--')
    plt.legend(loc='lower right')
    plt.title('Training and Validation Accuracy')

    plt.subplot(2, 1, 2)
    plt.plot(loss, label='Training Loss')
    plt.plot(val_loss, label='Validation Loss')
    plt.plot([len(history_1.epoch)-1, len(history_1.epoch)-1],
             plt.ylim(), label='Start Fine-Tuning', linestyle='--')
    plt.legend(loc='upper right')
    plt.title('Training and Validation Loss')
    plt.xlabel('epoch')
    plt.show()

if __name__ == '__main__':
    # Add the Firebase config setup required for the Canvas environment
    try:
        if 'tf.compat.v1.enable_v2_behavior' in dir(tf.compat.v1):
            tf.compat.v1.enable_v2_behavior()
    except:
        pass # Ignore if running in a non-TF environment

    # Setting up the environment check
    if not os.path.isdir(DATA_DIR):
        print("ERROR: DATA_DIR not found. Please ensure your dataset folder is structured correctly:")
        print("The folder 'Cofee disease dataset' must be one level above the 'Backend' folder.")
        print("Expected path: ../Cofee disease dataset/train")
    else:
        # Added a robust try/except around the train_model call to specifically catch the image error
        try:
            train_model()
        except Exception as e:
            # We already print a more detailed message in load_and_augment_data, 
            # this ensures any failure is clearly logged.
            print(f"\n--- FATAL ERROR DURING TRAINING ---\nTraining stopped due to an error: {e}")