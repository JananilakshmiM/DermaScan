import os
import tensorflow as tf
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from tensorflow.keras.callbacks import ModelCheckpoint, EarlyStopping, ReduceLROnPlateau
from model import build_model, NUM_CLASSES

# Define paths (Assumes data is organized in train/val directories by class)
DATA_DIR = '../data'
TRAIN_DIR = os.path.join(DATA_DIR, 'train')
VAL_DIR = os.path.join(DATA_DIR, 'val')
MODEL_SAVE_PATH = 'dermascan_model.h5'

BATCH_SIZE = 32
IMG_SIZE = (224, 224)
EPOCHS = 20

def train():
    # Define data generators with augmentation
    train_datagen = ImageDataGenerator(
        rescale=1./255,
        rotation_range=20,
        width_shift_range=0.2,
        height_shift_range=0.2,
        shear_range=0.2,
        zoom_range=0.2,
        horizontal_flip=True,
        fill_mode='nearest'
    )

    val_datagen = ImageDataGenerator(rescale=1./255)

    if not os.path.exists(TRAIN_DIR):
        print(f"Warning: Training directory {TRAIN_DIR} not found. Please populate data before running.")
        return

    train_generator = train_datagen.flow_from_directory(
        TRAIN_DIR,
        target_size=IMG_SIZE,
        batch_size=BATCH_SIZE,
        class_mode='categorical'
    )

    val_generator = val_datagen.flow_from_directory(
        VAL_DIR,
        target_size=IMG_SIZE,
        batch_size=BATCH_SIZE,
        class_mode='categorical'
    )

    model = build_model(num_classes=NUM_CLASSES)

    # Callbacks
    checkpoint = ModelCheckpoint(
        MODEL_SAVE_PATH, 
        monitor='val_accuracy', 
        save_best_only=True, 
        mode='max',
        verbose=1
    )
    early_stop = EarlyStopping(
        monitor='val_loss', 
        patience=5, 
        restore_best_weights=True
    )
    reduce_lr = ReduceLROnPlateau(
        monitor='val_loss', 
        factor=0.2, 
        patience=3, 
        min_lr=1e-6
    )

    # Train model
    history = model.fit(
        train_generator,
        epochs=EPOCHS,
        validation_data=val_generator,
        callbacks=[checkpoint, early_stop, reduce_lr]
    )
    
    print("Training complete!")

if __name__ == '__main__':
    train()
