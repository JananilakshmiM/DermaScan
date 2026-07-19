import tensorflow as tf
from tensorflow.keras.applications import MobileNetV2
from tensorflow.keras.layers import Dense, GlobalAveragePooling2D, Dropout
from tensorflow.keras.models import Model

# Typically 7 classes for HAM10000:
# Actinic keratoses (akiec), basal cell carcinoma (bcc), 
# benign keratosis-like lesions (bkl), dermatofibroma (df), 
# melanoma (mel), melanocytic nevi (nv), vascular lesions (vasc)
NUM_CLASSES = 7

def build_model(input_shape=(224, 224, 3), num_classes=NUM_CLASSES):
    """
    Builds a MobileNetV2-based CNN for skin lesion classification.
    Uses pre-trained weights from ImageNet.
    """
    base_model = MobileNetV2(
        input_shape=input_shape,
        include_top=False,
        weights='imagenet'
    )
    
    # Freeze the base model for initial training
    base_model.trainable = False
    
    x = base_model.output
    x = GlobalAveragePooling2D()(x)
    x = Dropout(0.5)(x)
    x = Dense(128, activation='relu')(x)
    x = Dropout(0.3)(x)
    predictions = Dense(num_classes, activation='softmax')(x)
    
    model = Model(inputs=base_model.input, outputs=predictions)
    
    model.compile(
        optimizer=tf.keras.optimizers.Adam(learning_rate=1e-4),
        loss='categorical_crossentropy',
        metrics=['accuracy']
    )
    
    return model

if __name__ == '__main__':
    model = build_model()
    model.summary()
