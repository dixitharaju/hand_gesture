import os
import numpy as np
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, Dropout, BatchNormalization
from tensorflow.keras.utils import to_categorical
from sklearn.model_selection import train_test_split
from tensorflow.keras.callbacks import EarlyStopping, ReduceLROnPlateau

# Define gestures (labels)
gestures = ['volume_up', 'volume_down', 'mouse_move', 'mouse_right_click']

# ✅ Path to dataset base directory
base_dir = "e:/Sem_5/computer_vision/lab9_10/landmark_dataset"

X, y = [], []

# Load dataset from folders
for idx, gesture in enumerate(gestures):
    folder = os.path.join(base_dir, gesture)
    if not os.path.exists(folder):
        raise FileNotFoundError(f"❌ Folder not found: {folder}")
    for file in os.listdir(folder):
        if file.endswith(".npy"):
            data = np.load(os.path.join(folder, file))
            X.append(data)
            y.append(idx)

X = np.array(X)
y = to_categorical(y, num_classes=len(gestures))

print("✅ Dataset shape:", X.shape, y.shape)

# Train/Test split
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42
)

# Build improved MLP model
model = Sequential([
    Dense(128, activation='relu', input_shape=(X.shape[1],)),
    BatchNormalization(),
    Dropout(0.3),

    Dense(256, activation='relu'),
    BatchNormalization(),
    Dropout(0.3),

    Dense(128, activation='relu'),
    Dense(len(gestures), activation='softmax')
])

model.compile(optimizer='adam', loss='categorical_crossentropy', metrics=['accuracy'])

# Callbacks for better training
early_stop = EarlyStopping(monitor='val_loss', patience=5, restore_best_weights=True)
reduce_lr = ReduceLROnPlateau(monitor='val_loss', factor=0.5, patience=3, verbose=1)

# Train
model.fit(
    X_train, y_train,
    validation_data=(X_test, y_test),
    epochs=50,
    batch_size=16,
    callbacks=[early_stop, reduce_lr],
    verbose=1
)

# Save model
model.save("gesture_model.h5")
print("✅ Model saved as gesture_model.h5")
