import numpy as np
import librosa
import tensorflow as tf
from tensorflow.keras import layers, models
from pathlib import Path
import joblib

class CNNModel:
    """CNN для классификации/поиска по спектрограммам"""
    
    def __init__(self, input_shape=(128, 128, 1), num_classes=None, db_path="database"):
        self.input_shape = input_shape
        self.num_classes = num_classes
        self.model = None
        self.db_path = Path(db_path)
        self.db_path.mkdir(exist_ok=True)
        self.label_encoder = None
        
    def build_model(self):
        """Построение CNN архитектуры"""
        inputs = layers.Input(shape=self.input_shape)
        x = inputs
        
        # Блок 1
        x = layers.Conv2D(32, (3, 3), padding='same', activation='relu')(x)
        x = layers.BatchNormalization()(x)
        x = layers.Conv2D(32, (3, 3), padding='same', activation='relu')(x)
        x = layers.MaxPooling2D((2, 2))(x)
        x = layers.Dropout(0.25)(x)
        
        # Блок 2
        x = layers.Conv2D(64, (3, 3), padding='same', activation='relu')(x)
        x = layers.BatchNormalization()(x)
        x = layers.Conv2D(64, (3, 3), padding='same', activation='relu')(x)
        x = layers.MaxPooling2D((2, 2))(x)
        x = layers.Dropout(0.25)(x)
        
        # Блок 3
        x = layers.Conv2D(128, (3, 3), padding='same', activation='relu')(x)
        x = layers.BatchNormalization()(x)
        x = layers.Conv2D(128, (3, 3), padding='same', activation='relu')(x)
        x = layers.MaxPooling2D((2, 2))(x)
        x = layers.Dropout(0.3)(x)
        
        # Блок 4
        x = layers.Conv2D(256, (3, 3), padding='same', activation='relu')(x)
        x = layers.BatchNormalization()(x)
        x = layers.GlobalAveragePooling2D()(x)
        
        # Dense слои
        x = layers.Dense(512, activation='relu')(x)
        x = layers.Dropout(0.5)(x)
        x = layers.Dense(256, activation='relu')(x)
        
        # ИСПРАВЛЕНИЕ: Сначала создаем слой, потом применяем его к x
        embedding_dense = layers.Dense(128, activation='linear', name='embedding')
        embeddings = embedding_dense(x)
        
        # Выход для классификации (если нужно)
        if self.num_classes:
            outputs = layers.Dense(self.num_classes, activation='softmax', name='classifier')(embeddings)
            self.model = models.Model(inputs=inputs, outputs=[embeddings, outputs])
        else:
            self.model = models.Model(inputs=inputs, outputs=embeddings)
        
        self.model.compile(
            optimizer=tf.keras.optimizers.Adam(learning_rate=0.001),
            loss='sparse_categorical_crossentropy' if self.num_classes else 'mse',
            metrics=['accuracy'] if self.num_classes else ['mae']
        )
        
        return self.model
    
    def audio_to_spectrogram(self, audio_path, img_size=128):
        """Конвертация аудио в мел-спектрограмму"""
        y, sr = librosa.load(audio_path, sr=22050, duration=10)
        
        mel_spec = librosa.feature.melspectrogram(y=y, sr=sr, n_mels=img_size)
        log_mel = librosa.power_to_db(mel_spec, ref=np.max)
        
        log_mel = (log_mel - log_mel.min()) / (log_mel.max() - log_mel.min() + 1e-8)
        
        if log_mel.shape[1] < img_size:
            pad_width = img_size - log_mel.shape[1]
            log_mel = np.pad(log_mel, ((0, 0), (0, pad_width)), mode='constant')
        else:
            log_mel = log_mel[:, :img_size]
        
        return log_mel.reshape(img_size, img_size, 1)
    
    def get_embedding(self, audio_path):
        """Получение эмбеддинга трека (128-мерный вектор)"""
        if self.model is None:
            self.load()
        
        spec = self.audio_to_spectrogram(audio_path)
        spec_batch = np.expand_dims(spec, axis=0)
        
        # Модель теперь всегда возвращает один тензор (эмбеддинг), если это contrastive модель
        embedding = self.model.predict(spec_batch, verbose=0)
        return embedding.flatten()
    
    def save(self, filename="cnn_model.keras"):
        self.model.save(self.db_path / filename)
        if self.label_encoder:
            joblib.dump(self.label_encoder, self.db_path / "label_encoder.pkl")
        print(f"Модель сохранена: {self.db_path / filename}")
    
    def load(self, filename="cnn_contrastive.keras"):
        model_path = self.db_path / filename
        if model_path.exists():
            self.model = tf.keras.models.load_model(model_path)
            encoder_path = self.db_path / "label_encoder.pkl"
            if encoder_path.exists():
                self.label_encoder = joblib.load(encoder_path)
            print(f"Модель загружена: {model_path}")
        else:
            raise FileNotFoundError(f"Модель не найдена: {model_path}")