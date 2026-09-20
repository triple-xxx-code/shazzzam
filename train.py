import numpy as np
import librosa
import tensorflow as tf
from tensorflow.keras import layers
from pathlib import Path
from cnn_model import CNNModel
from augmentation import AudioAugmenter
from sklearn.model_selection import train_test_split
import soundfile as sf
import matplotlib
matplotlib.use('Agg')  # Принудительно отключаем GUI, используем только рендер в файл
import matplotlib.pyplot as plt

class TripletMetricsCallback(tf.keras.callbacks.Callback):
    """
    Кастомный callback для оценки качества контрастного обучения.
    Вычисляет процент триплетов, где расстояние до Positive меньше, чем до Negative.
    Это аналог Accuracy/матрицы ошибок для задач метрического обучения.
    """
    def __init__(self, val_data, margin=0.5):
        super().__init__()
        self.val_anchors, self.val_positives, self.val_negatives = val_data
        self.margin = margin
        self.val_accuracies = []

    def on_epoch_end(self, epoch, logs=None):
        # Предсказываем эмбеддинги для валидационной выборки
        preds = self.model.predict([self.val_anchors, self.val_positives, self.val_negatives], verbose=0)
        emb_dim = preds.shape[1] // 3
        
        # Разделяем склеенный вывод обратно на 3 части
        a_emb = preds[:, :emb_dim]
        p_emb = preds[:, emb_dim:2*emb_dim]
        n_emb = preds[:, 2*emb_dim:]
        
        # Вычисляем евклидовы расстояния в квадрате (как в функции потерь)
        pos_dist = np.sum(np.square(a_emb - p_emb), axis=-1)
        neg_dist = np.sum(np.square(a_emb - n_emb), axis=-1)
        
        # Считаем, сколько раз модель правильно ранжировала триплет
        # (расстояние до positive меньше, чем до negative)
        correct_predictions = np.sum(pos_dist < neg_dist)
        total_predictions = len(pos_dist)
        
        accuracy = correct_predictions / total_predictions
        self.val_accuracies.append(accuracy)
        
        # Добавляем в логи Keras, чтобы они отображались в прогресс-баре и на графиках
        if logs is not None:
            logs['val_triplet_acc'] = accuracy
            
        print(f"\n— Val Triplet Accuracy: {accuracy:.4f} ({correct_predictions}/{total_predictions} верных ранжирований)")


class CNNTrainer:
    """Обучение CNN модели с аугментацией (только контрастное обучение)"""
    
    def __init__(self, db_path="database", img_size=128):
        self.db_path = Path(db_path)
        self.img_size = img_size
        self.cnn = CNNModel(input_shape=(img_size, img_size, 1), db_path=db_path)
        self.augmenter = AudioAugmenter()
    
    def prepare_contrastive_data(self, music_folder, samples_per_track=5):
        """Подготовка данных для контрастного обучения"""
        music_path = Path(music_folder)
        extensions = ['.mp3', '.wav', '.flac', '.ogg', '.m4a']
        
        tracks = sorted([str(f) for f in music_path.rglob("*") if f.suffix.lower() in extensions])
        print(f"Найдено {len(tracks)} треков")
        
        anchors, positives, negatives = [], [], []
        cnn_temp = CNNModel(input_shape=(self.img_size, self.img_size, 1), db_path=str(self.db_path))
        
        for i, track in enumerate(tracks):
            print(f"Обработка {i+1}/{len(tracks)}: {Path(track).name}")
            
            try:
                anchor_spec = cnn_temp.audio_to_spectrogram(track, self.img_size)
            except Exception as e:
                print(f"  Пропуск {track}: {e}")
                continue
            
            for _ in range(samples_per_track):
                # Positive
                aug_audio = self.augmenter.augment(track)
                if isinstance(aug_audio, tuple):
                    y, sr = aug_audio
                    temp_path = self.db_path / "temp_aug.wav"
                    sf.write(temp_path, y, sr)
                    try:
                        positive_spec = cnn_temp.audio_to_spectrogram(str(temp_path), self.img_size)
                    except Exception as e:
                        print(f"  Ошибка аугментации: {e}")
                        temp_path.unlink(missing_ok=True)
                        continue
                    temp_path.unlink(missing_ok=True)
                else:
                    continue
                
                # Negative
                other_tracks = [t for t in tracks if t != track]
                if not other_tracks:
                    continue
                neg_track = np.random.choice(other_tracks)
                
                try:
                    negative_spec = cnn_temp.audio_to_spectrogram(neg_track, self.img_size)
                except Exception:
                    continue
                
                anchors.append(anchor_spec)
                positives.append(positive_spec)
                negatives.append(negative_spec)
        
        return np.array(anchors), np.array(positives), np.array(negatives)
    
    def train_contrastive(self, music_folder, epochs=50, batch_size=32):
        """Обучение с контрастным loss (Triplet Loss) через Siamese Network"""
        print("Подготовка данных...")
        anchors, positives, negatives = self.prepare_contrastive_data(music_folder)
        print(f"Всего создано триплетов: {len(anchors)}")
        
        # Явное разделение на train и val для работы Early Stopping и кастомного Callback
        a_train, a_val, p_train, p_val, n_train, n_val = train_test_split(
            anchors, positives, negatives, test_size=0.15, random_state=42
        )
        print(f"Train триплетов: {len(a_train)}, Val триплетов: {len(a_val)}")
        
        input_shape = (self.img_size, self.img_size, 1)
        
        # 1. Создаем "тело" сети (Base Network)
        def build_base_network():
            inputs = tf.keras.Input(shape=input_shape)
            x = layers.Conv2D(32, (3, 3), padding='same', activation='relu')(inputs)
            x = layers.BatchNormalization()(x)
            x = layers.Conv2D(32, (3, 3), padding='same', activation='relu')(x)
            x = layers.MaxPooling2D((2, 2))(x)
            x = layers.Dropout(0.25)(x)
            
            x = layers.Conv2D(64, (3, 3), padding='same', activation='relu')(x)
            x = layers.BatchNormalization()(x)
            x = layers.Conv2D(64, (3, 3), padding='same', activation='relu')(x)
            x = layers.MaxPooling2D((2, 2))(x)
            x = layers.Dropout(0.25)(x)
            
            x = layers.Conv2D(128, (3, 3), padding='same', activation='relu')(x)
            x = layers.BatchNormalization()(x)
            x = layers.Conv2D(128, (3, 3), padding='same', activation='relu')(x)
            x = layers.MaxPooling2D((2, 2))(x)
            x = layers.Dropout(0.3)(x)
            
            x = layers.Conv2D(256, (3, 3), padding='same', activation='relu')(x)
            x = layers.BatchNormalization()(x)
            x = layers.GlobalAveragePooling2D()(x)
            
            x = layers.Dense(512, activation='relu')(x)
            x = layers.Dropout(0.5)(x)
            x = layers.Dense(256, activation='relu')(x)
            embeddings = layers.Dense(128, activation='linear', name='embedding')(x)
            return tf.keras.Model(inputs, embeddings, name="base_network")

        base_network = build_base_network()
        
        # 2. Создаем 3 отдельных входа для триплета
        anchor_input = tf.keras.Input(shape=input_shape, name="anchor_input")
        positive_input = tf.keras.Input(shape=input_shape, name="positive_input")
        negative_input = tf.keras.Input(shape=input_shape, name="negative_input")
        
        # Пропускаем через ОДНУ и ту же сеть (веса общие!)
        anchor_emb = base_network(anchor_input)
        positive_emb = base_network(positive_input)
        negative_emb = base_network(negative_input)
        
        # Склеиваем выходы для удобства расчета loss
        concatenated = layers.Concatenate()([anchor_emb, positive_emb, negative_emb])
        
        # Итоговая модель для обучения
        self.cnn.model = tf.keras.Model(
            inputs=[anchor_input, positive_input, negative_input],
            outputs=concatenated
        )
        
        # 3. Triplet Loss
        margin = 0.5
        def triplet_loss(y_true, y_pred):
            emb_dim = 128
            anchor = y_pred[:, :emb_dim]
            positive = y_pred[:, emb_dim:2*emb_dim]
            negative = y_pred[:, 2*emb_dim:]
            
            pos_dist = tf.reduce_sum(tf.square(anchor - positive), axis=-1)
            neg_dist = tf.reduce_sum(tf.square(anchor - negative), axis=-1)
            return tf.reduce_mean(tf.maximum(pos_dist - neg_dist + margin, 0.0))
        
        self.cnn.model.compile(
            optimizer=tf.keras.optimizers.Adam(0.0005),
            loss=triplet_loss
        )
        
        # 4. Callbacks: Ранняя остановка + Наша кастомная метрика
        callbacks = [
            tf.keras.callbacks.EarlyStopping(
                monitor='val_loss', 
                patience=5, 
                restore_best_weights=True,
                verbose=1
            ),
            tf.keras.callbacks.ReduceLROnPlateau(
                monitor='val_loss', 
                factor=0.5, 
                patience=4, 
                verbose=1
            ),
            TripletMetricsCallback(val_data=(a_val, p_val, n_val), margin=margin)
        ]
        
        # 5. Обучение
        history = self.cnn.model.fit(
            x=[a_train, p_train, n_train],
            y=np.zeros(len(a_train)), # dummy labels
            validation_data=([a_val, p_val, n_val], np.zeros(len(a_val))),
            epochs=epochs,
            batch_size=batch_size,
            callbacks=callbacks,
            shuffle=True
        )
        
        # 6. Построение графиков обучения
        self._plot_training_history(history)
        
        # 7. Сохраняем ТОЛЬКО base_network для использования в search.py и index.py
        save_path = self.db_path / "cnn_contrastive.keras"
        base_network.save(save_path)
        print(f"✅ Базовая модель для поиска сохранена: {save_path}")
        
        return history

    def _plot_training_history(self, history):
        """Построение и сохранение графиков обучения"""
        plt.figure(figsize=(14, 5))
        
        # График потерь (Loss)
        plt.subplot(1, 2, 1)
        plt.plot(history.history['loss'], label='Train Triplet Loss', linewidth=2)
        plt.plot(history.history['val_loss'], label='Val Triplet Loss', linewidth=2)
        plt.title('Triplet Loss по эпохам')
        plt.xlabel('Эпоха')
        plt.ylabel('Loss')
        plt.legend()
        plt.grid(True, alpha=0.3)
        
        # График метрики точности триплетов (аналог Accuracy)
        plt.subplot(1, 2, 2)
        plt.plot(history.history['val_triplet_acc'], label='Val Triplet Accuracy', color='green', linewidth=2)
        plt.title('Точность ранжирования (Pos < Neg)')
        plt.xlabel('Эпоха')
        plt.ylabel('Accuracy')
        plt.legend()
        plt.grid(True, alpha=0.3)
        plt.ylim([0.0, 1.05])
        
        plt.tight_layout()
        plot_path = self.db_path / "training_history.png"
        plt.savefig(plot_path, dpi=150)
        print(f"📊 График обучения сохранен: {plot_path}")
        # plt.show()
        # Очищаем память от графических объектов, чтобы избежать конфликтов с cv2
        plt.close('all')


if __name__ == "__main__":
    trainer = CNNTrainer()
    
    # Запуск контрастного обучения
    # Увеличил epochs до 50, но EarlyStopping остановит обучение, если улучшений нет 5 эпох подряд
    trainer.train_contrastive("music_library", epochs=50, batch_size=16)