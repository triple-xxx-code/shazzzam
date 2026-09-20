import numpy as np
import librosa
import soundfile as sf
from pathlib import Path
import random

class AudioAugmenter:
    """Аугментация аудио данных для обучения"""
    
    def __init__(self, sr=22050):
        self.sr = sr
    
    def add_noise(self, y, noise_level=0.005):
        """Добавление гауссова шума"""
        noise = np.random.randn(len(y)) * noise_level
        return y + noise
    
    def pitch_shift(self, y, sr, n_steps=2):
        """Изменение высоты тона (питч-шифт)"""
        return librosa.effects.pitch_shift(y=y, sr=sr, n_steps=n_steps)
    
    def time_stretch(self, y, rate=1.1):
        """Изменение темпа без изменения высоты"""
        return librosa.effects.time_stretch(y, rate=rate)
    
    def add_reverb(self, y, decay=0.5):
        """Простой эффект реверберации"""
        impulse = np.exp(-np.arange(len(y)) * decay / len(y))
        impulse = impulse / np.sum(impulse)
        return np.convolve(y, impulse, mode='same')
    
    def random_gain(self, y, min_db=-6, max_db=6):
        """Случайное усиление/ослабление громкости"""
        gain_db = random.uniform(min_db, max_db)
        gain = 10 ** (gain_db / 20)
        return y * gain
    
    def augment(self, audio_path, output_path=None, augmentations=None):
        """
        Применение случайной аугментации
        augmentations: список методов ['noise', 'pitch', 'tempo', 'reverb', 'gain']
        """
        y, sr = librosa.load(audio_path, sr=self.sr)
        
        if augmentations is None:
            augmentations = ['noise', 'pitch', 'tempo']
        
        # Применяем случайные аугментации
        if 'noise' in augmentations and random.random() > 0.5:
            y = self.add_noise(y, noise_level=random.uniform(0.001, 0.01))
        
        if 'pitch' in augmentations and random.random() > 0.5:
            y = self.pitch_shift(y, sr, n_steps=random.randint(-3, 3))
        
        if 'tempo' in augmentations and random.random() > 0.5:
            y = self.time_stretch(y, rate=random.uniform(0.9, 1.15))
        
        if 'reverb' in augmentations and random.random() > 0.7:
            y = self.add_reverb(y, decay=random.uniform(0.3, 0.7))
        
        if 'gain' in augmentations and random.random() > 0.5:
            y = self.random_gain(y)
        
        # Нормализация
        y = y / (np.max(np.abs(y)) + 1e-8)
        
        if output_path:
            sf.write(output_path, y, sr)
            return output_path
        
        return y, sr
    
    def generate_augmented_dataset(self, input_folder, output_folder, samples_per_track=3):
        """Генерация аугментированного датасета"""
        input_path = Path(input_folder)
        output_path = Path(output_folder)
        output_path.mkdir(parents=True, exist_ok=True)
        
        extensions = ['.mp3', '.wav', '.flac', '.ogg', '.m4a']
        count = 0
        
        for audio_file in input_path.rglob("*"):
            if audio_file.suffix.lower() not in extensions:
                continue
            
            for i in range(samples_per_track):
                out_file = output_path / f"{audio_file.stem}_aug{i}{audio_file.suffix}"
                self.augment(audio_file, out_file)
                count += 1
                print(f"Создан: {out_file.name}")
        
        print(f"Сгенерировано {count} аугментированных файлов")
        return count