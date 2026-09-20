import librosa
import numpy as np
import os
import joblib
from pathlib import Path
from datetime import datetime
from cnn_model import CNNModel
from vector_index import VectorIndex

class AudioIndexer:
    def __init__(self, db_path="database", use_cnn=True):
        self.db_path = Path(db_path)
        self.db_path.mkdir(exist_ok=True)
        self.fingerprints = {}
        self.metadata = {}
        self.use_cnn = use_cnn
        
        if use_cnn:
            self.cnn = CNNModel(db_path=db_path)
            try:
                self.cnn.load("cnn_contrastive.keras")
                print("CNN модель загружена")
            except FileNotFoundError:
                print("CNN модель не найдена, используется fallback на librosa")
                self.use_cnn = False
        
        self.vector_index = VectorIndex(db_path=db_path, dim=128)
    
    def extract_fingerprint(self, audio_path, hop_length=512):
        """Создание отпечатка трека"""
        if self.use_cnn:
            # CNN эмбеддинг (128-dim)
            embedding = self.cnn.get_embedding(audio_path)
            return {'embedding': embedding, 'method': 'cnn'}
        else:
            # Fallback на librosa
            y, sr = librosa.load(audio_path, sr=None)
            D = np.abs(librosa.stft(y, hop_length=hop_length))
            chroma = librosa.feature.chroma_cqt(y=y, sr=sr)
            mfcc = librosa.feature.mfcc(y=y, sr=sr, n_mfcc=20)
            tempo, _ = librosa.beat.beat_track(y=y, sr=sr)
            
            fingerprint = {
                'chroma_mean': np.mean(chroma, axis=1),
                'chroma_std': np.std(chroma, axis=1),
                'mfcc_mean': np.mean(mfcc, axis=1),
                'mfcc_std': np.std(mfcc, axis=1),
                'tempo': tempo,
                'duration': len(y) / sr,
                'spectral_centroid': np.mean(librosa.feature.spectral_centroid(y=y, sr=sr)),
                'zero_crossing_rate': np.mean(librosa.feature.zero_crossing_rate(y)),
                'method': 'librosa'
            }
            return fingerprint
    
    def index_library(self, music_folder="music_library", extensions=None):
        """Индексация библиотеки"""
        if extensions is None:
            extensions = ['.mp3', '.wav', '.flac', '.ogg', '.m4a']
        
        music_path = Path(music_folder)
        indexed_count = 0
        embeddings = []
        track_ids = []
        
        print(f"Начало индексации: {datetime.now()}")
        print(f"Режим: {'CNN' if self.use_cnn else 'librosa'}")
        
        for audio_file in music_path.rglob("*"):
            if audio_file.suffix.lower() not in extensions:
                continue
            
            try:
                print(f"Обработка: {audio_file.name}")
                fingerprint = self.extract_fingerprint(audio_file)
                
                track_id = f"{audio_file.parent.name}/{audio_file.name}"
                self.fingerprints[track_id] = fingerprint
                self.metadata[track_id] = {
                    'path': str(audio_file.absolute()),
                    'size': audio_file.stat().st_size,
                    'indexed_at': datetime.now().isoformat()
                }
                
                # Для FAISS
                if self.use_cnn:
                    embeddings.append(fingerprint['embedding'])
                    track_ids.append(track_id)
                
                indexed_count += 1
                
            except Exception as e:
                print(f"Ошибка {audio_file.name}: {e}")
        
        # Сохранение базы
        self.save_database()
        
        # Построение FAISS индекса
        if self.use_cnn and embeddings:
            self.vector_index.add_vectors(
                track_ids, embeddings,
                {tid: {'path': self.metadata[tid]['path']} for tid in track_ids}
            )
            self.vector_index.save()
        
        print(f"Проиндексировано {indexed_count} треков")
    
    def save_database(self):
        """Сохранение базы"""
        joblib.dump(self.fingerprints, self.db_path / "fingerprints.pkl")
        joblib.dump(self.metadata, self.db_path / "metadata.pkl")
        print("База сохранена")
    
    def load_database(self):
        """Загрузка базы"""
        fp_path = self.db_path / "fingerprints.pkl"
        meta_path = self.db_path / "metadata.pkl"
        
        if fp_path.exists() and meta_path.exists():
            self.fingerprints = joblib.load(fp_path)
            self.metadata = joblib.load(meta_path)
            print(f"Загружено {len(self.fingerprints)} треков")
        else:
            print("База не найдена")

if __name__ == "__main__":
    indexer = AudioIndexer(use_cnn=True)
    indexer.index_library()