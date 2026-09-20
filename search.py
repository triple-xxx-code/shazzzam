import librosa
import numpy as np
import joblib
from pathlib import Path
from sklearn.metrics.pairwise import cosine_similarity
from cnn_model import CNNModel
from vector_index import VectorIndex

class AudioSearcher:
    def __init__(self, db_path="database", use_vector=True):
        self.db_path = Path(db_path)
        self.fingerprints = {}
        self.metadata = {}
        self.use_vector = use_vector
        
        self.load_database()
        
        # CNN модель
        self.cnn = CNNModel(db_path=db_path)
        try:
            self.cnn.load("cnn_contrastive.keras")
            self.use_cnn = True
        except:
            self.use_cnn = False
        
        # FAISS индекс
        if use_vector:
            self.vector_index = VectorIndex(db_path=db_path, dim=128)
            try:
                self.vector_index.load()
            except:
                self.use_vector = False
    
    def load_database(self):
        """Загрузка базы"""
        fp_path = self.db_path / "fingerprints.pkl"
        meta_path = self.db_path / "metadata.pkl"
        
        if fp_path.exists() and meta_path.exists():
            self.fingerprints = joblib.load(fp_path)
            self.metadata = joblib.load(meta_path)
            print(f"Загружено {len(self.fingerprints)} треков")
        else:
            raise Exception("База не найдена! Запустите index.py")
    
    def extract_sample_fingerprint(self, sample_path):
        """Извлечение признаков из семпла"""
        if self.use_cnn:
            embedding = self.cnn.get_embedding(sample_path)
            return {'embedding': embedding, 'method': 'cnn'}
        else:
            y, sr = librosa.load(sample_path, sr=None)
            chroma = librosa.feature.chroma_cqt(y=y, sr=sr)
            mfcc = librosa.feature.mfcc(y=y, sr=sr, n_mfcc=20)
            tempo, _ = librosa.beat.beat_track(y=y, sr=sr)
            
            return {
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
    
    def compute_similarity(self, sample_fp, track_fp):
        """Вычисление сходства"""
        # CNN режим
        if sample_fp.get('method') == 'cnn' and track_fp.get('method') == 'cnn':
            sample_vec = sample_fp['embedding'].reshape(1, -1)
            track_vec = track_fp['embedding'].reshape(1, -1)
            return float(cosine_similarity(sample_vec, track_vec)[0][0])
        
        # Librosa fallback
        weights = {'chroma': 0.35, 'mfcc': 0.30, 'tempo': 0.15, 'spectral': 0.20}
        
        chroma_sim = cosine_similarity(
            np.array([sample_fp['chroma_mean']]),
            np.array([track_fp['chroma_mean']])
        )[0][0]
        
        mfcc_sim = cosine_similarity(
            np.array([sample_fp['mfcc_mean']]),
            np.array([track_fp['mfcc_mean']])
        )[0][0]
        
        tempo_diff = abs(sample_fp['tempo'] - track_fp['tempo'])
        tempo_sim = 1 / (1 + tempo_diff / 20)
        
        spectral_sim = cosine_similarity(
            np.array([[sample_fp['spectral_centroid'], sample_fp['zero_crossing_rate']]]),
            np.array([[track_fp['spectral_centroid'], track_fp['zero_crossing_rate']]])
        )[0][0]
        
        return (weights['chroma'] * chroma_sim +
                weights['mfcc'] * mfcc_sim +
                weights['tempo'] * tempo_sim +
                weights['spectral'] * spectral_sim)
    
    def search(self, sample_path, top_k=10, min_similarity=0.5):
        """Поиск наиболее похожих треков"""
        print(f"Поиск по семплу: {sample_path}")
        sample_fp = self.extract_sample_fingerprint(sample_path)
        
        # Быстрый поиск через FAISS
        if self.use_vector and sample_fp.get('method') == 'cnn':
            print("Используется FAISS (быстрый поиск)")
            results = self.vector_index.search(sample_fp['embedding'], top_k=top_k)
            return [r for r in results if r['similarity'] >= min_similarity]
        
        # Медленный поиск (brute force)
        print("Используется brute-force поиск")
        results = []
        for track_id, track_fp in self.fingerprints.items():
            sim = self.compute_similarity(sample_fp, track_fp)
            if sim >= min_similarity:
                results.append({
                    'track_id': track_id,
                    'similarity': float(sim),
                    'path': self.metadata[track_id]['path']
                })
        
        results.sort(key=lambda x: x['similarity'], reverse=True)
        return results[:top_k]

if __name__ == "__main__":
    searcher = AudioSearcher()
    results = searcher.search("samples/my_sample.wav", top_k=5)
    
    print("\nТоп совпадений:")
    for i, result in enumerate(results, 1):
        print(f"{i}. {result['track_id']} (сходство: {result['similarity']:.2f})")
        print(f"   Путь: {result['path']}")