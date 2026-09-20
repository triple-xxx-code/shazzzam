import numpy as np
import faiss
import joblib
from pathlib import Path

class VectorIndex:
    """Векторный поиск на базе FAISS"""
    
    def __init__(self, db_path="database", dim=128):
        self.db_path = Path(db_path)
        self.db_path.mkdir(exist_ok=True)
        self.dim = dim
        self.index = None
        self.track_ids = []
        self.metadata = {}
        
        # Инициализация индекса (IVF для больших баз)
        self._init_index()
    
    def _init_index(self):
        """Инициализация FAISS индекса"""
        # Используем IVF + PQ для больших баз (>100k векторов)
        # Для маленьких баз — обычный IndexFlatL2
        self.index = faiss.IndexFlatL2(self.dim)
        print(f"FAISS индекс инициализирован (dim={self.dim})")
    
    def add_vectors(self, track_ids, vectors, metadata=None):
        """
        Добавление векторов в индекс
        track_ids: список ID треков
        vectors: numpy array (N, dim)
        """
        vectors = np.array(vectors, dtype=np.float32)
        
        if vectors.shape[1] != self.dim:
            raise ValueError(f"Размерность {vectors.shape[1]} != {self.dim}")
        
        # Нормализация для cosine similarity через L2
        faiss.normalize_L2(vectors)
        
        self.index.add(vectors)
        self.track_ids.extend(track_ids)
        
        if metadata:
            self.metadata.update(metadata)
        
        print(f"Добавлено {len(track_ids)} векторов. Всего: {self.index.ntotal}")
    
    def search(self, query_vector, top_k=10):
        """Поиск ближайших векторов"""
        query = np.array([query_vector], dtype=np.float32)
        faiss.normalize_L2(query)
        
        distances, indices = self.index.search(query, top_k)
        
        results = []
        for dist, idx in zip(distances[0], indices[0]):
            if idx < len(self.track_ids):
                track_id = self.track_ids[idx]
                # Преобразуем L2 distance в similarity (0..1)
                similarity = 1 - (dist / 2)  # для нормализованных векторов
                results.append({
                    'track_id': track_id,
                    'similarity': float(similarity),
                    'distance': float(dist),
                    'path': self.metadata.get(track_id, {}).get('path', '')
                })
        
        return results
    
    def save(self):
        """Сохранение индекса"""
        faiss.write_index(self.index, str(self.db_path / "faiss.index"))
        joblib.dump(self.track_ids, self.db_path / "track_ids.pkl")
        joblib.dump(self.metadata, self.db_path / "vector_metadata.pkl")
        print("FAISS индекс сохранён")
    
    def load(self):
        """Загрузка индекса"""
        index_path = self.db_path / "faiss.index"
        ids_path = self.db_path / "track_ids.pkl"
        
        if index_path.exists() and ids_path.exists():
            self.index = faiss.read_index(str(index_path))
            self.track_ids = joblib.load(ids_path)
            meta_path = self.db_path / "vector_metadata.pkl"
            if meta_path.exists():
                self.metadata = joblib.load(meta_path)
            print(f"FAISS индекс загружен: {self.index.ntotal} векторов")
        else:
            raise FileNotFoundError("FAISS индекс не найден. Запустите index.py")
    
    def remove(self, track_ids_to_remove):
        """Удаление векторов по ID (для горячего обновления)"""
        # FAISS не поддерживает прямое удаление, перестраиваем индекс
        keep_mask = [tid not in track_ids_to_remove for tid in self.track_ids]
        
        # Получаем все векторы
        all_vectors = faiss.vector_to_array(self.index.reconstruct_n(0, self.index.ntotal))
        all_vectors = all_vectors.reshape(self.index.ntotal, self.dim)
        
        # Фильтруем
        kept_vectors = all_vectors[keep_mask]
        kept_ids = [tid for tid, keep in zip(self.track_ids, keep_mask) if keep]
        
        # Перестраиваем
        self.index = faiss.IndexFlatL2(self.dim)
        if len(kept_vectors) > 0:
            self.index.add(kept_vectors)
        
        self.track_ids = kept_ids
        
        # Удаляем из метаданных
        for tid in track_ids_to_remove:
            self.metadata.pop(tid, None)
        
        print(f"Удалено {len(track_ids_to_remove)} векторов. Осталось: {self.index.ntotal}")