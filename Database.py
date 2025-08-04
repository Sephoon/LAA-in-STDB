import pickle
import random
import numpy as np
from sklearn.preprocessing import MultiLabelBinarizer
from tqdm import tqdm


class Database:

    def __init__(self, records, mode):
        print(f"Initializing database with {len(records)} records in '{mode}' mode...")
        self.records_volume = [volume for _, _, volume in records]
        records = [[pt, doc] for pt, doc, _ in records]
        self.raw_records = records
        self.mode = mode
        self._generate_encrypted_database()
        self._generate_en_doc_occ_matrix()

    def _generate_token_mapping(self, items_list):
        encrypted_tokens = []
        plaintext_to_ciphertext = {}
        ciphertext_to_plaintext = {}
        for item in items_list:
            if item in plaintext_to_ciphertext:
                ciphertext = plaintext_to_ciphertext[item]
            else:
                ciphertext = random.randrange(1_000_000_000)
                while ciphertext in ciphertext_to_plaintext:
                    ciphertext = random.randrange(1_000_000_000)
                plaintext_to_ciphertext[item] = ciphertext
                ciphertext_to_plaintext[ciphertext] = item
            encrypted_tokens.append(ciphertext)
        return encrypted_tokens, plaintext_to_ciphertext, ciphertext_to_plaintext

    def _generate_encrypted_database(self):
        # ... (内部核心逻辑，保持不变) ...
        unique_documents = []
        new_records = []
        if self.mode == 'M2M':
            print("  - Using 'M2M' mode: documents with same content share a doc_id.")
            all_kws_tuples = [tuple(kwl) for _, kwl in self.raw_records]
            unique_doc_contents_tuples = list(dict.fromkeys(all_kws_tuples))
            unique_documents = [list(t) for t in unique_doc_contents_tuples]
            content_to_id = {content: i for i, content in enumerate(unique_doc_contents_tuples)}
            for geo, kwl in self.raw_records:
                doc_id = content_to_id[tuple(kwl)]
                new_records.append((geo, doc_id))
        elif self.mode == 'O2M':
            print("  - Using 'O2M' mode: each record has a unique doc_id.")
            for i, (geo, kwl) in enumerate(self.raw_records):
                doc_id = i
                new_records.append((geo, doc_id))
            unique_documents = [kwl for _, kwl in self.raw_records]

        print(f"  - Processed {len(self.raw_records)} raw records, generated {len(unique_documents)} unique documents.")
        self.unique_documents = unique_documents
        self.new_records = new_records

        points = [geo for geo, _ in self.new_records]
        _, pt2ct_points, ct2pt_points = self._generate_token_mapping(points)
        self.points_mapping = (pt2ct_points, ct2pt_points)
        self.pt2ct_points = pt2ct_points
        self.ct2pt_points = ct2pt_points

        unique_doc_ids = list(range(len(self.unique_documents)))
        _, doc_id_pt2ct, doc_id_ct2pt = self._generate_token_mapping(unique_doc_ids)
        self.documents_mapping = (doc_id_pt2ct, doc_id_ct2pt)

        encrypted_points = [self.pt2ct_points[geo] for geo, _ in self.new_records]
        doc_ids_for_records = [doc_id for _, doc_id in self.new_records]
        encrypted_documents = [doc_id_pt2ct[doc_id] for doc_id in doc_ids_for_records]

        self.en_records = list(zip(encrypted_points, encrypted_documents))
        print("  - Database encryption and setup complete.")

    def _generate_en_doc_occ_matrix(self):
        pt2ct_docs, _ = self.documents_mapping
        ordered_enc_doc_ids = sorted(pt2ct_docs.values())
        if not ordered_enc_doc_ids:
            self.matrix_M = np.empty((0, 0), dtype=np.int16)
            self.ordered_enc_doc_ids = []
            return

        # 反查明文 ID，取出关键词
        ct2pt_docs = {v: k for k, v in pt2ct_docs.items()}
        doc_plain_ids = [ct2pt_docs[eid] for eid in ordered_enc_doc_ids]
        corpus_keywords = [self.unique_documents[pid] for pid in doc_plain_ids]  # list(list(str))

        # ① 生成二值文档-关键词矩阵  (n_doc × n_kw)
        mlb = MultiLabelBinarizer(sparse_output=True)
        X = mlb.fit_transform(corpus_keywords).astype(np.int16)  # csr_matrix

        # ② 共现矩阵：一次乘法即可
        #    M[i, j] = ∑_k X[i,k]*X[j,k] == 两文档共同出现的关键词数
        M = (X @ X.T).toarray()  # 结果是 int16
        # 对角线目前是词汇数，无需再单独赋值
        self.matrix_M = M
        self.ordered_enc_doc_ids = ordered_enc_doc_ids
    # --- 模式1: 模拟客户端抽样查询 ---

    def simulate_client_queries(self, N0, N1, fraction, distribution='uniform'):
        print(f"\n--- Simulating {fraction} query ratio ({distribution} distribution) ---")
        # 步骤 1: 生成查询池
        query_pool = self._generate_all_possible_query_ranges(N0, N1)
        print(f"  - Generated a pool of {len(query_pool)} possible queries.")
        num_samples = int(len(query_pool) * fraction)

        # 步骤 2: 从池中抽样
        sampled_queries = self._sample_queries(query_pool, num_samples, distribution)
        print(f"  - Sampled {len(sampled_queries)} queries from the pool.")

        # 步骤 3: 获取抽样查询的响应
        actual_responses, _ = self._get_actual_responses_for_queries(sampled_queries)
        print("  - Calculated actual responses for sampled queries.")

        token_response_pairs = list(zip(sampled_queries, actual_responses))

        return actual_responses,token_response_pairs

    def _generate_all_possible_query_ranges(self, N0, N1):
        """(私有) 生成给定网格范围内的所有可能的矩形查询。"""
        responses = []
        # 为了效率，可以只在外层循环显示进度条
        for min0 in tqdm(range(1, N0 + 1), desc="Generating Query Pool", leave=False):
            for min1 in range(1, N1 + 1):
                for max0 in range(min0, N0 + 1):
                    for max1 in range(min1, N1 + 1):
                        responses.append((min0, max0, min1, max1))
        return responses

    @staticmethod
    def _sample_queries(query_pool, num_samples, distribution='uniform'):
        """(私有/静态) 根据分布从查询池中抽样。"""
        if distribution == 'uniform':
            return random.choices(query_pool, k=num_samples)

        pool_size = len(query_pool)
        x = np.linspace(0, 1, pool_size)
        weights = None

        if distribution == 'gaussian':
            from scipy.stats import norm
            mu, sigma = 0.5, 0.2
            weights = norm.pdf(x, loc=mu, scale=sigma)
        elif distribution == 'beta':
            from scipy.stats import beta
            a, b = 2, 1
            weights = beta.pdf(x, a, b)
        else:
            raise NotImplementedError(f"Distribution '{distribution}' not implemented.")

        return random.choices(query_pool, weights=weights, k=num_samples)

    def _get_actual_responses_for_queries(self, query_ranges):
        ground_truth_map = self.points_mapping[0]
        en_points = list(set(ground_truth_map.values()))
        map_points_to_coordinates= self.points_mapping[1]
        coords = np.array([map_points_to_coordinates[p] for p in en_points])
        actual = []
        unique_rs = set()

        # 对每个查询范围利用布尔索引加速判断
        for (min0, max0, min1, max1) in tqdm(query_ranges):
            mask = (coords[:, 0] >= min0) & (coords[:, 0] <= max0) & (coords[:, 1] >= min1) & (coords[:, 1] <= max1)
            indices = np.where(mask)[0]
            tokens_in_range = [en_points[i] for i in indices]
            actual.append(set(tokens_in_range))
            # 将符合查询的真实坐标加入 unique_rs 集合
            for coord in coords[indices]:
                unique_rs.add(tuple(coord))
        return actual, unique_rs

    # --- 模式2: 获取完全查询泄漏 ---

    def get_all_query_response_with_value(self, N0, N1):
        ground_truth_map = self.points_mapping[0]
        en_points = set(ground_truth_map.values())
        tokens = list(en_points)
        n = len(tokens)
        tokens_np = np.asarray(tokens)
        pt2coord = self.points_mapping[1]
        # ---------------- 1. 预处理：把点按 x、y 维度分桶 ----------------
        x2idxs = [set() for _ in range(N0 + 1)]  # x2idxs[x] = 该列所有点的下标
        y2idxs = [set() for _ in range(N1 + 1)]
        for idx, t in enumerate(tokens):
            x, y = pt2coord[t]
            if not (1 <= x <= N0 and 1 <= y <= N1):
                raise ValueError(f"坐标 {x, y} 超出给定上限")
            x2idxs[x].add(idx)
            y2idxs[y].add(idx)

        # ---------------- 2. 穷举四维区间（增量更新） ----------------
        resps = []

        for min0 in tqdm(range(1, N0 + 1), desc="min0"):
            active_x = set()  # 当前 x∈[min0, max0] 的点
            for max0 in range(min0, N0 + 1):
                active_x |= x2idxs[max0]  # 只增量加入新的一列
                if not active_x:  # 整个 x-条为空，跳过 y 的循环
                    continue

                for min1 in range(1, N1 + 1):
                    active_y = set()  # 重置 y 累加器
                    for max1 in range(min1, N1 + 1):
                        active_y |= y2idxs[max1]  # 同理，增量加入新的一行
                        # 交集得到最终落在矩形里的下标
                        idxs = active_x & active_y
                        if idxs:
                            resps.append(set(tokens_np[list(idxs)]))

        return resps

    # 其他辅助方法
    def get_known_part(self, percentage):
        # ... (保持不变) ...
        if percentage > 1:
            percentage /= 100
        num_records = len(self.new_records)
        sample_size = int(num_records * percentage)
        indices = random.sample(range(num_records), sample_size)
        sampled_records = [self.new_records[i] for i in indices]
        known_indices = [i + 1 for i in indices]
        return known_indices, sampled_records

    def generate_doc_occ_matrix(self, known_records):
        """
        只要在全局共现矩阵 M 中抽子矩阵即可，时间 O(k²)→O(1)。
        """
        # --- 第一步：拿到要保留的“全局文档编号”（明文 doc_id） ---
        known_global_ids = sorted({doc_id for _, doc_id in known_records})
        if not known_global_ids:
            return np.empty((0, 0), dtype=np.int16), []

        # --- 第二步：把明文 doc_id → 加密 doc_id → 在 matrix_M 中的行号 ---
        pt2ct_docs, _ = self.documents_mapping
        enc_ids = [pt2ct_docs[gid] for gid in known_global_ids]
        full_row_index = [self.ordered_enc_doc_ids.index(eid) for eid in enc_ids]

        # --- 第三步：一次性切片 ---
        sub_M = self.matrix_M[np.ix_(full_row_index, full_row_index)].copy()

        return sub_M, known_global_ids