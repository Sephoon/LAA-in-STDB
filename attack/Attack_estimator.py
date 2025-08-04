# Attack_estimator.py
# -*- coding: utf-8 -*-
import random
from collections import defaultdict, Counter
from functools import lru_cache

import numpy as np
from pydistinct.stats_estimators import (
    smoothed_jackknife_estimator,
    hybrid_estimator,
    sichel_estimator,
    jackknife_estimator,
    shlossers_estimator,
    chao_lee_estimator,
)
from tqdm import tqdm


# --------------------------------------------------------------------------- #
#                              1. 密度估计器                                   #
# --------------------------------------------------------------------------- #
class DensityEstimator:
    """封装密度估计逻辑，并用 lru_cache 加速。"""

    def __init__(self, token_response_pairs, estimator_name="chao_lee"):
        self.token_response_pairs = token_response_pairs
        self.estimator_name = estimator_name
        self._build_inverted_index()

    # ---------- 预处理 ----------
    def _build_inverted_index(self):
        self.point_to_tokens = defaultdict(list)
        for token, response in self.token_response_pairs:
            for point in response:
                self.point_to_tokens[point].append(token)

    # ---------- 单点 / 点对估计 ----------
    @lru_cache(maxsize=2**16)
    def estimate_rho_for_point(self, p):
        tokens = self.point_to_tokens.get(p, [])
        n, d, f = self.build_fingerprint(tokens)
        if n == 0:
            return 0
        return chao_lee_estimator(sequence=tokens)

    @lru_cache(maxsize=2**20)
    def estimate_rho_for_pair(self, p, q):
        if p == q:
            return self.estimate_rho_for_point(p)
        tokens_p = self.point_to_tokens.get(p, [])
        tokens_q = self.point_to_tokens.get(q, [])
        common_tokens = list((Counter(tokens_p) & Counter(tokens_q)).elements())
        if not common_tokens:
            return 0
        return chao_lee_estimator(sequence=common_tokens)

    # ---------- Fingerprint ----------
    @staticmethod
    def build_fingerprint(items):
        n = len(items)
        if n == 0:
            return 0, 0, {}
        freq = Counter(items)
        d = len(freq)
        f = Counter(freq.values())
        return n, d, f


# --------------------------------------------------------------------------- #
#                                2. 误差模型                                   #
# --------------------------------------------------------------------------- #
class BiasCorrector:
    """基于 KNN 的系统性误差校正模型。"""

    def __init__(self, initial_map, density_estimator, N0, N1, k=5):
        print("[BiasCorrector] Initializing ...")
        self.density_estimator = density_estimator
        self.N0 = N0
        self.N1 = N1
        self.k = k
        self.anchor_data = []

        if len(initial_map) < self.k:
            print(
                f"  - [Warning] anchor 数量不足 ({len(initial_map)} < k={self.k})，自动下调 k"
            )
            self.k = max(1, len(initial_map) - 1)

        anchor_pts = list(initial_map.keys())
        print(f"  - Learning from {len(anchor_pts)} anchors")

        for i in tqdm(range(len(anchor_pts)), desc="Learning Error Factors"):
            for j in range(i + 1, len(anchor_pts)):
                pt_a, pt_b = anchor_pts[i], anchor_pts[j]
                ct_a, ct_b = initial_map[pt_a], initial_map[pt_b]

                try:
                    rho_ab = self.calculate_query_density(pt_a, pt_b, N0, N1)
                    if rho_ab == 0:
                        continue
                    rho_hat_ab = density_estimator.estimate_rho_for_pair(ct_a, ct_b)
                    error_factor = rho_hat_ab / rho_ab
                    feature = np.array([pt_a[0], pt_a[1], pt_b[0], pt_b[1]])
                    self.anchor_data.append({"feature": feature, "factor": error_factor})
                except ValueError:
                    continue

        if not self.anchor_data:
            print("  - [Warning] 未能生成有效锚点对，误差模型将退化为恒 1.")
        else:
            print(f"  - Stored {len(self.anchor_data)} error factors.")

    # ---------- 预测 ----------
    @lru_cache(maxsize=2**18)
    def predict_correction_factor(self, pt_u, pt_r):
        if not self.anchor_data:
            return 1.0
        target = np.array([pt_u[0], pt_u[1], pt_r[0], pt_r[1]])
        distances = [
            (np.linalg.norm(target - data["feature"]), data["factor"])
            for data in self.anchor_data
        ]
        distances.sort(key=lambda x: x[0])
        neighbors = distances[: self.k]

        total_w, weighted_sum = 0.0, 0.0
        for dist, factor in neighbors:
            w = 1.0 / (dist + 1e-6)
            total_w += w
            weighted_sum += factor * w
        return weighted_sum / total_w if total_w else 1.0

    # ---------- 理论密度 ----------
    @staticmethod
    def calculate_query_density(u, r, N0, N1):
        if u[0] <= r[0] and u[1] <= r[1]:
            return u[0] * u[1] * (N0 + 1 - r[0]) * (N1 + 1 - r[1])
        if r[0] <= u[0] and r[1] <= u[1]:
            return r[0] * r[1] * (N0 + 1 - u[0]) * (N1 + 1 - u[1])
        if u[0] <= r[0] and u[1] >= r[1]:
            return u[0] * (N0 + 1 - r[0]) * r[1] * (N1 + 1 - u[1])
        if r[0] <= u[0] and r[1] >= u[1]:
            return r[0] * (N0 + 1 - u[0]) * u[1] * (N1 + 1 - r[1])
        raise ValueError("Points u and r have no dominance/anti-dominance relation")


# --------------------------------------------------------------------------- #
#                            3. 主攻击流程                                     #
# --------------------------------------------------------------------------- #
class Attack_estimator:
    """带误差校正的范围查询 + 共现矩阵联合攻击"""

    # ------------------- 初始化 ------------------- #
    def __init__(
        self,
        known_records,
        en_records,
        density_estimator,
        candidate_set_size,
        dominance_confidence_threshold,
        N0,
        N1,
        max_tool_points,
        error_model_k,
        mode,
        M,
        M_prime,
        known_volume,
        encrypted_volume,
        ordered_enc_doc_ids,
        known_global_ids,
        alpha,
        num_point_anchors
    ):
        self.known_records = known_records
        self.en_records = en_records
        self.en_points = {c for c, _ in en_records}

        self.density_estimator = density_estimator
        self.candidate_set_size = candidate_set_size
        self.dominance_confidence_threshold = dominance_confidence_threshold

        self.N0 = N0
        self.N1 = N1
        self.max_tool_points = max_tool_points
        self.error_model_k = error_model_k
        self.mode = mode

        self.M = M
        self.M_prime = M_prime
        self.known_volume = known_volume
        self.encrypted_volume = encrypted_volume
        self.ordered_enc_doc_ids = ordered_enc_doc_ids
        self.known_global_ids = known_global_ids
        self.alpha = alpha
        self.num_point_anchors = num_point_anchors
        self.initial_doc_map = {}
        self.points_map = {}
        self.doc_map = {}

    # ------------------- 对外入口 ------------------- #
    def attack(self):
        print("==========  Attack_estimator start  ==========")

        initial_points_map,initial_docs_map = self.find_initial_mapping(
            self.known_records,
            self.en_records,
            self.mode,
            self.M,
            self.M_prime,
            self.ordered_enc_doc_ids,
            self.known_global_ids,
            self.known_volume,
            self.encrypted_volume
        )
        # dict.update() 返回 None，必须显式合并
        self.initial_points_map = {
            **initial_points_map
        }
        # print(f"Initial anchor points: {len(self.initial_points_map)}\n")

        self.initial_doc_map = {
            **initial_docs_map
        }
        # print(f"Initial anchor docs: {len(self.initial_doc_map)}\n")


        anchors_items = list(initial_points_map.items())
        k = min(self.num_point_anchors, len(anchors_items))
        anchors_items = random.sample(anchors_items, k)
        initial_points_map = dict(anchors_items)

        print(f"[Init] total points anchor points = {len(initial_points_map)}")


        print(f"[Init] total documents anchor points = {len(initial_points_map)}")

        # 2. 范围查询攻击（坐标）
        point_map = self._range_attack(initial_points_map)
        self.points_map = point_map

        # 3. 共现矩阵攻击（文档）
        doc_map = self._keywords_attack()
        self.doc_map = doc_map

        print("==========  Attack_estimator finished  ==========")
        return point_map, self.doc_map

    # ------------------- Phase-A : 坐标恢复 ------------------- #
    def _range_attack(self, initial_map):
        error_model = BiasCorrector(
            initial_map, self.density_estimator, self.N0, self.N1, k=self.error_model_k
        )
        solved_map = initial_map.copy()
        all_known_pts = {c for c, _ in self.known_records}

        # ---------- Phase-1：生成候选 ----------
        print("\n[Phase-1] Building candidate sets ...")
        unsolved_initial = all_known_pts - set(solved_map.keys())
        unmatched_cts_initial = self.en_points - set(solved_map.values())
        candidate_dict = {}

        for pt_u in tqdm(unsolved_initial, desc="Candidates"):
            ct_scores = []
            for ct_u in unmatched_cts_initial:
                tot_err, cnt = 0.0, 0
                for pt_r in initial_map:
                    try:
                        rho = self.calculate_query_density(pt_u, pt_r, self.N0, self.N1)
                        if rho == 0:
                            continue
                        factor = error_model.predict_correction_factor(pt_u, pt_r)
                        rho_corr = rho * factor
                        rho_hat = self.density_estimator.estimate_rho_for_pair(
                            ct_u, initial_map[pt_r]
                        )
                        tot_err += abs(rho_hat - rho_corr) / rho_corr
                        cnt += 1
                    except (ValueError, ZeroDivisionError):
                        continue
                if cnt:
                    ct_scores.append((ct_u, tot_err / cnt))
            ct_scores.sort(key=lambda x: x[1])
            candidate_dict[pt_u] = [c for c, _ in ct_scores[: self.candidate_set_size]]
            # if len(candidate_dict[pt_u]) > 0:
            #     solved_map[pt_u] = candidate_dict[pt_u][0]
        print("  - Candidate sets ready.")

        # ---------- Phase-2：迭代解析 ----------
        print("\n[Phase-2] Iterative solving ...")
        iteration = 1
        while True:
            unsolved = all_known_pts - set(solved_map.keys())
            if not unsolved:
                break
            print(f"\n--- Iteration {iteration} --- (solved={len(solved_map)})")

            potential = {}
            for pt_u in tqdm(unsolved, desc=f"Iter {iteration}", leave=False):
                tools = []
                for pt_r in solved_map:
                    try:
                        rho = self.calculate_query_density(pt_u, pt_r, self.N0, self.N1)
                        if rho:
                            tools.append((pt_r, rho))
                    except ValueError:
                        continue
                if not tools:
                    continue
                tools.sort(key=lambda x: x[1], reverse=True)
                tools = [t[0] for t in tools[: self.max_tool_points]]

                candidates = [
                    c
                    for c in candidate_dict.get(pt_u, [])
                    if c not in solved_map.values()
                ]
                cand_scores = {}
                for cand in candidates:
                    tot_err, cnt = 0.0, 0
                    for pt_r in tools:
                        try:
                            rho = self.calculate_query_density(
                                pt_u, pt_r, self.N0, self.N1
                            )
                            factor = error_model.predict_correction_factor(pt_u, pt_r)
                            rho_corr = rho * factor
                            rho_hat = self.density_estimator.estimate_rho_for_pair(
                                cand, solved_map[pt_r]
                            )
                            tot_err += abs(rho_hat - rho_corr) / rho_corr
                            cnt += 1
                        except (ValueError, ZeroDivisionError):
                            continue
                    if cnt:
                        cand_scores[cand] = tot_err / cnt
                if cand_scores:
                    best_cand, min_err = min(cand_scores.items(), key=lambda x: x[1])
                    potential[pt_u] = (best_cand, min_err)

            if not potential:
                print("  - No potential solutions, stopping.")
                break

            # 选择满足阈值且互不冲突的解
            chosen = {}
            used_ct = set(solved_map.values())
            for pt, (cand, err) in sorted(potential.items(), key=lambda x: x[1][1]):
                if (
                    err < self.dominance_confidence_threshold
                    and cand not in used_ct
                ):
                    chosen[pt] = cand
                    used_ct.add(cand)

            if not chosen:
                print("  - No candidates pass confidence threshold, stopping.")
                break

            print(f"  - newly solved {len(chosen)} points.")
            solved_map.update(chosen)
            iteration += 1

        print(f"[Range Attack] solved total = {len(solved_map)}")
        return solved_map

    # ------------------- Phase-B : 文档映射 ------------------- #
    def _keywords_attack(self):

        print("\n>>> Phase-B : recovering document IDs with volume feature ...")

        # ---------- 预处理 ----------
        pt2ct_points = self.points_map
        enc_id_to_matrix_idx = {eid: i for i, eid in enumerate(self.ordered_enc_doc_ids)}
        global_id_to_local_idx = {gid: i for i, gid in enumerate(self.known_global_ids)}

        # 体积分布签名
        known_doc2vols = defaultdict(list)
        for (coord, gid), vol in zip(self.known_records, self.known_volume):
            known_doc2vols[gid].append(vol)

        enc_doc2vols = defaultdict(list)
        for (coord, eid), vol in zip(self.en_records, self.encrypted_volume):
            enc_doc2vols[eid].append(vol)

        def freeze(counter):
            # 把 Counter 转成可哈希对象
            return tuple(sorted(counter.items()))

        known_sig = {gid: freeze(Counter(vs)) for gid, vs in known_doc2vols.items()}
        enc_sig = {eid: freeze(Counter(vs)) for eid, vs in enc_doc2vols.items()}

        # ---------- 动态分组（与原代码相同） ----------
        coord_to_known_ids = defaultdict(set)
        for coord, doc_id in self.known_records:
            coord_to_known_ids[coord].add(doc_id)

        enc_coord_to_enc_docs = defaultdict(set)
        for enc_coord, enc_doc_id in self.en_records:
            enc_coord_to_enc_docs[enc_coord].add(enc_doc_id)

        ordered_known_groups, ordered_encrypted_groups = [], []
        for coord, known_ids_set in tqdm(coord_to_known_ids.items(), desc="Grouping coords"):
            enc_coord = pt2ct_points.get(coord)
            if enc_coord is None:
                continue
            enc_ids_set = enc_coord_to_enc_docs.get(enc_coord)
            if enc_ids_set is None:
                continue
            ordered_known_groups.append(list(known_ids_set))
            ordered_encrypted_groups.append(list(enc_ids_set))

        # ================================================================= #
        # Phase-1 : 快速映射 —— 对角线体积唯一 + volume 签名唯一
        # ================================================================= #
        print("Phase-1: fast mapping by diag-volume && volume-signature ...")
        recovered_mapping = {}

        # 1) 先用 volume-signature 唯一值直接匹配
        sig_counter_kn = Counter(known_sig.values())
        sig_counter_en = Counter(enc_sig.values())
        unique_sigs = {s for s in sig_counter_kn if sig_counter_kn[s] == 1 and sig_counter_en[s] == 1}
        for sig in unique_sigs:
            kn_gid = next(g for g, s in known_sig.items() if s == sig)
            en_eid = next(e for e, s in enc_sig.items() if s == sig)
            recovered_mapping[kn_gid] = en_eid

        # 2) 再利用矩阵对角线元素唯一性
        for known_ids, encrypted_ids in zip(ordered_known_groups, ordered_encrypted_groups):
            # 取对角线元素
            try:
                known_feats = [self.M_prime[global_id_to_local_idx[gid], global_id_to_local_idx[gid]]
                               for gid in known_ids]
                enc_feats = [self.M[enc_id_to_matrix_idx[eid], enc_id_to_matrix_idx[eid]]
                             for eid in encrypted_ids]
            except KeyError:
                continue

            if len(set(known_feats)) != len(known_feats) or len(set(enc_feats)) != len(enc_feats):
                continue

            k_feat_map = {f: gid for f, gid in zip(known_feats, known_ids)}
            e_feat_map = {f: eid for f, eid in zip(enc_feats, encrypted_ids)}
            for f in set(k_feat_map) & set(e_feat_map):
                # 再次用 volume-signature 佐证
                if known_sig[k_feat_map[f]] == enc_sig[e_feat_map[f]]:
                    recovered_mapping[k_feat_map[f]] = e_feat_map[f]

        # 加入初始 doc-map（如果有）
        recovered_mapping.update(self.initial_doc_map)
        used_encrypted = set(recovered_mapping.values())
        print(f"Phase-1 done, initial mappings: {len(recovered_mapping)}")

        # ================================================================= #
        # Phase-2 : 迭代细化 —— 加入 volume-signature 完全一致约束
        # ================================================================= #
        print("Phase-2: iterative refinement with volume signature ...")
        iteration = 0
        while True:
            iteration += 1
            new_found = 0

            reference_items = list(recovered_mapping.items())
            if not reference_items:
                print("  * no reference set, abort.")
                break
            sample_size = min(len(reference_items), int(self.alpha))
            if sample_size:
                reference_items = random.sample(reference_items, sample_size)

            for known_ids, encrypted_ids in zip(ordered_known_groups, ordered_encrypted_groups):
                unmapped_known = [gid for gid in known_ids if gid not in recovered_mapping]
                for kn_gid in unmapped_known:
                    kn_idx = global_id_to_local_idx.get(kn_gid)
                    if kn_idx is None:
                        continue

                    valid = []
                    for en_id in encrypted_ids:
                        if en_id in used_encrypted:
                            continue
                        en_idx = enc_id_to_matrix_idx.get(en_id)
                        if en_idx is None:
                            continue

                        # 1) 对角线体积必须相同
                        if self.M_prime[kn_idx, kn_idx] != self.M[en_idx, en_idx]:
                            continue
                        # 2) volume-signature 必须完全一致
                        if known_sig.get(kn_gid) != enc_sig.get(en_id):
                            continue
                        # 3) 保持与参考集的矩阵一致性
                        if all(self.M_prime[kn_idx, global_id_to_local_idx[rk]] ==
                               self.M[en_idx, enc_id_to_matrix_idx[re]]
                               for rk, re in reference_items):
                            valid.append(en_id)

                    if len(valid) == 1:
                        chosen = valid[0]
                        recovered_mapping[kn_gid] = chosen
                        used_encrypted.add(chosen)
                        new_found += 1

            print(f"  iteration {iteration}: +{new_found}")
            if new_found == 0 or iteration > 50:
                break

        print(f"Phase-B complete, total recovered docs: {len(recovered_mapping)}")
        return recovered_mapping

    @staticmethod
    def find_initial_mapping(
            known_records,
            en_records,
            mode,
            # --- 策略1：基于矩阵的参数 ---
            M,
            M_prime,
            ordered_enc_doc_ids,
            known_global_ids,
            # --- 策略2：基于预计算体积的参数 ---
            known_volumes_list,  # 注意：原函数中的 known_volumes 是一个列表
            encrypted_volumes_list,  # 原函数中的 records_volume
    ):
        final_doc_map = {}

        # ---------- 策略 1: 基于矩阵对角线元素进行映射 ----------
        # 仅当所有矩阵相关参数都提供时才执行
        if M is not None and M_prime is not None and ordered_enc_doc_ids and known_global_ids:
            print("\nAttempting matrix-based mapping...")
            if M.size == 0 or M_prime.size == 0:
                print("  * M or M' is empty, skipping matrix-based mapping.")
            else:
                # 从矩阵对角线提取体积信息
                global_id_to_local_idx = {gid: i for i, gid in enumerate(known_global_ids)}
                enc_id_to_matrix_idx = {eid: i for i, eid in enumerate(ordered_enc_doc_ids)}

                known_vols_from_matrix = {
                    gid: M_prime[global_id_to_local_idx[gid], global_id_to_local_idx[gid]]
                    for gid in known_global_ids
                }
                encrypted_vols_from_matrix = {
                    eid: M[enc_id_to_matrix_idx[eid], enc_id_to_matrix_idx[eid]]
                    for eid in ordered_enc_doc_ids
                }

                # 寻找唯一值匹配
                unique_known = {v for v, c in Counter(known_vols_from_matrix.values()).items() if c == 1}
                unique_enc = {v for v, c in Counter(encrypted_vols_from_matrix.values()).items() if c == 1}
                recoverable = unique_known & unique_enc

                if recoverable:
                    vol2kn = {v: gid for gid, v in known_vols_from_matrix.items()}
                    vol2en = {v: eid for eid, v in encrypted_vols_from_matrix.items()}
                    doc_map_from_matrix = {vol2kn[v]: vol2en[v] for v in recoverable}

                    print(f"  * Found {len(doc_map_from_matrix)} document mapping(s) from matrix diagonals.")
                    final_doc_map.update(doc_map_from_matrix)
                else:
                    print("  * No unique recoverable values found in matrix diagonals.")

        # ---------- 策略 2: 基于外部传入的体积列表进行映射 ----------
        # 仅当所有体积列表相关参数都提供时才执行
        if known_volumes_list and encrypted_volumes_list:
            print(f"\nAttempting volume-list-based mapping...")

            # 将体积列表与ID关联起来
            # 注意：这里我们假设一个文档ID可以有多个记录，但其体积特征是与记录本身绑定的。
            # 为了进行文档级别的匹配，我们选择每个文档ID对应的第一个体积值作为代表。
            # 如果一个文档的所有记录体积都相同，这没问题。如果不同，这是一个简化处理。

            known_gid_to_vol = {gid: vol for (coord, gid), vol in zip(known_records, known_volumes_list)}
            enc_eid_to_vol = {eid: vol for (coord, eid), vol in zip(en_records, encrypted_volumes_list)}

            unique_kn = {v for v, c in Counter(known_gid_to_vol.values()).items() if c == 1}
            unique_en = {v for v, c in Counter(enc_eid_to_vol.values()).items() if c == 1}
            recoverable = unique_kn & unique_en

            if recoverable:
                vol2kn_gid = {v: gid for gid, v in known_gid_to_vol.items() if v in recoverable}
                vol2en_eid = {v: eid for eid, v in enc_eid_to_vol.items() if v in recoverable}

                doc_map_from_volume = {vol2kn_gid[v]: vol2en_eid[v] for v in recoverable}

                # 在合并前打印本次找到的数量
                new_mappings_count = len(set(doc_map_from_volume.keys()) - set(final_doc_map.keys()))
                print(
                    f"  * Found {len(doc_map_from_volume)} document mapping(s) from volume list ({new_mappings_count} are new).")
                final_doc_map.update(doc_map_from_volume)
            else:
                print("  * No unique recoverable values found in volume lists.")

        # ---------- 最终处理: 基于合并后的 doc_map 生成 point_map ----------
        if not final_doc_map:
            print("\nNo document mappings found from any source. Cannot create point map.")
            return {}, {}

        print(f"\nTotal unique document mappings found: {len(final_doc_map)}")

        # 1. 为所有相关文档建立出现位置的索引
        kn_doc2coords = defaultdict(list)
        for coord, gid in known_records:
            if gid in final_doc_map:
                kn_doc2coords[gid].append(coord)

        en_doc2coords = defaultdict(list)
        for coord, eid in en_records:
            # 注意这里要用 final_doc_map.values() 来检查
            if eid in final_doc_map.values():
                en_doc2coords[eid].append(coord)

        # 2. 根据模式筛选一对一的坐标，生成 point_map
        point_map = {}
        for kn_gid, en_id in final_doc_map.items():
            kn_locs = kn_doc2coords.get(kn_gid, [])
            en_locs = en_doc2coords.get(en_id, [])

            if mode == "O2M":  # One-to-Many: 只要双方都至少出现一次，就取第一个作为锚点
                if kn_locs and en_locs:
                    point_map[kn_locs[0]] = en_locs[0]
            else:  # M2M (Many-to-Many): 只有当双方都只出现一次时，才认为是可靠的锚点
                if len(kn_locs) == 1 and len(en_locs) == 1:
                    point_map[kn_locs[0]] = en_locs[0]

        print(f"  * Recovered {len(point_map)} anchor point(s) and {len(final_doc_map)} document mapping(s) in total.")

        return point_map, final_doc_map


    @staticmethod
    def calculate_query_density(u, r, N0, N1):
        if u[0] <= r[0] and u[1] <= r[1]:
            return u[0] * u[1] * (N0 + 1 - r[0]) * (N1 + 1 - r[1])
        if r[0] <= u[0] and r[1] <= u[1]:
            return r[0] * r[1] * (N0 + 1 - u[0]) * (N1 + 1 - u[1])
        if u[0] <= r[0] and u[1] >= r[1]:
            return u[0] * (N0 + 1 - r[0]) * r[1] * (N1 + 1 - u[1])
        if r[0] <= u[0] and r[1] >= u[1]:
            return r[0] * (N0 + 1 - u[0]) * u[1] * (N1 + 1 - r[1])
        raise ValueError("Points u and r have no dominance/anti-dominance relation")