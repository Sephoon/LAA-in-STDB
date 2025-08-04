import random
from collections import defaultdict, Counter
from functools import lru_cache
from tqdm import tqdm


class Attack:
    def __init__(
        self,
        known_records,
        en_records,
        mode,
        RM,
        N0,
        N1,
        M,
        M_prime,
        known_volume,
        encrypted_volume,
        ordered_enc_doc_ids,
        known_global_ids,
        alpha,
    ):
        self.known_records = known_records
        self.en_records = en_records
        self.mode = mode

        self.RM = RM
        self.N0 = N0
        self.N1 = N1

        self.M = M
        self.M_prime = M_prime

        self.known_volume = known_volume
        self.encrypted_volume = encrypted_volume

        self.ordered_enc_doc_ids = ordered_enc_doc_ids
        self.known_global_ids = known_global_ids
        self.alpha = alpha

        # 运行过程中逐渐填充
        self.points_map = {}   # 明文坐标 -> 加密坐标
        self.doc_map    = {}   # 明文 doc_id -> 加密 doc_id
        self.initial_points_map = {}  # 初始锚点
        self.initial_doc_map = {}

    # --------------------------------------------------------------------- #
    # 主入口
    # --------------------------------------------------------------------- #
    def attack(self):

        print("==========  Range-Query Attack Start  ==========")

        # # ---------- 1. 利用两种策略生成初始锚点 ----------
        # initial_points_from_matrix,initial_doc_from_matrix= self.find_initial_point_mapping_by_matrix(
        #     self.known_records,
        #     self.en_records,
        #     self.M,
        #     self.M_prime,
        #     self.ordered_enc_doc_ids,
        #     self.known_global_ids,
        #     self.mode,
        # )
        #
        # initial_points_map_from_volume,initial_doc_map_from_volume = self.find_initial_mapping_by_volume(
        #     self.known_records,
        #     self.en_records,
        #     self.known_volume,
        #     self.encrypted_volume,
        #     self.mode,
        # )
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
        print(f"Initial anchor points: {len(self.initial_points_map)}\n")

        self.initial_doc_map = {
            **initial_docs_map
        }
        print(f"Initial anchor docs: {len(self.initial_doc_map)}\n")

        # ---------- 2. Phase-A：坐标映射 ----------
        point_map = self._range_attack()
        self.points_map = point_map

        # ---------- 3. Phase-B：文档映射 ----------
        doc_map = self._keywords_attack()
        self.doc_map = doc_map

        print("==========  Attack Finished  ==========")
        return self.points_map, self.doc_map

    # --------------------------------------------------------------------- #
    #  Phase-A：范围查询攻击（坐标）
    # --------------------------------------------------------------------- #
    def _range_attack(self):
        print(">>> Phase-A : recovering coordinates ...")

        # 3.1 倒排索引 / 单点密度
        print("Step 3.1: Building inverted index ...")
        point2resp, single_rho = self.build_inverted_index(self.RM)
        rho_pair = self.make_pair_density_fn(point2resp)

        # 3.2 已解 / 未解集合
        print("Step 3.2: Injecting anchor points ...")
        solved_en_point = self.initial_points_map.copy()             # 明→密
        all_known_points = {c for c, _ in self.known_records}
        points_to_solve = all_known_points - set(solved_en_point.keys())

        all_en_points = {c for c, _ in self.en_records}
        candidate_en_points = all_en_points - set(solved_en_point.values())
        candidate_en_points_rho = {p: single_rho.get(p, 0) for p in candidate_en_points}

        # 3.3 为每个未解点找单点密度候选
        print("Step 3.3: Generating density-based candidates ...")
        unsolved_candidates = defaultdict(list)
        for pt_u in points_to_solve:
            x, y = pt_u
            self_rho_theory = x * y * (self.N0 + 1 - x) * (self.N1 + 1 - y)
            for ep, rho_actual in candidate_en_points_rho.items():
                if rho_actual == self_rho_theory:
                    unsolved_candidates[pt_u].append(ep)

        # 3.4 唯一候选直接确定
        print("Step 3.4: Fixing points with unique candidate ...")
        unsolved = []
        for pt_u, cand in unsolved_candidates.items():
            if len(cand) == 1 and cand[0] not in solved_en_point.values():
                solved_en_point[pt_u] = cand[0]
            else:
                unsolved.append(pt_u)
        print(f"  - solved so far: {len(solved_en_point)} / {len(all_known_points)}")

        # 3.5 构造工具点
        print("Step 3.5: Constructing tool points ...")
        tool_points = defaultdict(lambda: [[], []])   # {u : [domain, anti_domain]}
        current_solved_pts = list(solved_en_point.keys())
        for u in unsolved:
            ux, uy = u
            for s in current_solved_pts:
                sx, sy = s
                if (ux < sx and uy < sy) or (ux > sx and uy > sy):
                    tool_points[u][0].append(s)
                elif (ux < sx and uy > sy) or (ux > sx and uy < sy):
                    tool_points[u][1].append(s)

        # 3.6 迭代利用配对密度求解
        print("Step 3.6: Iteratively solving remaining points ...")
        for u in tqdm(unsolved, desc="Attacking"):
            domain, anti_domain = tool_points[u]
            if not domain or not anti_domain:
                continue

            d0 = domain[0]
            a0 = anti_domain[0]
            target_rho_d = self.calculate_query_density(u, d0, self.N0, self.N1)
            target_rho_ad = self.calculate_query_density(u, a0, self.N0, self.N1)

            sp_d0 = solved_en_point[d0]
            sp_a0 = solved_en_point[a0]

            for cand_ep in unsolved_candidates.get(u, []):
                if cand_ep in solved_en_point.values():
                    continue
                if (
                    rho_pair(cand_ep, sp_d0) == target_rho_d
                    and rho_pair(cand_ep, sp_a0) == target_rho_ad
                ):
                    solved_en_point[u] = cand_ep
                    break

        print(f"Phase-A complete, total solved points: {len(solved_en_point)}")
        return solved_en_point

    # --------------------------------------------------------------------- #
    #  Phase-B：关键词 / 文档映射
    # --------------------------------------------------------------------- #
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

    # ------------------------------------------------------------------ #
    #                   ——  以下为静态 / 工具函数 ——                      #
    # ------------------------------------------------------------------ #
    @staticmethod
    def build_inverted_index(RM):
        point2resp = defaultdict(set)
        for rid, resp in enumerate(RM):
            for p in resp:
                point2resp[p].add(rid)
        single_rho = {p: len(r) for p, r in point2resp.items()}
        return point2resp, single_rho

    @staticmethod
    def make_pair_density_fn(point2resp):
        @lru_cache(maxsize=1 << 20)
        def pair_density(p, q):
            if p == q:
                return len(point2resp.get(p, set()))
            return len(point2resp.get(p, set()) & point2resp.get(q, set()))

        return pair_density

    @staticmethod
    def calculate_query_density(u, r, N0, N1):
        ux, uy = u
        rx, ry = r
        if ux <= rx and uy <= ry:
            return ux * uy * (N0 + 1 - rx) * (N1 + 1 - ry)
        if rx <= ux and ry <= uy:
            return rx * ry * (N0 + 1 - ux) * (N1 + 1 - uy)
        if ux <= rx and uy >= ry:
            return ux * (N0 + 1 - rx) * ry * (N1 + 1 - uy)
        if rx <= ux and ry >= uy:
            return rx * (N0 + 1 - ux) * uy * (N1 + 1 - ry)
        raise ValueError("u 与 r 无支配 / 反支配关系")

    # ------------------------------------------------------------------ #
    #                 ——  两种初始锚点生成策略  ——                       #
    # ------------------------------------------------------------------ #
    # @staticmethod
    # def find_initial_point_mapping_by_matrix(
    #     known_records,
    #     en_records,
    #     M,
    #     M_prime,
    #     ordered_enc_doc_ids,
    #     known_global_ids,
    #     mode,
    # ):
    #     # ---------- 1. volume 唯一值建立 doc_id 映射 ----------
    #     if M.size == 0 or M_prime.size == 0:
    #         print("  * M 或 M' 为空，无法根据矩阵建立锚点")
    #         return {}, {} # 修改点: 返回两个空字典
    #
    #     global_id_to_local_idx = {gid: i for i, gid in enumerate(known_global_ids)}
    #     enc_id_to_matrix_idx = {eid: i for i, eid in enumerate(ordered_enc_doc_ids)}
    #
    #     known_volumes = {
    #         gid: M_prime[global_id_to_local_idx[gid], global_id_to_local_idx[gid]]
    #         for gid in known_global_ids
    #     }
    #     encrypted_volumes = {
    #         eid: M[enc_id_to_matrix_idx[eid], enc_id_to_matrix_idx[eid]]
    #         for eid in ordered_enc_doc_ids
    #     }
    #
    #     unique_known = {v for v, c in Counter(known_volumes.values()).items() if c == 1}
    #     unique_enc = {v for v, c in Counter(encrypted_volumes.values()).items() if c == 1}
    #     recoverable = unique_known & unique_enc
    #     if not recoverable:
    #         return {}, {} # 修改点: 返回两个空字典
    #
    #     vol2kn = {v: gid for gid, v in known_volumes.items()}
    #     vol2en = {v: eid for eid, v in encrypted_volumes.items()}
    #     # 这是您需要的文档映射
    #     doc_map = {vol2kn[v]: vol2en[v] for v in recoverable}
    #
    #     # ---------- 2. 出现位置索引 ----------
    #     kn_doc2coords = defaultdict(list)
    #     for coord, gid in known_records:
    #         if gid in doc_map:
    #             kn_doc2coords[gid].append(coord)
    #
    #     en_doc2coords = defaultdict(list)
    #     for coord, eid in en_records:
    #         if eid in doc_map.values():
    #             en_doc2coords[eid].append(coord)
    #
    #     # ---------- 3. 筛选一对一 -> 坐标映射 ----------
    #     point_map = {}
    #     for kn_gid, en_id in doc_map.items():
    #         kn_locs = kn_doc2coords.get(kn_gid, [])
    #         en_locs = en_doc2coords.get(en_id, [])
    #         if mode == "O2M":
    #             if kn_locs and en_locs:
    #                 point_map[kn_locs[0]] = en_locs[0]
    #         else:  # M2M
    #             if len(kn_locs) == 1 and len(en_locs) == 1:
    #                 point_map[kn_locs[0]] = en_locs[0]
    #
    #     # 修改点: 同时返回 point_map 和 doc_map
    #     return point_map, doc_map
    #
    # @staticmethod
    # def find_initial_mapping_by_volume(
    #     known_records,
    #     en_records,
    #     known_volumes,
    #     records_volume,
    #     mode,
    # ):
    #     print(f"\nAttempting volume-based mapping ({mode}) ...")
    #
    #     unique_kn = {v for v, c in Counter(known_volumes).items() if c == 1}
    #     unique_en = {v for v, c in Counter(records_volume).items() if c == 1}
    #     recoverable = unique_kn & unique_en
    #     if not recoverable:
    #         print("  * no unique volume value found")
    #         return {}, {}  # 修改点: 返回两个空字典
    #
    #     vol2kn = {}
    #     for (coord, gid), vol in zip(known_records, known_volumes):
    #         if vol in recoverable:
    #             vol2kn[vol] = gid
    #
    #     vol2en = {}
    #     for (coord, eid), vol in zip(en_records, records_volume):
    #         if vol in recoverable:
    #             vol2en[vol] = eid
    #
    #     # 这是您需要的文档映射
    #     doc_map = {vol2kn[v]: vol2en[v] for v in recoverable}
    #
    #     # 出现位置索引
    #     kn_doc2coords = defaultdict(list)
    #     for coord, gid in known_records:
    #         if gid in doc_map:
    #             kn_doc2coords[gid].append(coord)
    #
    #     en_doc2coords = defaultdict(list)
    #     for coord, eid in en_records:
    #         if eid in doc_map.values():
    #             en_doc2coords[eid].append(coord)
    #
    #     point_map = {}
    #     for kn_gid, en_id in doc_map.items():
    #         kn_locs = kn_doc2coords.get(kn_gid, [])
    #         en_locs = en_doc2coords.get(en_id, [])
    #         if mode == "O2M":
    #             if kn_locs and en_locs:
    #                 point_map[kn_locs[0]] = en_locs[0]
    #         else:  # M2M
    #             if len(kn_locs) == 1 and len(en_locs) == 1:
    #                 point_map[kn_locs[0]] = en_locs[0]
    #
    #     print(
    #         f"  * recovered {len(point_map)} anchor point(s) and {len(doc_map)} document mapping(s) from volume feature.")
    #
    #     # 修改点: 同时返回 point_map 和 doc_map
    #     return point_map, doc_map
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
        known_volumes_list, # 注意：原函数中的 known_volumes 是一个列表
        encrypted_volumes_list, # 原函数中的 records_volume
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