


def Range_statistic(solved_map, ground_truth_map, known_records):
    """计算攻击的准确率和召回率。

    Args:
        solved_map (dict): 攻击算法得出的 `明文 -> 密文` 映射。
        ground_truth_map (dict): 真实的 `明文 -> 密文` 映射。
        known_records (list): 用于攻击的已知记录列表。

    Returns:
        tuple: (precision, recall)
    """
    known_points = {c for c, _ in known_records}

    correct_count = 0
    for plain_point, solved_en_point in solved_map.items():
        if ground_truth_map.get(plain_point) == solved_en_point:
            correct_count += 1

    solved_count = len(solved_map)
    known_points_num = len(known_points)

    precision = correct_count / solved_count if solved_count > 0 else 0
    recall = solved_count / known_points_num if known_points_num > 0 else 0

    print(f"\nAttack finished. Total points mapped: {len(solved_map)}")
    print(f"  - Correctly solved: {correct_count}")
    print(f"  - Total solved: {solved_count}")
    print(f"  - Total known points to be solved: {known_points_num}")
    print(f"  - Precision: {precision:.4f}")
    print(f"  - Recall: {recall:.4f}")

    return precision, recall

def Keywords_statistic(solved_map, ground_truth_map, known_records):

    known_points = set(known_records)

    correct_count = 0
    for plain_kw, solved_cipher in solved_map.items():
        if ground_truth_map.get(plain_kw) == solved_cipher:
            correct_count += 1

    solved_count       = len(solved_map)
    known_points_num   = len(known_points)

    precision = correct_count / solved_count     if solved_count     > 0 else 0
    recall    = solved_count / known_points_num if known_points_num > 0 else 0
    print(f"\nAttack finished. Total document mapped: {len(solved_map)}")
    print(f"  - Correctly solved: {correct_count}")
    print(f"  - Total solved: {solved_count}")
    print(f"  - Total known points to be solved: {known_points_num}")
    print(f"  - Precision: {precision:.4f}")
    print(f"  - Recall: {recall:.4f}")

    return precision, recall

def Overall_statistics(known_records, pt2ct_points, pt2ct_doc, solved_points_map, solved_doc_map ):
    num_known     = len(known_records)   # 已知明文总数
    num_recovered = 0                    # 已恢复(无论对错)
    num_correct   = 0                    # 已恢复且正确

    # -- 遍历每一条已知明文 ----------
    for point, doc_id in known_records:

        point_recovered = False   # 该条是否被任何阶段恢复
        point_correct   = True    # 恢复是否正确（只有 recovered == True 时才有意义）
        doc_recovered = False
        doc_correct = True


        # 1) 位置(point) 部分
        if point in solved_points_map:
            point_recovered = True
            if pt2ct_points.get(point) != solved_points_map[point]:
                point_correct = False


        # 2) 文档/关键字(doc_id) 部分
        if doc_id in solved_doc_map:
            doc_recovered = True
            if pt2ct_doc.get(doc_id) != solved_doc_map[doc_id]:
                doc_correct = False

        if point_recovered and doc_recovered:
            num_recovered += 1
            if point_correct and doc_correct:
                num_correct += 1

    # -- 指标计算 ----------
    recovery_rate = num_recovered / num_known   if num_known     else 0
    accuracy      = num_correct   / num_recovered if num_recovered else 0

    # -- 打印 ----------
    print("========== Overall Statistics (Location + Keywords) ==========")
    print(f"  已知明文数                 : {num_known}")
    print(f"  已恢复条数                 : {num_recovered}")
    print(f"  其中正确条数               : {num_correct}")
    print(f"  Recovery rate (覆盖率)      : {recovery_rate:.4f}")
    print(f"  Accuracy (正确率)           : {accuracy:.4f}")
    print("==============================================================")

    return recovery_rate, accuracy

