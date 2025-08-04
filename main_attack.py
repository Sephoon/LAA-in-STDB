import pickle
from Database import Database
from attack.Attack import Attack
from utils import Range_statistic, Keywords_statistic, Overall_statistics


def main():
    with open("data/yfcc_50x50_4w.pkl", "rb") as f:
        raw_records = pickle.load(f)
    DB = Database(raw_records, mode='O2M')
    idx,known_records = DB.get_known_part(1)
    M_, known_global_ids = DB.generate_doc_occ_matrix(known_records)
    known_volumes = [DB.records_volume[i - 1] for i in idx]
    N0, N1 = 50,50
    RM = DB.get_all_query_response_with_value(N0, N1)
    AT = Attack(known_records, DB.en_records, DB.mode,RM, N0, N1, DB.matrix_M, M_,known_volumes,DB.records_volume, DB.ordered_enc_doc_ids,known_global_ids, 50)
    points_map, doc_map = AT.attack()

    Range_statistic(points_map, DB.points_mapping[0], known_records)
    Keywords_statistic(doc_map,DB.documents_mapping[0], known_global_ids)
    Overall_statistics(known_records, DB.points_mapping[0], DB.documents_mapping[0], points_map, doc_map)


if __name__ == '__main__':
    main()