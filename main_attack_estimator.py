import pickle

import numpy as np
from tqdm import tqdm

from Database import Database
from attack.Attack_estimator import Attack_estimator,DensityEstimator
from utils import Range_statistic, Keywords_statistic, Overall_statistics


def main():
    with open("data/yfcc_25x25_4w.pkl", "rb") as f:
        raw_records = pickle.load(f)
    DB = Database(raw_records, mode='O2M')
    idx,known_records = DB.get_known_part(0.1)
    M_, known_global_ids = DB.generate_doc_occ_matrix(known_records)
    known_volumes = [DB.records_volume[i - 1] for i in idx]
    N0, N1 =25,25
    RM,token_response_pairs = DB.simulate_client_queries(N0, N1,0.5,"uniform")
    DE = DensityEstimator(token_response_pairs,"chao_lee")
    AT = Attack_estimator(known_records,DB.en_records,DE,20,0.3,N0,N1,30,
                          5,DB.mode,DB.matrix_M, M_, known_volumes,DB.records_volume, DB.ordered_enc_doc_ids,known_global_ids, 80,30)
    points_map, doc_map = AT.attack()

    Range_statistic(points_map, DB.points_mapping[0], known_records)
    Keywords_statistic(doc_map,DB.documents_mapping[0], known_global_ids)
    Overall_statistics(known_records, DB.points_mapping[0], DB.documents_mapping[0], points_map, doc_map)



if __name__ == '__main__':
    main()