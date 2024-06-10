import numpy as np
import argparse
import random
import pandas as pd
from tensorflow.keras import models, layers, optimizers, losses
from tensorflow.keras.utils import custom_object_scope
import tensorflow as tf
from help_code_demo_tf import ECG_DataSET, ToTensor, create_dataset, F1, FB, Sensitivity, Specificity, BAC, ACC, PPV, NPV
from iesdcontest2024_demo_example_tensorflow.models.model_tf_2 import AFNet


def main():
    seed = 222
    tf.random.set_seed(seed)
    np.random.seed(seed)
    random.seed(seed)

    # Hyperparameters
    BATCH_SIZE_TEST = 1
    SIZE = args.size
    path_data = args.path_data
    path_records = args.path_record
    path_net = args.path_net
    path_indices = args.path_indices

    test_indice_path = args.path_indices + 'test_indice.csv'
    test_indices = pd.read_csv(test_indice_path)  # Adjust delimiter if necessary

    physical_devices = tf.config.list_physical_devices('GPU')
    tf.config.experimental.set_memory_growth(physical_devices[0], True)

    subjects = test_indices['Filename'].apply(lambda x: x.split('-')[0]).unique().tolist()

    # List to store metrics for each participant
    subject_metrics = []

    # Load trained network
    net = models.load_model(path_net + 'ECG_net_tf.h5')

    subjects_above_threshold = 0

    print(subjects)
    test_counter = 0
    for subject_id in subjects:
        print(subject_id)
        testset = ECG_DataSET(root_dir=path_data,
                               indice_dir=path_indices,
                               mode='test',
                               size=SIZE,
                               subject_id=subject_id,
                               transform=ToTensor())

        testloader = create_dataset(testset, BATCH_SIZE_TEST)

        segs_TP = 0
        segs_TN = 0
        segs_FP = 0
        segs_FN = 0

        for ECG_test, labels_test in testloader:
            predictions = net(ECG_test, training=False)
            predicted_test = tf.argmax(predictions, axis=1)

            seg_label = labels_test.numpy()[0]

            if seg_label == 0:
                segs_FP += np.sum(predicted_test.numpy() != labels_test.numpy())
                segs_TN += np.sum(predicted_test.numpy() == labels_test.numpy())
            elif seg_label == 1:
                segs_FN += np.sum(predicted_test.numpy() != labels_test.numpy())
                segs_TP += np.sum(predicted_test.numpy() == labels_test.numpy())

            test_counter += 1
            # print(predictions, test_counter)

        # Calculate metrics for the current participant
        f1 = round(F1([segs_TP, segs_FN, segs_FP, segs_TN]), 5)
        fb = round(FB([segs_TP, segs_FN, segs_FP, segs_TN]), 5)
        se = round(Sensitivity([segs_TP, segs_FN, segs_FP, segs_TN]), 5)
        sp = round(Specificity([segs_TP, segs_FN, segs_FP, segs_TN]), 5)
        bac = round(BAC([segs_TP, segs_FN, segs_FP, segs_TN]), 5)
        acc = round(ACC([segs_TP, segs_FN, segs_FP, segs_TN]), 5)
        ppv = round(PPV([segs_TP, segs_FN, segs_FP, segs_TN]), 5)
        npv = round(NPV([segs_TP, segs_FN, segs_FP, segs_TN]), 5)

        subject_metrics.append([f1, fb, se, sp, bac, acc, ppv, npv])
        if fb > 0.9:
            subjects_above_threshold += 1

    subject_metrics_array = np.array(subject_metrics)
    average_metrics = np.mean(subject_metrics_array, axis=0)

    avg_f1, avg_fb, avg_se, avg_sp, avg_bac, avg_acc, avg_ppv, avg_npv = average_metrics

    # Print average metric values
    print(f"Final F-1: {avg_f1:.5f}")
    print(f"Final F-B: {avg_fb:.5f}")
    print(f"Final SEN: {avg_se:.5f}")
    print(f"Final SPE: {avg_sp:.5f}")
    print(f"Final BAC: {avg_bac:.5f}")
    print(f"Final ACC: {avg_acc:.5f}")
    print(f"Final PPV: {avg_ppv:.5f}")
    print(f"Final NPV: {avg_npv:.5f}")

    proportion_above_threshold = subjects_above_threshold / len(subjects)
    print("G Score:", proportion_above_threshold)

    with open(path_records + 'seg_stat.txt', 'w') as f:
        f.write(f"Final F-1: {avg_f1:.5f}\n")
        f.write(f"Final F-B: {avg_fb:.5f}\n")
        f.write(f"Final SEN: {avg_se:.5f}\n")
        f.write(f"Final SPE: {avg_sp:.5f}\n")
        f.write(f"Final BAC: {avg_bac:.5f}\n")
        f.write(f"Final ACC: {avg_acc:.5f}\n")
        f.write(f"Final PPV: {avg_ppv:.5f}\n")
        f.write(f"Final NPV: {avg_npv:.5f}\n\n")
        f.write(f"G Score: {proportion_above_threshold}\n")

if __name__ == '__main__':
    argparser = argparse.ArgumentParser()
    argparser.add_argument('--size', type=int, default=1250)
    argparser.add_argument('--path_data', type=str, default='D:/Data_Experiment/data_ECGdb_ICMCdb/training_dataset/')
    argparser.add_argument('--path_net', type=str, default='./saved_models/')
    argparser.add_argument('--path_record', type=str, default='./records/')
    argparser.add_argument('--path_indices', type=str, default='./data_indices/')
    args = argparser.parse_args()

    main()
