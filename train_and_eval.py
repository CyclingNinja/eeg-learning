import time
import csv
import mne
import numpy as np
import pandas as pd
import torch
import matplotlib as plt
from sklearn.metrics import confusion_matrix
from braindecode.visualization import plot_confusion_matrix
from eeg_learning.models import ModelFactory
from eeg_learning.io.dataset_builder import DatasetBuilder
from eeg_learning.tools.metrics import convolution_matrix, find_all_zero, weight_function
from eeg_learning.tools.dataset_splitting import DatasetSplitter
from eeg_learning.tools.plotting import validation_graph
from eeg_learning.config import load

from torch.nn.functional import elu, relu, gelu

import warnings

warnings.filterwarnings("once")

pd.set_option("display.max_columns", 10)

cfg = load()

run = cfg["run"]
data = cfg["data"]
preprocessing = cfg["preprocessing"]
windowing = cfg["windowing"]
split = cfg["split"]
training = cfg["training"]
output = cfg["output"]
model_cfg = cfg["model"]
deep4 = cfg["model"]["deep4"]
tcn = cfg["model"]["tcn"]
shallow = cfg["model"]["shallow"]
vit = cfg["model"]["vit"]


with open(output["log_path"], "a") as f:
    writer = csv.writer(
        f,
        delimiter=",",
        lineterminator="\n",
    )
    writer.writerow([time.strftime("%Y-%m-%d_%H:%M:%S", time.localtime(time.time()))])
    writer.writerow(
        [
            "train_loss",
            "valid_loss",
            "train_accuracy",
            "valid_accuracy",
            "etl_time",
            "model_training_time",
            "test_acc",
            "test_precision",
            "test_recall",
            "n_repetition",
            "random_state",
            "tuab",
            "tueg",
            "n_tuab",
            "n_tueg",
            "n_load",
            "preload",
            "window_len_s",
            "tuab_path",
            "tueg_path",
            "saved_data",
            "saved_path",
            "saved_windows_data",
            "saved_windows_path",
            "load_saved_data",
            "load_saved_windows",
            "bandpass_filter",
            "low_cut_hz",
            "high_cut_hz",
            "standardization",
            "factor_new",
            "init_block_size",
            "n_jobs",
            "n_classes",
            "lr",
            "weight_decay",
            "batch_size",
            "n_epochs",
            "tmin",
            "tmax",
            "multiple",
            "sec_to_cut",
            "duration_recording_sec",
            "max_abs_val",
            "sampling_freq",
            "test_on_eval",
            "split_way",
            "train_size",
            "valid_size",
            "test_size",
            "shuffle",
            "model_name",
            "final_conv_length",
            "window_stride_samples",
            "relabel_dataset",
            "relabel_label",
            "channels",
            "dropout",
            "precision_per_recording",
            "recall_per_recording",
            "acc_per_recording",
            "mcc",
            "mcc_per_recording",
            "activation",
            "remove_attribute",
        ]
    )


print(
    run["random_state"],
    data["tuab"],
    data["tueg"],
    data["n_tuab"],
    data["n_tueg"],
    data["n_load"],
    data["preload"],
    windowing["window_len_s"],
    data["tuab_path"],
    data["tueg_path"],
    data["save_recordings"],
    data["save_recordings_path"],
    data["save_windows"],
    data["save_windows_path"],
    data["load_saved_recordings"],
    data["load_saved_windows"],
    preprocessing["bandpass_filter"],
    preprocessing["low_cut_hz"],
    preprocessing["high_cut_hz"],
    preprocessing["standardization"],
    preprocessing["factor_new"],
    preprocessing["init_block_size"],
    run["n_jobs"],
    windowing["tmin"],
    windowing.get("tmax"),
    windowing["multiple"],
    windowing["sec_to_cut"],
    windowing["duration_recording_sec"],
    preprocessing["max_abs_val"],
    preprocessing["sampling_freq"],
    training["test_on_eval"],
    split["split_way"],
    split["train_size"],
    split["valid_size"],
    split["test_size"],
    split["shuffle"],
    data["relabel_datasets"],
    data["relabel_labels"],
    windowing["channels"],
)

cuda = torch.cuda.is_available()  # check if GPU is available, if True chooses to use it
device = "cuda" if cuda else "cpu"
print("device:", device)
if cuda:
    torch.backends.cudnn.benchmark = True
torch.set_num_threads(run["n_jobs"])  # Sets the available number of threads

mne.set_log_level(run["mne_log_level"])

data_loading_start = time.time()

windows_ds = DatasetBuilder(
    load_saved_windows=data["load_saved_windows"],
    saved_windows_path=data["save_windows_path"],
    save_windows=data["save_windows"],
    load_saved_data=data["load_saved_recordings"],
    saved_data_path=data["save_recordings_path"],
    save_preprocessed=data["save_recordings"],
    use_tuab=data["tuab"],
    use_tueg=data["tueg"],
    tuab_path=data["tuab_path"],
    tueg_path=data["tueg_path"],
    n_tuab=data["n_tuab"],
    n_tueg=data["n_tueg"],
    tmin=windowing["tmin"],
    tmax=windowing.get("tmax"),
    channels=windowing["channels"],
    relabel_label=data["relabel_labels"],
    relabel_dataset=data["relabel_datasets"],
    sampling_freq=preprocessing["sampling_freq"],
    sec_to_cut=windowing["sec_to_cut"],
    duration_recording_sec=windowing["duration_recording_sec"],
    max_abs_val=preprocessing["max_abs_val"],
    multiple=windowing["multiple"],
    bandpass_filter=preprocessing["bandpass_filter"],
    low_cut_hz=preprocessing["low_cut_hz"],
    high_cut_hz=preprocessing["high_cut_hz"],
    standardization=preprocessing["standardization"],
    factor_new=preprocessing["factor_new"],
    init_block_size=preprocessing["init_block_size"],
    window_len_s=windowing["window_len_s"],
    window_stride_samples=windowing.get("window_stride_samples"),
    n_load=data["n_load"],
    preload=data["preload"],
    n_jobs=run["n_jobs"],
).build()
window_len_samples = windows_ds[0][0].shape[1]

# Split the data:
splitter = DatasetSplitter(
    windows_ds,
    split["train_size"],
    split["valid_size"],
    split["test_size"],
    run["random_state"],
    shuffle=split["shuffle"],
    remove_attribute=None,
)
train_set, valid_set, test_set = splitter.split_data(split["split_way"])
print("len_valid_train", len(train_set.description.loc[:, ["path"]]) + len(valid_set.description.loc[:, ["path"]]))
print("len_test", len(test_set.description.loc[:, ["path"]]))
etl_time = time.time() - data_loading_start

n_channels = windows_ds[0][0].shape[0]

print("n_channels:", n_channels)


for i in range(run["n_repetitions"]):
    print(
        i,
        run["random_state"],
        data["tuab"],
        data["tueg"],
        data["n_tuab"],
        data["n_tueg"],
        data["n_load"],
        data["preload"],
        windowing["window_len_s"],
        data["tuab_path"],
        data["tueg_path"],
        data["save_recordings"],
        data["save_recordings_path"],
        data["save_windows"],
        data["save_windows_path"],
        data["load_saved_recordings"],
        data["load_saved_windows"],
        preprocessing["bandpass_filter"],
        preprocessing["low_cut_hz"],
        preprocessing["high_cut_hz"],
        preprocessing["standardization"],
        preprocessing["factor_new"],
        preprocessing["init_block_size"],
        run["n_jobs"],
        training["n_classes"],
        training["learning_rate"],
        training["weight_decay"],
        training["batch_size"],
        training["n_epochs"],
        windowing["tmin"],
        windowing.get("tmax"),
        windowing["multiple"],
        windowing["sec_to_cut"],
        windowing["duration_recording_sec"],
        preprocessing["max_abs_val"],
        preprocessing["sampling_freq"],
        training["test_on_eval"],
        split["split_way"],
        split["train_size"],
        split["valid_size"],
        split["test_size"],
        split["shuffle"],
        model_cfg["name"],
        model_cfg["final_conv_length"],
        windowing.get("window_stride_samples"),
        data["relabel_datasets"],
        data["relabel_labels"],
        windowing["channels"],
    )
    if split["shuffle"] and i > 0:
        # Re-split the data to ensure each repetition uses a different split:
        splitter = DatasetSplitter(
            windows_ds,
            split["train_size"],
            split["valid_size"],
            split["test_size"],
            run["random_state"] + i,
            shuffle=split["shuffle"],
        )
        train_set, valid_set, test_set = splitter.split_data(split["split_way"])

    mne.set_log_level(run["mne_log_level"])

    def exp(dropout=0.2):
        if deep4["activation"] == "elu":  # choose the activation function
            nonlin = elu
        elif deep4["activation"] == "relu":
            nonlin = relu
        elif deep4["activation"] == "gelu":
            nonlin = gelu

        # select the model(first-stage)
        if model_cfg["name"] in ModelFactory.available():
            model = ModelFactory.create(
                model_cfg["name"],
                n_channels=n_channels,
                n_classes=training["n_classes"],
                input_window_samples=window_len_samples,
                drop_prob=dropout,
                final_conv_length=model_cfg["final_conv_length"],
                # deep4
                deep4_n_filters_time=deep4["n_filters_time"],
                deep4_n_filters_spat=deep4["n_filters_spat"],
                deep4_filter_time_length=deep4["filter_time_length"],
                deep4_pool_time_length=deep4["pool_time_length"],
                deep4_pool_time_stride=deep4["pool_time_stride"],
                deep4_n_filters_2=deep4["n_filters_2"],
                deep4_filter_length_2=deep4["filter_length_2"],
                deep4_n_filters_3=deep4["n_filters_3"],
                deep4_filter_length_3=deep4["filter_length_3"],
                deep4_n_filters_4=deep4["n_filters_4"],
                deep4_filter_length_4=deep4["filter_length_4"],
                deep4_first_pool_mode=deep4["first_pool_mode"],
                deep4_later_pool_mode=deep4["later_pool_mode"],
                first_nonlin=nonlin,
                later_nonlin=nonlin,
                # shallow_smac
                shallow_n_filters_time=shallow["n_filters_time"],
                shallow_filter_time_length=shallow["filter_time_length"],
                shallow_n_filters_spat=shallow["n_filters_spat"],
                shallow_pool_time_length=shallow["pool_time_length"],
                shallow_pool_time_stride=shallow["pool_time_stride"],
                shallow_split_first_layer=shallow["split_first_layer"],
                shallow_batch_norm=shallow["batch_norm"],
                shallow_batch_norm_alpha=shallow["batch_norm_alpha"],
                # tcn_1
                n_blocks=tcn["n_blocks"],
                n_filters=tcn["n_filters"],
                kernel_size=tcn["kernel_size"],
                add_log_softmax=tcn["add_log_softmax"],
                last_layer_type=tcn["last_layer_type"],
                # vit
                patch_size=vit["patch_size"],
                dim=vit["dim"],
                depth=vit["depth"],
                heads=vit["heads"],
                mlp_dim=vit["mlp_dim"],
                emb_dropout=vit["emb_dropout"],
                # sleep2020 / sleep2018 / usleep
                sampling_freq=preprocessing["sampling_freq"],
            )

        if cuda:
            model.cuda()
        training_setup_end = time.time()

        # Start training loop
        model_training_start = time.time()

        from skorch.callbacks import LRScheduler
        from skorch.helper import predefined_split
        from braindecode import EEGClassifier
        from skorch.callbacks import Checkpoint, EarlyStopping

        # set the learning rate scheduler
        monitor = lambda net: all(net.history[-1, ("train_loss_best", "valid_loss_best")])
        cp = Checkpoint(monitor=monitor, dirname="", f_criterion=None, f_optimizer=None, load_best=False)
        callbacks = [
            "accuracy",
            ("lr_scheduler", LRScheduler("CosineAnnealingLR", T_max=training["n_epochs"] - 1)),
            ("cp", cp),
        ]
        if training["earlystopping"]:
            es_patience = training["n_epochs"] // 3
            es = EarlyStopping(threshold=training["es_threshold"], threshold_mode="rel", patience=es_patience)
            callbacks.append(("es", es))

        # Set various parameters for training
        eeg_classifier = EEGClassifier(
            model,
            criterion=torch.nn.NLLLoss(weight_function(train_set.get_metadata().target, device)),
            optimizer=torch.optim.AdamW,
            train_split=predefined_split(valid_set) if training["test_on_eval"] else None,
            optimizer__lr=training["learning_rate"],
            optimizer__weight_decay=training["weight_decay"],
            batch_size=training["batch_size"],
            callbacks=callbacks,
            device=device,
        )

        # Prevent GPU memory fragmentation
        torch.cuda.empty_cache()

        # Model training for a specified number of epochs. `y` is None as it is already supplied
        # in the dataset.
        global i
        if not output["load_pretrained_model"]:  # Choose to load a model or train a model
            eeg_classifier.fit(train_set, y=None, epochs=training["n_epochs"])
            eeg_classifier.save_params(
                "./saved_models/"
                + model_cfg["name"]
                + time.strftime("%Y-%m-%d_%H-%M-%S", time.localtime(time.time()))
                + "params.pt"
            )
        else:
            eeg_classifier.initialize()
            eeg_classifier.load_params("./saved_models/" + params[i])
        model_training_time = time.time() - model_training_start

        if not output["load_pretrained_model"]:
            # Extract loss and accuracy values for plotting from history object
            results_columns = ["train_loss", "valid_loss", "train_accuracy", "valid_accuracy"]

            df = pd.DataFrame(
                eeg_classifier.history[:, results_columns],
                columns=results_columns,
                index=eeg_classifier.history[:, "epoch"],
            )
            # get percent of misclass for better visual comparison to loss
            df = df.assign(train_misclass=100 - 100 * df.train_accuracy, valid_misclass=100 - 100 * df.valid_accuracy)
            print(df)

            if output["plot_result"]:  # whether plot the result
                validation_graph(eeg_classifier)

        # test on the testset
        print("test:", test_set.description)
        y_true = test_set.get_metadata().target
        starts = find_all_zero(test_set.get_metadata()["i_window_in_trial"].tolist())
        y_pred = eeg_classifier.predict(test_set)
        y_pred_proba = eeg_classifier.predict_proba(test_set)
        print("diff:", sum((np.exp(np.array(y_pred_proba[:, 1])) > 0.5) != y_pred))

        # generate confusion matrices
        confusion_mat_per_recording = convolution_matrix(starts, y_true, y_pred)
        confusion_mat_per_recording_proba = convolution_matrix(starts, y_true, y_pred, True, y_pred_proba)
        print(confusion_mat_per_recording)
        print(confusion_mat_per_recording_proba)

        confusion_mat = confusion_matrix(y_true, y_pred)
        print(confusion_mat)

        # generate various evaluation index
        precision = confusion_mat[0, 0] / (confusion_mat[0, 0] + confusion_mat[1, 0])
        recall = confusion_mat[0, 0] / (confusion_mat[0, 0] + confusion_mat[0, 1])
        acc = (confusion_mat[0, 0] + confusion_mat[1, 1]) / (
            confusion_mat[0, 0] + confusion_mat[0, 1] + confusion_mat[1, 1] + confusion_mat[1, 0]
        )
        mcc = MCC(confusion_mat)
        precision_per_recording = confusion_mat_per_recording[0, 0] / (
            confusion_mat_per_recording[0, 0] + confusion_mat_per_recording[1, 0]
        )
        recall_per_recording = confusion_mat_per_recording[0, 0] / (
            confusion_mat_per_recording[0, 0] + confusion_mat_per_recording[0, 1]
        )
        acc_per_recording = (confusion_mat_per_recording[0, 0] + confusion_mat_per_recording[1, 1]) / (
            confusion_mat_per_recording[0, 0]
            + confusion_mat_per_recording[0, 1]
            + confusion_mat_per_recording[1, 1]
            + confusion_mat_per_recording[1, 0]
        )
        mcc_per_recording = MCC(confusion_mat_per_recording)
        end = time.time()
        print("precision:", precision)
        print("recall:", recall)
        print("acc:", acc)
        print("mcc:", mcc)
        print("precision_per_recording:", precision_per_recording)
        print("recall_per_recording:", recall_per_recording)
        print("acc_per_recording:", acc_per_recording)
        print("mcc:", mcc_per_recording)
        print("etl_time:", etl_time)
        print("model_training_time:", model_training_time)

        with open(output["log_path"], "a") as f:  # save the results
            writer = csv.writer(
                f,
                delimiter=",",
                lineterminator="\n",
            )
            if not output["load_pretrained_model"]:
                his_len = len(df)
                for i2 in range(his_len - 1):
                    writer.writerow([df.loc[i2 + 1][0], df.loc[i2 + 1][1], df.loc[i2 + 1][2], df.loc[i2 + 1][3]])

                writer.writerow(
                    [
                        df.loc[his_len][0],
                        df.loc[his_len][1],
                        df.loc[his_len][2],
                        df.loc[his_len][3],
                        etl_time,
                        model_training_time,
                        acc,
                        precision,
                        recall,
                        i,
                        run["random_state"],
                        data["tuab"],
                        data["tueg"],
                        data["n_tuab"],
                        data["n_tueg"],
                        data["n_load"],
                        data["preload"],
                        windowing["window_len_s"],
                        data["tuab_path"],
                        data["tueg_path"],
                        data["save_recordings"],
                        data["save_recordings_path"],
                        data["save_windows"],
                        data["save_windows_path"],
                        data["load_saved_recordings"],
                        data["load_saved_windows"],
                        preprocessing["bandpass_filter"],
                        preprocessing["low_cut_hz"],
                        preprocessing["high_cut_hz"],
                        preprocessing["standardization"],
                        preprocessing["factor_new"],
                        preprocessing["init_block_size"],
                        run["n_jobs"],
                        training["n_classes"],
                        training["learning_rate"],
                        training["weight_decay"],
                        training["batch_size"],
                        training["n_epochs"],
                        windowing["tmin"],
                        windowing.get("tmax"),
                        windowing["multiple"],
                        windowing["sec_to_cut"],
                        windowing["duration_recording_sec"],
                        preprocessing["max_abs_val"],
                        preprocessing["sampling_freq"],
                        training["test_on_eval"],
                        split["split_way"],
                        split["train_size"],
                        split["valid_size"],
                        split["test_size"],
                        split["shuffle"],
                        model_cfg["name"],
                        model_cfg["final_conv_length"],
                        windowing.get("window_stride_samples"),
                        data["relabel_datasets"],
                        data["relabel_labels"],
                        windowing["channels"],
                        dropout,
                        precision_per_recording,
                        recall_per_recording,
                        acc_per_recording,
                        mcc,
                        mcc_per_recording,
                        deep4["activation"],
                        None,
                    ]
                )
            else:
                writer.writerow(
                    [
                        "test_model",
                        "test_model",
                        "test_model",
                        "test_model",
                        etl_time,
                        model_training_time,
                        acc,
                        precision,
                        recall,
                        i,
                        run["random_state"],
                        data["tuab"],
                        data["tueg"],
                        data["n_tuab"],
                        data["n_tueg"],
                        data["n_load"],
                        data["preload"],
                        windowing["window_len_s"],
                        data["tuab_path"],
                        data["tueg_path"],
                        data["save_recordings"],
                        data["save_recordings_path"],
                        data["save_windows"],
                        data["save_windows_path"],
                        data["load_saved_recordings"],
                        data["load_saved_windows"],
                        preprocessing["bandpass_filter"],
                        preprocessing["low_cut_hz"],
                        preprocessing["high_cut_hz"],
                        preprocessing["standardization"],
                        preprocessing["factor_new"],
                        preprocessing["init_block_size"],
                        run["n_jobs"],
                        training["n_classes"],
                        training["learning_rate"],
                        training["weight_decay"],
                        training["batch_size"],
                        training["n_epochs"],
                        windowing["tmin"],
                        windowing.get("tmax"),
                        windowing["multiple"],
                        windowing["sec_to_cut"],
                        windowing["duration_recording_sec"],
                        preprocessing["max_abs_val"],
                        preprocessing["sampling_freq"],
                        training["test_on_eval"],
                        split["split_way"],
                        split["train_size"],
                        split["valid_size"],
                        split["test_size"],
                        split["shuffle"],
                        model_cfg["name"],
                        model_cfg["final_conv_length"],
                        windowing.get("window_stride_samples"),
                        data["relabel_datasets"],
                        data["relabel_labels"],
                        windowing["channels"],
                        dropout,
                        precision_per_recording,
                        recall_per_recording,
                        acc_per_recording,
                        mcc,
                        mcc_per_recording,
                        deep4["activation"],
                        None,
                    ]
                )

        if output["plot_result"]:
            labels = ["normal", "abnormal"]
            plot_confusion_matrix(confusion_mat, class_names=labels)
            plt.show()
            plot_confusion_matrix(confusion_mat_per_recording, class_names=labels)
            plt.show()

        if output["train_whole_dataset_again"]:  # Store all the information needed for the second stage model
            with open("./training_detail.csv", "a") as f1:
                print("len_train_valid", len(train_set) + len(valid_set))
                print("len_test", len(test_set))

                writer1 = csv.writer(
                    f1,
                    delimiter=",",
                    lineterminator="\n",
                )
                writer1.writerow([time.strftime("%Y-%m-%d_%H:%M:%S", time.localtime(time.time()))])

                windows_true = (
                    list(train_set.get_metadata().target)
                    + list(valid_set.get_metadata().target)
                    + list(test_set.get_metadata().target)
                )  # save labels
                len_true = len(windows_true)
                for i in range(len_true // 16384):
                    writer1.writerow(windows_true[i * 16384 : (i + 1) * 16384])
                writer1.writerow(windows_true[(len_true) // 16384 * 16384 :])

                windows_pred = np.exp(
                    np.concatenate(
                        (
                            np.array(eeg_classifier.predict_proba(train_set)[:, 1]),
                            np.array(eeg_classifier.predict_proba(valid_set)[:, 1]),
                            np.array(eeg_classifier.predict_proba(test_set)[:, 1]),
                        )
                    )
                )  # Store predicted probabilities for all data

                for i in range(len_true // 16384):
                    writer1.writerow(windows_pred[i * 16384 : (i + 1) * 16384])
                writer1.writerow(windows_pred[(len_true) // 16384 * 16384 :])

                # Store how many windows each recording has
                len_train = len(list(train_set.get_metadata().target))
                len_valid_train = len(list(valid_set.get_metadata().target)) + len_train
                writer1.writerow(
                    find_all_zero(train_set.get_metadata()["i_window_in_trial"].tolist())
                    + [x + len_train for x in find_all_zero(valid_set.get_metadata()["i_window_in_trial"].tolist())]
                    + [
                        y + len_valid_train
                        for y in find_all_zero(test_set.get_metadata()["i_window_in_trial"].tolist())
                    ]
                )

                # Store the session and patient to which each recording belongs
                paths = (
                    np.array(train_set.description.loc[:, ["path"]]).tolist()
                    + np.array(valid_set.description.loc[:, ["path"]]).tolist()
                    + np.array(test_set.description.loc[:, ["path"]]).tolist()
                )
                patients = []
                sessions = []
                for i in range(len(paths)):
                    splits = paths[i][0].split("\\")
                    patients.append(splits[-3])
                    sessions.append(splits[-2])
                print("patients", patients)
                print("sessions", sessions)
                writer1.writerow(patients)
                writer1.writerow(sessions)

        return acc

    exp(dropout=model_cfg["dropout"])
