# # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # #
# Copyright 2022 The Hume AI Authors. All Rights Reserved.
# Code available under a Creative Commons Attribution-Non Commercial-No Derivatives 4.0
# International Licence (CC BY-NC-ND) license.
# # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # #

import numpy as np
import pandas as pd
from sklearn.metrics import recall_score, mean_squared_error, mean_absolute_error
import torch
from models import MLPReg, MLPClass


class EvalMetrics:
    def CCC(y_true, y_pred):
        x_mean = np.nanmean(y_true, dtype="float32")
        y_mean = np.nanmean(y_pred, dtype="float32")
        x_var = 1.0 / (len(y_true) - 1) * np.nansum((y_true - x_mean) ** 2)
        y_var = 1.0 / (len(y_pred) - 1) * np.nansum((y_pred - y_mean) ** 2)
        cov = np.nanmean((y_true - x_mean) * (y_pred - y_mean))
        return round((2 * cov) / (x_var + y_var + (x_mean - y_mean) ** 2), 4)

    def CCCL(y_true, y_pred):
        return 1 - CCC(y_true, y_pred)

    def MAE(y_true, y_pred):
        return mean_absolute_error(y_true, y_pred)

    def MSE(y_true, y_pred):
        return mean_squared_error(y_true, y_pred)

    def UAR(y_true, y_pred):
        return recall_score(y_true, y_pred, average="macro")

    def CCCLoss(x, y):
        xy = torch.cat((x, y))
        ccc = 2 * torch.cov(xy) / (x.var() + y.var() + (x.mean() - y.mean())**2)
        return 1 - ccc


class CCCLoss:
    def __init(self):
        super(CCCLoss, self).__init__()

    def forward(self, x, y):
        xy = torch.cat((x, y))
        ccc = 2*torch.cov(xy) / (x.var() + y.var() + (x.mean() - y.mean())**2)
        return 1 - ccc


class Processing:
    def normalise(scaler, X, y, task):
        train_X = scaler.fit_transform(X[0])
        train_X = pd.DataFrame(train_X).values
        if task != "type":
            train_y, val_y, test_y = (
                y[0].astype(float),
                y[1].astype(float),
                y[2].astype(float),
            )
        else:
            train_y, val_y, test_y = (y[0], y[1], y[2])

        val_X = scaler.transform(X[1])
        test_X = scaler.transform(X[2])
        test_X = pd.DataFrame(test_X).values
        return (
            [train_X, val_X, test_X],
            [train_y, val_y, test_y],
        )


class EarlyStopping:
    # Implementation adapted from
    # https://github.com/Bjarten/early-stopping-pytorch
    def __init__(
        self,
        patience=7,
        verbose=False,
        delta=0.1,
        trace_func=print,
    ):
        self.patience = patience
        self.verbose = verbose
        self.counter = 0
        self.best_score = None
        self.early_stop = False
        self.val_loss_min = np.Inf
        self.delta = delta
        self.trace_func = trace_func

    def __call__(self, val_loss, model):
        score = -val_loss
        if self.best_score is None:
            self.best_score = score
        elif score < self.best_score + self.delta:
            self.counter += 1
            if self.counter >= self.patience:
                self.early_stop = True
        else:
            self.best_score = score
            self.counter = 0


class StorePredictions:
    def storehigh(
        task,
        feat_dimensions,
        classes,
        dev,
        timestamp,
        store_name,
        seed_score,
        X,
        test_file_ids,
    ):
        model = MLPReg(feat_dimensions, len(classes)).to(dev)
        model.load_state_dict(
            torch.load(f"tmp/{timestamp}_{store_name}_model_{seed_score}_{task}.pth")
        )
        test_pred = model(torch.from_numpy(X[2].astype(np.float32)).to(dev))
        test_dict_info = {
            "File_ID": list(test_file_ids.values),
            "Awe": np.array(test_pred[:, 0].cpu().detach().numpy()),
            "Excitement": np.array(test_pred[:, 1].cpu().detach().numpy()),
            "Amusement": np.array(test_pred[:, 2].cpu().detach().numpy()),
            "Awkwardness": np.array(test_pred[:, 3].cpu().detach().numpy()),
            "Fear": np.array(test_pred[:, 4].cpu().detach().numpy()),
            "Horror": np.array(test_pred[:, 5].cpu().detach().numpy()),
            "Distress": np.array(test_pred[:, 6].cpu().detach().numpy()),
            "Triumph": np.array(test_pred[:, 7].cpu().detach().numpy()),
            "Sadness": np.array(test_pred[:, 8].cpu().detach().numpy()),
            "Surprise": np.array(test_pred[:, 9].cpu().detach().numpy()),
        }

        test_prediction_csv = pd.DataFrame.from_dict(test_dict_info).sort_values(
            by="File_ID"
        )
        test_prediction_csv.to_csv(
            f"preds/Test_A-VB_{timestamp}_{task}_{seed_score}_{store_name}.csv",
            index=False,
        )

    def storetwo(
        task,
        feat_dimensions,
        classes,
        dev,
        timestamp,
        store_name,
        seed_score,
        X,
        test_file_ids,
    ):
        model = MLPReg(feat_dimensions, len(classes)).to(dev)
        model.load_state_dict(
            torch.load(f"tmp/{timestamp}_{store_name}_model_{seed_score}_{task}.pth")
        )
        test_pred = model(torch.from_numpy(X[2].astype(np.float32)).to(dev))
        test_dict_info = {
            "File_ID": list(test_file_ids.values),
            "Valence": np.array(test_pred[:, 0].cpu().detach().numpy()),
            "Arousal": np.array(test_pred[:, 1].cpu().detach().numpy()),
        }

        test_prediction_csv = pd.DataFrame.from_dict(test_dict_info).sort_values(
            by="File_ID"
        )
        test_prediction_csv.to_csv(
            f"preds/Test_A-VB_{timestamp}_{task}_{seed_score}_{store_name}.csv",
            index=False,
        )

    def storeculture(
        task,
        feat_dimensions,
        classes,
        dev,
        timestamp,
        store_name,
        seed_score,
        X,
        test_file_ids,
    ):
        model = MLPReg(feat_dimensions, len(classes)).to(dev)
        model.load_state_dict(
            torch.load(f"tmp/{timestamp}_{store_name}_model_{seed_score}_{task}.pth")
        )
        test_pred = model(torch.from_numpy(X[2].astype(np.float32)).to(dev))

        test_dict_info = {
            "File_ID": list(test_file_ids.values),
            "China_Awe": np.array(test_pred[:, 0].cpu().detach().numpy()),
            "China_Excitement": np.array(test_pred[:, 1].cpu().detach().numpy()),
            "China_Amusement": np.array(test_pred[:, 2].cpu().detach().numpy()),
            "China_Awkwardness": np.array(test_pred[:, 3].cpu().detach().numpy()),
            "China_Fear": np.array(test_pred[:, 4].cpu().detach().numpy()),
            "China_Horror": np.array(test_pred[:, 5].cpu().detach().numpy()),
            "China_Distress": np.array(test_pred[:, 6].cpu().detach().numpy()),
            "China_Triumph": np.array(test_pred[:, 7].cpu().detach().numpy()),
            "China_Sadness": np.array(test_pred[:, 8].cpu().detach().numpy()),
            "United States_Awe": np.array(test_pred[:, 9].cpu().detach().numpy()),
            "United States_Excitement": np.array(test_pred[:, 10].cpu().detach().numpy()),
            "United States_Amusement": np.array(test_pred[:, 11].cpu().detach().numpy()),
            "United States_Awkwardness": np.array(test_pred[:, 12].cpu().detach().numpy()),
            "United States_Fear": np.array(test_pred[:, 13].cpu().detach().numpy()),
            "United States_Horror": np.array(test_pred[:, 14].cpu().detach().numpy()),
            "United States_Distress": np.array(test_pred[:, 15].cpu().detach().numpy()),
            "United States_Triumph": np.array(test_pred[:, 16].cpu().detach().numpy()),
            "United States_Sadness": np.array(test_pred[:, 17].cpu().detach().numpy()),
            "South Africa_Awe": np.array(test_pred[:, 18].cpu().detach().numpy()),
            "South Africa_Excitement": np.array(test_pred[:, 19].cpu().detach().numpy()),
            "South Africa_Amusement": np.array(test_pred[:, 20].cpu().detach().numpy()),
            "South Africa_Awkwardness": np.array(test_pred[:, 21].cpu().detach().numpy()),
            "South Africa_Fear": np.array(test_pred[:, 22].cpu().detach().numpy()),
            "South Africa_Horror": np.array(test_pred[:, 23].cpu().detach().numpy()),
            "South Africa_Distress": np.array(test_pred[:, 24].cpu().detach().numpy()),
            "South Africa_Triumph": np.array(test_pred[:, 25].cpu().detach().numpy()),
            "South Africa_Sadness": np.array(test_pred[:, 26].cpu().detach().numpy()),
            "Venezuela_Awe": np.array(test_pred[:, 27].cpu().detach().numpy()),
            "Venezuela_Excitement": np.array(test_pred[:, 28].cpu().detach().numpy()),
            "Venezuela_Amusement": np.array(test_pred[:, 29].cpu().detach().numpy()),
            "Venezuela_Awkwardness": np.array(test_pred[:, 30].cpu().detach().numpy()),
            "Venezuela_Fear": np.array(test_pred[:, 31].cpu().detach().numpy()),
            "Venezuela_Horror": np.array(test_pred[:, 32].cpu().detach().numpy()),
            "Venezuela_Distress": np.array(test_pred[:, 33].cpu().detach().numpy()),
            "Venezuela_Triumph": np.array(test_pred[:, 34].cpu().detach().numpy()),
            "Venezuela_Sadness": np.array(test_pred[:, 35].cpu().detach().numpy()),
            "China_Surprise": np.array(test_pred[:, 36].cpu().detach().numpy()),
            "United States_Surprise": np.array(test_pred[:, 37].cpu().detach().numpy()),
            "South Africa_Surprise": np.array(test_pred[:, 38].cpu().detach().numpy()),
            "Venezuela_Surprise": np.array(test_pred[:, 39].cpu().detach().numpy()),
        }

        test_prediction_csv = pd.DataFrame.from_dict(test_dict_info).sort_values(
            by="File_ID"
        )
        test_prediction_csv.to_csv(
            f"preds/Test_A-VB_{timestamp}_{task}_{seed_score}_{store_name}.csv",
            index=False,
        )

    def storetype(
        task,
        feat_dimensions,
        classes,
        dev,
        timestamp,
        store_name,
        seed_score,
        X,
        test_file_ids,
        le,
    ):
        model = MLPClass(feat_dimensions, len(classes)).to(dev)
        model.load_state_dict(
            torch.load(f"tmp/{timestamp}_{store_name}_model_{seed_score}_{task}.pth")
        )
        test_pred = model(torch.from_numpy(X[2].astype(np.float32)).to(dev))

        t_pred = torch.max(test_pred, 1)
        t_pred = le.inverse_transform(t_pred.indices.cpu())
        test_dict_info = {
            "File_ID": list(test_file_ids.values),
            "Voc_Type": t_pred,
        }
        test_prediction_csv = pd.DataFrame.from_dict(test_dict_info).sort_values(
            by="File_ID"
        )
        test_prediction_csv.to_csv(
            f"preds/Test_A-VB_{timestamp}_{task}_{seed_score}_{store_name}.csv",
            index=False,
        )
