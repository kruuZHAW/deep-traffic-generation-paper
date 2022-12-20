# flake8: noqa
from argparse import ArgumentParser

import pytorch_lightning as pl
from pytorch_lightning import LightningModule, Trainer
from pytorch_lightning.callbacks import EarlyStopping, ModelCheckpoint
from pytorch_lightning.core.lightning import LightningModule
from pytorch_lightning.loggers import TensorBoardLogger
from sklearn.preprocessing import MinMaxScaler

from .abstract import AE, VAE
from .datasets import TrafficDataset
from .lsr import VampPriorLSR, NormalLSR, ExemplarLSR
from .networks import FCN, RNN, TCN
from .utils import get_dataloaders


def cli_main(
    cls: LightningModule,
    dataset_cls: TrafficDataset,
    data_shape: str,
    seed: int = 42,
) -> None:
    pl.seed_everything(seed, workers=True)
    # ------------
    # args
    # ------------
    parser = ArgumentParser()
    parser.add_argument(
        "--train_ratio", dest="train_ratio", type=float, default=0.8
    )
    parser.add_argument(
        "--val_ratio", dest="val_ratio", type=float, default=0.2
    )
    parser.add_argument(
        "--batch_size", dest="batch_size", type=int, default=1000
    )
    parser.add_argument(
        "--test_batch_size",
        dest="test_batch_size",
        type=int,
        default=None,
    )
    parser.add_argument(
        "--early_stop", dest="early_stop", type=int, default=None
    )
    parser = dataset_cls.add_argparse_args(parser)
    parser = Trainer.add_argparse_args(parser)
    parser, _ = cls.add_model_specific_args(parser)
    args = parser.parse_args()

    # ------------
    # data
    # ------------
    dataset = dataset_cls.from_file(
        args.data_path,
        features=args.features,
        shape=data_shape,
        scaler=MinMaxScaler(feature_range=(-1, 1)),
        info_params={"features": args.info_features, "index": args.info_index},
    )

    train_loader, val_loader, test_loader = get_dataloaders(
        dataset,
        args.train_ratio,
        args.val_ratio,
        args.batch_size,
        args.test_batch_size,
    )

    # ------------
    # logger
    # ------------
    tb_logger = TensorBoardLogger(
        "lightning_logs/",
        name=args.network_name,
        default_hp_metric=False,
        log_graph=True,
    )

    # ------------
    # model
    # ------------
    model = cls(
        dataset_params=dataset.parameters,
        config=args,
    )

    # ------------
    # training
    # ------------
    checkpoint_callback = ModelCheckpoint(monitor="hp/valid_loss")
    # checkpoint_callback = ModelCheckpoint()
    if args.early_stop is not None:
        early_stopping = EarlyStopping(
            "hp/valid_loss", patience=args.early_stop
        )
        trainer = Trainer.from_argparse_args(
            args,
            callbacks=[checkpoint_callback, early_stopping],
            logger=tb_logger,
            # deterministic=True,
        )
    else:
        trainer = Trainer.from_argparse_args(
            args,
            callbacks=[checkpoint_callback],
            logger=tb_logger,
            # deterministic=True,
        )

    if val_loader is not None:
        trainer.fit(model, train_loader, val_loader)
    else:
        trainer.fit(model, train_loader)

    # ------------
    # testing
    # ------------
    if test_loader is not None:
        trainer.test(test_dataloaders=test_loader)
