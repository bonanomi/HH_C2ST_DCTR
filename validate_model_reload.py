import argparse
import numpy as np
import tensorflow as tf

import c2st_config as cfg
from c2st_artifacts import load_scaler, load_test_fold, load_stage_arrays, model_path
from c2st_core import apply_scaler


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--channel', choices=cfg.CHANNELS, default='2mu')
    ap.add_argument('--stage', choices=cfg.STAGES, default='after')
    ap.add_argument('--rows', type=int, default=200000,
                    help='Rows to re-predict for serialization check; 0 means full test fold')
    args = ap.parse_args()
    print('TensorFlow GPUs:', tf.config.list_physical_devices('GPU'))

    test = load_test_fold(cfg.ARTIFACT_DIR, args.channel, columns=cfg.FEATURES)
    n = len(test) if args.rows <= 0 else min(args.rows, len(test))
    test = test.iloc[:n]
    scaler = load_scaler(cfg.ARTIFACT_DIR, args.channel)
    x = apply_scaler(test, cfg.FEATURES, scaler)
    model = tf.keras.models.load_model(model_path(cfg.ARTIFACT_DIR, args.channel, args.stage))
    p = model.predict(x, batch_size=cfg.BATCH_SIZE, verbose=0).reshape(-1)
    saved = np.asarray(load_stage_arrays(cfg.ARTIFACT_DIR, args.channel, args.stage)['p_test'])[:n]
    diff = np.abs(p - saved)
    print(f'{args.channel}/{args.stage}: rows={n:_}, max|delta p|={diff.max():.3e}, mean|delta p|={diff.mean():.3e}')

if __name__ == '__main__':
    main()
