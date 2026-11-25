"""Opt-in cache of selected physical tables. Only use your own trusted cache."""
from pathlib import Path
import hashlib
import json
import os
import tempfile
import time
import joblib
from .config import jsonable, process_entries
from .loading import branch_files, producer_path


def input_manifest(config, year):
    yc = config['years'][year]
    source = yc['source']
    paths = set()
    for side in ('data', 'mc'):
        entries = process_entries(yc[f'{side}_processes'])
        for entry in entries.values():
            if entry.get('source', year) != year:
                raise ValueError('Plot cache expects the configured per-year source')
            spec = yc[f'{side}_weights']
            producers = {source.get('feature_producer', 'dl_ml_inputs'), 'reduction'}
            fields = list(spec.get('fields', []))
            if side == 'mc':
                for weight in config.get('mc_weight_variations', {}).values():
                    fields.extend(weight.get('fields', []))
            if side == 'mc' and entry.get('is_dy') and yc.get('dy_weight'):
                fields.append(yc['dy_weight'])
            for field in fields:
                producers.add(field.get('producer', 'event_weights') if isinstance(field, dict) else 'event_weights')
            directories = [producer_path(source, p) for p in sorted(producers)] if source.get('format', 'cf') == 'cf' else []
            for dataset in entry['datasets']:
                if source.get('format', 'cf') == 'cf':
                    for directory in directories:
                        if directory is not None:
                            paths.update(branch_files(directory, dataset, source.get('shift', 'nominal')).values())
                else:
                    paths.update(Path(source['root']).glob(dataset))
    result = []
    for path in sorted(paths):
        stat = path.stat()
        result.append((str(path.resolve()), stat.st_size, stat.st_mtime_ns))
    return result


def cached_population(config, year, channel, region, features, loader, cache_dir=None, refresh=False):
    start = time.monotonic()
    if cache_dir is None:
        result = loader(config, year, channel, region, features)
        print(f'Loaded {year}/{channel}/{region} in {time.monotonic()-start:.1f}s (cache disabled)', flush=True)
        return result
    payload = dict(version=1, year=year, channel=channel, region=region, features=list(features),
        year_config=config['years'][year], channels=config['channels'],
        channel_ids=config['channel_ids'], region_ids=config['region_ids'],
        selections=config.get('selections', {}), mc_weight_variations=config.get('mc_weight_variations', {}),
        inputs=input_manifest(config, year),
        code={name: hashlib.sha256(Path(__file__).with_name(name).read_bytes()).hexdigest()
              for name in ('loading.py', 'plot_year_transfer.py', 'plot_cache.py')})
    key = hashlib.sha256(json.dumps(jsonable(payload), sort_keys=True).encode()).hexdigest()
    folder = Path(cache_dir)
    folder.mkdir(parents=True, exist_ok=True)
    path = folder / (key + '.joblib')
    if path.exists() and not refresh:
        result = joblib.load(path)
        print(f'Cache HIT {year}/{channel}/{region}: {time.monotonic()-start:.1f}s', flush=True)
        return result
    print(f'Cache MISS {year}/{channel}/{region}; reading input parquet files', flush=True)
    result = loader(config, year, channel, region, features)
    if input_manifest(config, year) != payload['inputs']:
        raise RuntimeError('Input files changed during loading; cache not written. Rerun with stable inputs.')
    fd, temporary = tempfile.mkstemp(prefix=key + '.', suffix='.tmp', dir=folder)
    os.close(fd)
    try:
        joblib.dump(result, temporary, compress=0)
        os.replace(temporary, path)
    finally:
        Path(temporary).unlink(missing_ok=True)
    print(f'Cache saved {year}/{channel}/{region}: {time.monotonic()-start:.1f}s', flush=True)
    return result

