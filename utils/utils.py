import json
from pathlib import Path

def read_json(file_path):
    with open(file_path, 'r') as f:
        json_data = json.load(f)
    return json_data


def prefix_zero(n, total_length):
    return '0' * (total_length - len(str(n))) + str(n)


def get_latest_version(model_dir, model_name):
    latest_version, latest_version_path = (-1, "")
    model_dir = Path(model_dir)
    model_provider = model_name.split("/")[0] # "klue"/roberta-large
    model_title = "-".join(model_name.split("/")[1].split()) # klue/"roberta-large"
    # print(model_title)
    for child in model_dir.iterdir():
        if child.is_dir() and child.name == model_provider:
            model_files = list(child.glob(f"{model_title}_*.ckpt"))
            # print(model_files)
            if len(model_files) > 0:
                model_versions = [(int(model_file.stem.split("_")[-9][-2:]), model_file) for model_file in model_files]
                latest_version, latest_version_path = sorted(model_versions, key=lambda x: x[0], reverse=True)[0]
                break
    return latest_version, latest_version_path


def get_version(model_dir, model_name, best=False):
    version, version_path = (-1, "")
    model_dir = Path(model_dir)
    model_provider = model_name.split("/")[0] # "klue"/roberta-large
    model_title = "-".join(model_name.split("/")[1].split()) # klue/"roberta-large"
    # print(model_title)
    for child in model_dir.iterdir():
        if child.is_dir() and child.name == model_provider:
            model_files = list(child.glob(f"{model_title}_*.ckpt"))
            # print(model_files)
            if len(model_files) > 0:
                model_versions = [(int(model_file.stem.split("_")[-9][-2:]), float(model_file.stem.split("_")[-3]), model_file) for model_file in model_files]
                if best:
                    # the best performance version
                    func = lambda x: x[1]
                else: 
                    # latest version
                    func = lambda x: x[0]
                version, version_perf, version_path = sorted(model_versions, key=func, reverse=True)[0]
                break
    return version, version_perf, version_path


def format_pearson(pearson_value):
    # Scale and convert to integer
    return str(int(pearson_value * 1000))

def float_only(n):
    return str(n).split(".")[1]