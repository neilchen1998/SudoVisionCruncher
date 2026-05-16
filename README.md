# SudoVisionCruncher

## Summary

## Requirements

* [uv](https://docs.astral.sh/uv/)
* Python 3.11.15
* [Tensorflow](https://www.tensorflow.org/) 2.15.1
* [Metal](https://developer.apple.com/metal/tensorflow-plugin/) 1.1.0 (for Mac Silicon)

## Instructions

Download Python with **uv**:

```zsh
uv venv --python 3.11.15
```

Activate the environment:

```zsh
source .venv/bin/activate
```

Install the packages:

```zsh
uv pip install tensorflow
```

Install the packages via *requirements.txt*:

```zsh
uv pip install -r requirements.txt
```

Deactivate the environment:

```zsh
deactivate
```

Run the main script:

```zsh
uv run ./main.py
```
