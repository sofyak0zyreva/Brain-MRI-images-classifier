# Brain MRI images classifier

Comprehensive deep learning-based solution for the automatic classification of brain MRI images. The project is used to identify and categorize various types of brain tumors and pathological conditions. Based on Kaggle datasets [mhantor/mri-based-brain-tumor-images](https://www.kaggle.com/datasets/mhantor/mri-based-brain-tumor-images/data).

## System Requirements

python 3.8+

## Install project dependencies

```bash
git clone https://github.com/sofyak0zyreva/Brain-MRI-images-classifier.git
cd Brain-MRI-images-classifier
python3 -m venv venv
source venv/bin/activate
pip install --upgrade pip
pip install -r requirements.txt
```

## Run model

```bash
python3 model.py
```

## Single image prediction

```bash
python3 single_image_processing.py <path_to_image>
```

```bash
python3 single_image_processing.py "images/Validation/Normal/Normal (46).jpg"
```

## Test

```bash
pytest tests/ -v
```

## Docs
Docs are availible [here](https://sofyak0zyreva.github.io/Brain-MRI-images-classifier/)
Or you can generate them locally:
```bash
cd docs
make html
```

## Supported Platforms

This project supports the following platforms:

- **Linux:** Officially tested on Ubuntu 24.04 and Fedora 42.
- **Windows:** Compatible with Windows 10 and 11.
- **macOS:** Tested on macOS.
- **kvadraOS:** Tested on kvadra Operating System.

- **Architectures:** Primarily x86_64, ARM64 support.

## Devs
- [Sofya Kozyreva](https://github.com/sofyak0zyreva)
- [Ksenia Kotelnikova](https://github.com/p1onerka)
- [Aleksei Dmitrievstev](https://github.com/admitrievtsev)
- [Kostya Oreshin](https://github.com/sevenbunu)
