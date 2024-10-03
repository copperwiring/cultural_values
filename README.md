Code for cultural understanding of values in multimodal data

Code structure:
- `create_dataset/`: contains the code to create the dataset
    |- `run_create_data.py`: contains the script to create the dataset
- `download_data/`: contains the code to download the data
    |- `download_ds.sh`: contains the script to download dollar street dataset
- `cvqa_chosen/`: contains selected data from the CVQA dataset
    |- `images/`: contains the images from the CVQA dataset
    |- `metadata.csv`: contains the metadata of the images
- `notebooks/`: contains the notebooks used in the project
- `main/`: main code of the project
- `requirements.txt`: contains the dependencies of the project
- `README.md`: contains the information about the project

## Installation

1. Createa  python vitural environment:
```bash
python3 -m venv cultural-values
```

2. To install the dependencies, run the following command:
```bash
pip install -r requirements.txt
```

## Running the code

1. 



To run the code, run the following command:
```bash
python main/run_main.py
```


