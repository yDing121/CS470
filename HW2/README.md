# Installation

Start by installing the required packages. I am using a virtual environment on
python 3.11:

Run at root: `python -m venv venv`

Activate environment: `source venv/bin/activate` for Unix or `venv/Scripts/activate` for windows

Install requirements: `pip install -r requirements.txt`

# Running script

To run, start a terminal at the `HW2` root directory, then:
`python script.py --input <input_file> --min_support <min_support> --output <output_file> --colname <column_name> --pickle <True/False> --algorithm <algorithm_variant>`


Command line options:

`--input`: Path to the input CSV file (default: `data.csv`)

`--min_support`: Minimum absolute support count (default: `500`)

`--output`: Output file name (default: `output.txt`)

`--colname`: Name of the column in the CSV containing transactions (default: `text_keywords`)

`--pickle`: Whether to save itemsets using pickle (default: `False`)

`--algorithm`: Which Apriori variant to use (`k1km1`, `km1km1`, or `trie`) (default: `km1km1`)

These scripts have been tested on 64-bit Windows 10 with python `3.12` and 64-bit Pop!_OS Jammy with python `3.11`.
