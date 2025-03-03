# Installation

Start by installing the required packages. I am using a virtual environment on
python 3.11:

Run at root: `python -m venv venv`

Activate environment: `source venv/bin/activate`

Install requirements: `pip install -r requirements.txt`

# Running script

Command line options:

`--input` - Input file path relative to the HW2 root

`--min_support` - Minimum absolute support

`--colname` - Name of the column from which frequent itemsets will be mined.
Defaults to `"text_keywords"`

`--output` - Output file name, to be written to the HW2 root.
