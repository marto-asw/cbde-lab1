# Lab1 

This project contains Python scripts that need to be executed using. Before running the scripts, all required dependencies must be installed. 

## Structure

The repository is organized as follows: 

	•	P0.py – Connects to PostgreSQL and stores raw sentences
	•	P1.py – Connects to PostgreSQL and stores embeddings
	•	P2.py – PostgreSQL benchmark analysis
	•	C0.py – Connects to ChromaDB and stores raw sentences
	•	C1.py – Connects to ChromaDB and stores embeddings
	•	C2.py – ChromaDB benchmark analysis
	•	README.md – This file
	•	requirements.txt – Python dependencies
	•	database.ini – Not included in the repository; contains PostgreSQL credentials
  • config.py and connect.py - Configuration files to connect to the db
 
## create a python enviroment

python3 -m venv env

## Install dependencies in the enviroment

source env/bin/activate

pip install -r requirements.txt

## Database Setup

To connect to your PostgreSQL database, create a file named database.ini in the project root with the following structure:

[postgresql]
host=your_host
port=your_port
database=your_database
user=your_username
password=your_password

## Run scripts
Python files can be run as follows: 

python file_name.py   
