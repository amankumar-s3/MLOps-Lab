# MLOps-Lab

Assignment 1 – Software Engineering with ML FlowLab
Introduction to MLOps and Development Environment Setup

## Overview
This repository documents my exploration of the MLOps ecosystem, including
setting up a local development environment, tracking experiments with MLflow,
and organizing an ML project using standard MLOps folder structure.

## Project Structure## Tech Stack
- Python 3.11+
- Git & GitHub
- Docker Desktop
- MLflow (experiment tracking)

## Setup Instructions
1. Clone this repository:
```bash
   git clone https://github.com/amankumar-s3/MLOps-Lab.git
   cd MLOps-Lab
```
2. Create and activate a virtual environment:
```bash
   python3 -m venv venv
   source venv/bin/activate
```
3. Install dependencies:
```bash
   pip install -r requirements.txt
```
4. Run MLflow UI:
```bash
   mlflow ui
```
   Then open http://localhost:5000 in your browser.

## MLflow Tracking
This project uses MLflow to track experiment parameters, metrics, and
artifacts. Run `src/train.py` to log a new run.

## Author
Aman Kumar
