# Student Performance Prediction System

A machine learning project that predicts students' final grades using academic performance data and a Random Forest Regression model.

## Project Overview

This application uses students' study time, past failures, absences, and previous grades to predict their final grade (G3).

It includes a Streamlit web interface where users can enter student information and receive predictions.

## Technologies Used

* Python
* Pandas
* NumPy
* Scikit-learn
* Matplotlib
* Seaborn
* Streamlit

## Machine Learning Model

**Algorithm:** Random Forest Regression

**Input Features:**

* Study time
* Number of past failures
* Number of absences
* First period grade (G1)
* Second period grade (G2)

**Target:** Final grade (G3)

## Model Evaluation

| Metric                    | Result |
| ------------------------- | -----: |
| Mean Absolute Error (MAE) |  1.053 |
| Mean Squared Error (MSE)  |  2.614 |
| R² Score                  |  0.873 |

These results are from one 80/20 train-test split. They are not a guarantee of performance on other student populations.

## Dataset

The project uses the Student Performance dataset from the UCI Machine Learning Repository.

The dataset contains records of students from Portuguese secondary schools. Predictions should not be assumed to generalize to other schools or university students without further validation.

## How to Run

### 1. Clone the repository

```bash
git clone YOUR_GITHUB_REPOSITORY_URL
cd StudentPerformanceML
```

### 2. Create and activate a virtual environment

```bash
py -3.11 -m venv venv
```

On Windows:

```bash
venv\Scripts\activate
```

### 3. Install dependencies

```bash
pip install -r requirements.txt
```

### 4. Run the application

```bash
streamlit run app.py
```

The application will open in your browser.

## Project Structure

```text
StudentPerformanceML/
├── data/
│   └── student-mat.csv
├── analysis.py
├── app.py
├── student_performance_model.pkl
├── actual_vs_predicted.png
├── requirements.txt
└── README.md
```

## Future Improvements

* Add more relevant student features.
* Compare additional machine learning algorithms.
* Improve model validation.
* Deploy the application online.
* Add visualizations and more detailed prediction insights.

## Disclaimer

This application is an educational machine learning project. Its predictions should not be used as the sole basis for academic decisions.
