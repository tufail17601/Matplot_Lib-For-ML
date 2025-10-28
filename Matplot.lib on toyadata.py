import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

# Step 2: Create sample student data
data = {
    'Name': ['Ali', 'Sara', 'John', 'Amina', 'Bilal', 'Zara', 'Usman'],
    'Hours_Studied': [5, 2, 8, 3, 6, 7, 1],
    'Previous_Score': [80, 60, 90, 70, 85, 88, 55],
    'Participation': [3, 1, 5, 2, 4, 5, 1],
    'Final_Grade': [85, 63, 95, 72, 92, 94, 60]
}

df = pd.DataFrame(data)

# Step 3: Data analysis using Pandas
print("Top 5 students:")
print(df.head())  

print("\n Average study Hours:",df['Hours_Studied'].mean())
print("\n Average Final Grades:",df['Final_Grade'].mean())

#USING NUMPY:
hours = np.array(df['Hours_Studied'])
grades = np.array(df['Final_Grade'])

std_hours = np.std(hours)
std_grades = np.std(grades)

print("\nStandard deviation of study Hours:",round(std_hours,2))
print("\nStandard deviation of Final grades:",round(std_grades,2))

correlation = np.corrcoef(hours, grades)[0, 1]
print("Correlation between Hours Studied and Final Grade:", round(correlation, 2))

#BAR CHART:
plt.figure(figsize=(8,4))
plt.bar(df['Name'],df['Final_Grade'],color = 'green')
plt.title('Student Final Grades')
plt.xlabel('Student Names')
plt.ylabel('Grades')
plt.grid(True)
plt.tight_layout()
plt.show()

#SCATTER PLOT:
plt.figure(figsize=(6,4))
plt.scatter(df['Hours_Studied'],df['Final_Grade'],color = 'orange',label = 'HOURS VS GRADES')
plt.title('HOURS VS GRADES')
plt.xlabel('Hours')
plt.ylabel('Grades')
plt.legend()
plt.tight_layout()
plt.grid(True)
plt.show()

#LINe ploT:
plt.figure(figsize=(7,5))
plt.plot(df['Previous_Score'],df['Final_Grade'],color = 'gray',marker = 'o',linestyle = '--')
plt.title('Previous score vs Final Grades')
plt.xlabel('Previous Score')
plt.ylabel('Grades')
plt.grid(True)
plt.tight_layout()
plt.show()

#simple prediction model:
def predict_grade(row):
    pred = (row['Previous_Score'] * 0.6) + (row['Hours_Studied'] * 4) + (row['Participation'] * 2)
    return min(100, round(pred))

df['Predicted_Grade'] = df.apply(predict_grade, axis=1)

print("\nPredicted Grades vs Actual Grades:")
print(df[['Name', 'Final_Grade', 'Predicted_Grade']])