import pandas as pd
import numpy as np

class DataQualityAnalyzer:
    
    
    def __init__(self, file_path):
      
        self.file_path = file_path
        try:
            self.df = pd.read_csv(file_path)
            self.cleaned_df = self.df.copy()
            print(f"Successfully loaded '{file_path}' with {self.df.shape[0]} rows and {self.df.shape[1]} columns.\n")
        except Exception as e:
            print(f"Error loading file: {e}")
            self.df = None
            
        self.suggestions = []
        self.report = {}

    def analyze_data(self):
       
        if self.df is None:
            return
        
     
        missing = self.df.isnull().sum()
        missing_pct = (missing / len(self.df)) * 100
        self.report['missing'] = pd.DataFrame({'Missing_Count': missing, 'Missing_Percent': missing_pct})
        self.report['missing'] = self.report['missing'][self.report['missing']['Missing_Count'] > 0]
        
       
        self.report['duplicates'] = self.df.duplicated().sum()
        
        types_and_ranges = []
        for col in self.df.columns:
            col_info = {'Column': col, 'Type': self.df[col].dtype}
            if pd.api.types.is_numeric_dtype(self.df[col]):
                col_info['Min'] = self.df[col].min()
                col_info['Max'] = self.df[col].max()
            else:
                col_info['Min'] = 'N/A'
                col_info['Max'] = 'N/A'
            types_and_ranges.append(col_info)
        self.report['types_ranges'] = pd.DataFrame(types_and_ranges)
        
       
        outliers_dict = {}
        for col in self.df.select_dtypes(include=[np.number]).columns:
            Q1 = self.df[col].quantile(0.25)
            Q3 = self.df[col].quantile(0.75)
            IQR = Q3 - Q1
            lower_bound = Q1 - 1.5 * IQR
            upper_bound = Q3 + 1.5 * IQR
            
            
            outliers_count = self.df[(self.df[col] < lower_bound) | (self.df[col] > upper_bound)].shape[0]
            if outliers_count > 0:
                outliers_dict[col] = outliers_count
        self.report['outliers'] = outliers_dict

    def generate_suggestions(self):
        
        print("Suggested Cleaning Steps ")
        step_num = 1
        
       
        if self.report['duplicates'] > 0:
            print(f"{step_num}. Remove {self.report['duplicates']} duplicate row(s).")
            self.suggestions.append('remove_duplicates')
            step_num += 1
            
        
        for col, row in self.report['missing'].iterrows():
            if row['Missing_Percent'] > 50:
                print(f"{step_num}. Drop column '{col}' (has >50% missing values: {row['Missing_Percent']:.1f}%).")
                self.suggestions.append(('drop_column', col))
            else:
                print(f"{step_num}. Fill missing values in '{col}' with median/mode (has {row['Missing_Percent']:.1f}% missing).")
                self.suggestions.append(('fill_missing', col))
            step_num += 1

        if not self.suggestions:
            print("Data looks clean! No suggestions needed.")
        print("\n")

    def simulate_cleaning(self):
        
        for action in self.suggestions:
            if action == 'remove_duplicates':
                self.cleaned_df = self.cleaned_df.drop_duplicates()
            elif isinstance(action, tuple):
                task, col = action
                if task == 'drop_column':
                    self.cleaned_df = self.cleaned_df.drop(columns=[col])
                elif task == 'fill_missing':
                    if pd.api.types.is_numeric_dtype(self.cleaned_df[col]):
                        self.cleaned_df[col] = self.cleaned_df[col].fillna(self.cleaned_df[col].median())
                    else:
                        self.cleaned_df[col] = self.cleaned_df[col].fillna(self.cleaned_df[col].mode()[0])

    def print_comparison_report(self):
      
        print(" DATA QUALITY REPORT: BEFORE VS AFTER ")
        
       
        missing_before = self.df.isnull().any(axis=1).sum() 
        missing_after = self.cleaned_df.isnull().any(axis=1).sum()
        print(f"Rows with missing values: {missing_before} before -> {missing_after} after")
        
       
        dup_before = self.report['duplicates']
        dup_after = self.cleaned_df.duplicated().sum()
        print(f"Duplicate rows:           {dup_before} before -> {dup_after} after")
        
       
        print("\nOutliers detected (Before cleaning):")
        for col, count in self.report['outliers'].items():
            print(f" - '{col}': {count} outlier(s)")
            
        print("\nData Types & Ranges (Before cleaning):")
        print(self.report['types_ranges'].to_string(index=False))
      

import csv
with open('sample_data.csv', 'w', newline='') as f:
    writer = csv.writer(f)
    writer.writerow(['ID', 'Age', 'Salary', 'Department'])
    writer.writerow(['1', '25', '50000', 'IT'])
    writer.writerow(['2', '30', '60000', 'HR'])
    writer.writerow(['2', '30', '60000', 'HR']) 
    writer.writerow(['3', '', '55000', 'IT'])   
    writer.writerow(['4', '45', '150000', ''])  
    writer.writerow(['5', '22', '', 'Sales'])  

if __name__ == "__main__":
   
    analyzer = DataQualityAnalyzer("sample_data.csv")
    
   
    analyzer.analyze_data()
    analyzer.generate_suggestions()
    analyzer.simulate_cleaning()
    analyzer.print_comparison_report()


    