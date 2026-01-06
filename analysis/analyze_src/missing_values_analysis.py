from abc import ABC, abstractmethod
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

class MissingValuesAnalysisTemplate(ABC):
    def analysis(self, df: pd.DataFrame):
        self.identify_missing_value(df)
        self.visualize_missing_value(df)
        
    @abstractmethod
    def identify_missing_value(self, df: pd.DataFrame):
        pass
    @abstractmethod
    def visualize_missing_value(self, df: pd.DataFrame):
        pass
    
class MissingValueAnalysis(MissingValuesAnalysisTemplate):
    def identify_missing_value(self, df: pd.DataFrame):
        print("\nMissing values")
        missing_value = df.isnull().sum()
        print(missing_value[missing_value > 0])
        
    def visualize_missing_value(self, df: pd.DataFrame):
        print("\nVisualizing of missing value..")
        plt.figure(figsize=(12, 8))
        sns.heatmap(df.isnull(), cbar = False, cmap = "viridis")
        plt.title("Missing value heatmap")
        plt.show()
        
        
if __name__ == "__main__":
     # Example usage of the SimpleMissingValuesAnalysis class.

    # Load the data
    # df = pd.read_csv('../extracted-data/your_data_file.csv')

    # Perform Missing Values Analysis
    # missing_values_analyzer = MissingValueAnalysis()
    # missing_values_analyzer.analyze(df)
    pass