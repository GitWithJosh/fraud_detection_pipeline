import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import confusion_matrix, accuracy_score, precision_score, recall_score, f1_score, roc_auc_score
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler

# https://www.kaggle.com/datasets/nelgiriyewithana/credit-card-fraud-detection-dataset-2023/data?select=creditcard_2023.csv

class CreditCardFraudDetector:
    def __init__(self, data_path):
        self.data_path = data_path
        self.model = None
        self.X_train = None
        self.X_test = None
        self.y_train = None
        self.y_test = None
        self.scaler = StandardScaler()
        
    def load_data(self):
        print("Loading data...")
        data = pd.read_csv(self.data_path)
        self.X = data.drop('Class', axis=1)
        self.y = data['Class']
        print("Data loaded.")
        
    def split_data(self, test_size=0.2, random_state=42):
        print("Splitting data...")
        self.X_train, self.X_test, self.y_train, self.y_test = train_test_split(self.X, self.y, test_size=test_size, random_state=random_state)
        print("Data splitted.")
        
    def preprocess_data(self):
        print("Preprocessing data...")
        self.X_train = self.scaler.fit_transform(self.X_train)
        self.X_test = self.scaler.transform(self.X_test)
        print("Data preprocessed.")
        
    def train_model(self, n_estimators=50, random_state=42):
        print("Training model...")
        self.model = RandomForestClassifier(n_estimators=n_estimators, random_state=random_state)
        self.model.fit(self.X_train, self.y_train)
        print("Model trained.")

        
    def evaluate_model(self):
        print("Evaluating model...")
        y_pred = self.model.predict(self.X_test)
        accuracy = accuracy_score(self.y_test, y_pred)
        precision = precision_score(self.y_test, y_pred)
        recall = recall_score(self.y_test, y_pred)
        f1 = f1_score(self.y_test, y_pred)
        roc_auc = roc_auc_score(self.y_test, y_pred)

        print("Model evaluation results:")
        print("Accuracy:", accuracy)
        print("Precision:", precision)
        print("Recall:", recall)
        print("F1 Score:", f1)
        print("ROC AUC Score:", roc_auc)

        
    def visualize_data(self, sample_size=1000):
        fraud_indices = self.y[self.y == 1].index
        non_fraud_indices = self.y[self.y == 0].index
        
        fraud_sample = self.X.loc[fraud_indices].sample(sample_size, random_state=42)
        non_fraud_sample = self.X.loc[non_fraud_indices].sample(sample_size, random_state=42)
        
        plt.figure(figsize=(12, 6))
        plt.subplot(1, 2, 1)
        plt.scatter(non_fraud_sample['V1'], non_fraud_sample['V2'], label='Non-Fraud', alpha=0.5)
        plt.scatter(fraud_sample['V1'], fraud_sample['V2'], label='Fraud', alpha=0.5)
        plt.xlabel('V1')
        plt.ylabel('V2')
        plt.title('Scatter plot of V1 vs V2')
        plt.legend()
        
        plt.subplot(1, 2, 2)
        sns.histplot(data=self.y)
        plt.title('Class Distribution')
        plt.xlabel('Class')
        plt.ylabel('Count')
        
        plt.show()
        
    def visualize_confusion_matrix(self):
        print("Visualizing confusion matrix...")
        y_pred = self.model.predict(self.X_test)
        cm = confusion_matrix(self.y_test, y_pred)
        
        plt.figure(figsize=(8, 6))
        sns.heatmap(cm, annot=True, fmt='d', cmap='Blues')
        plt.title('Confusion Matrix')
        plt.xlabel('Predicted Label')
        plt.ylabel('True Label')
        plt.show()
        print("Confusion matrix visualized.")

if __name__ == '__main__':
    # Example usage
    detector = CreditCardFraudDetector('./creditcard_2023.csv')
    detector.load_data()
    detector.split_data()
    detector.preprocess_data()
    detector.train_model()
    detector.evaluate_model()
    detector.visualize_confusion_matrix()
