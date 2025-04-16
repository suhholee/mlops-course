from metaflow import FlowSpec, step, Parameter, Flow
import pandas as pd
import numpy as np
import mlflow
import os

class MLScoringFlow(FlowSpec):
    """
    Flow for scoring new data using the best model from the training flow.
    """
    
    # Parameters
    data_path = Parameter('data_path', 
                         help='Path to the data to score',
                         default='./data/processed_test_data.csv')
    
    output_path = Parameter('output_path',
                          help='Path to save predictions',
                          default='./data/predictions.csv')
    
    training_flow_name = Parameter('training_flow_name',
                                 help='Name of the training flow to get the model from',
                                 default='MLTrainingFlow')
    
    mlflow_tracking_uri = Parameter('mlflow_tracking_uri', 
                                  help='MLFlow tracking URI',
                                  default='http://localhost:5001')
    
    @step
    def start(self):
        """Start the flow"""
        print("Starting the ML Scoring Flow")
        
        self.next(self.load_data)
    
    @step
    def load_data(self):
        """Load the data to score"""
        print(f"Loading data from {self.data_path}")
        
        # Load the data
        self.df = pd.read_csv(self.data_path)
        
        # If target is in the data, separate it
        if 'On-base Plus Slugging' in self.df.columns:
            self.X = self.df.drop('On-base Plus Slugging', axis=1)
            self.y_true = self.df['On-base Plus Slugging']
            self.has_target = True
        else:
            self.X = self.df
            self.has_target = False
        
        print(f"Loaded data with {len(self.X)} samples")
        
        self.next(self.load_model)
    
    @step
    def load_model(self):
        """Load the best model from the training flow"""
        # Get the latest run of the training flow
        run = Flow(self.training_flow_name).latest_run
        
        # Get model info from the training flow
        self.model_info = run.data.model_info
        
        print(f"Retrieved model info from training flow: {self.model_info}")
        
        # Set MLflow tracking URI
        mlflow.set_tracking_uri(self.mlflow_tracking_uri)
        
        # Load the model from MLflow
        self.model = mlflow.sklearn.load_model(
            f"models:/{self.model_info['registered_model_name']}/latest"
        )
        
        print(f"Loaded model: {self.model_info['name']}")
        
        self.next(self.score_data)
    
    @step
    def score_data(self):
        """Score the data using the loaded model"""
        # Make predictions
        self.predictions = self.model.predict(self.X)
        
        print(f"Made predictions for {len(self.predictions)} samples")
        
        # If we have true values, calculate metrics
        if self.has_target:
            self.rmse = np.sqrt(np.mean((self.predictions - self.y_true) ** 2))
            self.r2 = self.model.score(self.X, self.y_true)
            
            print(f"RMSE: {self.rmse:.4f}")
            print(f"R²: {self.r2:.4f}")
        
        self.next(self.save_predictions)
    
    @step
    def save_predictions(self):
        """Save the predictions to a file"""
        # Create a DataFrame with IDs and predictions
        results_df = self.X.copy()
        results_df['predicted_OPS'] = self.predictions
        
        if self.has_target:
            results_df['actual_OPS'] = self.y_true
            results_df['error'] = results_df['actual_OPS'] - results_df['predicted_OPS']
        
        # Create the output directory if it doesn't exist
        os.makedirs(os.path.dirname(self.output_path), exist_ok=True)
        
        # Save to CSV
        results_df.to_csv(self.output_path, index=False)
        
        print(f"Saved predictions to {self.output_path}")
        
        self.next(self.end)
    
    @step
    def end(self):
        """End the flow"""
        print("ML Scoring Flow completed successfully")
        print(f"Predictions saved to {self.output_path}")
        
        if self.has_target:
            print(f"Final metrics - RMSE: {self.rmse:.4f}, R²: {self.r2:.4f}")
        
if __name__ == '__main__':
    MLScoringFlow()