from metaflow import FlowSpec, step, Parameter, Flow, kubernetes, conda_base, retry, timeout, catch

# @conda_base(libraries={
#     'pandas': '2.2.2',
#     'numpy': '1.26.4',
#     'mlflow': '2.5.1'
# }, python='3.9.16')

class MLScoringFlowGCP(FlowSpec):
    """
    Flow for scoring new data using the best model from the training flow.
    Running in GCP Kubernetes.
    """
    
    # Parameters
    data_path = Parameter('data_path', 
                        help='Path to the data to score',
                        default='gs://storage-siiiiiuuuuu-metaflow-default/data/processed_test_data.csv')

    output_path = Parameter('output_path',
                          help='Path to save predictions',
                          default='gs://storage-siiiiiuuuuu-metaflow-default/data/predictions_gcp.csv')
        
    training_flow_name = Parameter('training_flow_name',
                                 help='Name of the training flow to get the model from',
                                 default='MLTrainingFlowGCP')
    
    mlflow_tracking_uri = Parameter('mlflow_tracking_uri', 
                                  help='MLFlow tracking URI',
                                  default='https://mlflow-server-1000766075950.us-west2.run.app')
    
    @kubernetes
    @retry(times=3)
    @timeout(minutes=10)
    @catch(var='start_error')
    @step
    def start(self):
        """Start the flow"""
        import subprocess
        subprocess.check_call(['pip', 'install', 'pandas', 'numpy', 'scikit-learn', 'mlflow', 
                              'google-cloud-storage', 'google-auth', 'fsspec', 'gcsfs'])
        
        print("Starting the ML Scoring Flow in GCP")
        
        if hasattr(self, 'start_error'):
            print(f"Error caught in start: {self.start_error}")
            
        self.next(self.load_data)
    
    @kubernetes
    @retry(times=3)
    @timeout(minutes=15)
    @catch(var='load_error')
    @step
    def load_data(self):
        """Load the data to score"""
        import subprocess
        subprocess.check_call(['pip', 'install', 'pandas', 'numpy', 'scikit-learn', 'mlflow', 
                              'google-cloud-storage', 'google-auth', 'fsspec', 'gcsfs'])
        
        import pandas as pd
        
        if hasattr(self, 'load_error'):
            print(f"Error caught in load_data: {self.load_error}")
            
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
    
    @kubernetes
    @retry(times=3)
    @timeout(minutes=15)
    @catch(var='model_error')
    @step
    def load_model(self):
        """Load the best model from the training flow"""
        import subprocess
        subprocess.check_call(['pip', 'install', 'pandas', 'numpy', 'scikit-learn', 'mlflow', 
                              'google-cloud-storage', 'google-auth'])
        
        import mlflow
        
        if hasattr(self, 'model_error'):
            print(f"Error caught in load_model: {self.model_error}")
            
        try:
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
            self.model_loaded = True
        except Exception as e:
            print(f"Error loading model: {e}")
            self.model_loaded = False
            # We'll skip to the end if model loading fails
            self.next(self.end)
            return
        
        self.next(self.score_data)
    
    @kubernetes
    @retry(times=3)
    @timeout(minutes=15)
    @catch(var='score_error')
    @step
    def score_data(self):
        """Score the data using the loaded model"""
        import subprocess
        subprocess.check_call(['pip', 'install', 'pandas', 'numpy', 'scikit-learn', 'mlflow', 
                              'google-cloud-storage', 'google-auth'])
        
        import numpy as np
        
        if hasattr(self, 'score_error'):
            print(f"Error caught in score_data: {self.score_error}")
            
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
    
    @kubernetes
    @retry(times=3)
    @timeout(minutes=10)
    @catch(var='save_error')
    @step
    def save_predictions(self):
        """Save the predictions to a file"""
        import subprocess
        subprocess.check_call(['pip', 'install', 'pandas', 'numpy', 'scikit-learn', 
                              'google-cloud-storage', 'google-auth', 'fsspec', 'gcsfs'])
        
        import os
        import pandas as pd
        
        if hasattr(self, 'save_error'):
            print(f"Error caught in save_predictions: {self.save_error}")
            
        # Create a DataFrame with IDs and predictions
        results_df = self.X.copy()
        results_df['predicted_OPS'] = self.predictions
        
        if self.has_target:
            results_df['actual_OPS'] = self.y_true
            results_df['error'] = results_df['actual_OPS'] - results_df['predicted_OPS']
        
        # Save to CSV
        results_df.to_csv(self.output_path, index=False)
        
        print(f"Saved predictions to {self.output_path}")
        
        self.next(self.end)
    
    @kubernetes
    @step
    def end(self):
        """End the flow"""
        import subprocess
        subprocess.check_call(['pip', 'install', 'pandas', 'numpy', 'scikit-learn'])
        
        print("ML Scoring Flow in GCP completed successfully")
        
        if hasattr(self, 'model_loaded') and not self.model_loaded:
            print("Note: Model loading was not successful. No predictions were made.")
            return
            
        if hasattr(self, 'output_path'):
            print(f"Predictions saved to {self.output_path}")
        
        if hasattr(self, 'has_target') and self.has_target and hasattr(self, 'rmse') and hasattr(self, 'r2'):
            print(f"Final metrics - RMSE: {self.rmse:.4f}, R²: {self.r2:.4f}")
        
if __name__ == '__main__':
    MLScoringFlowGCP()