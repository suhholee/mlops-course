from metaflow import FlowSpec, step, Parameter
import pandas as pd
import numpy as np
from sklearn.model_selection import cross_val_score
from sklearn.ensemble import RandomForestRegressor, GradientBoostingRegressor
from sklearn.linear_model import LinearRegression
import mlflow
import os
import pickle

class MLTrainingFlow(FlowSpec):
    """
    Flow for training multiple models on baseball data and selecting the best one.
    """
    
    # Parameters that can be passed to the flow
    data_path = Parameter('data_path', 
                         help='Path to the training data',
                         default='data/processed_train_data.csv')
                         
    test_data_path = Parameter('test_data_path', 
                           help='Path to the test data',
                           default='data/processed_test_data.csv')
                           
    cv_folds = Parameter('cv_folds', 
                         help='Number of cross-validation folds',
                         default=5)
                         
    random_state = Parameter('random_state', 
                            help='Random state for reproducibility',
                            default=42)
    
    mlflow_tracking_uri = Parameter('mlflow_tracking_uri', 
                                  help='MLFlow tracking URI',
                                  default='http://localhost:5001')
    
    mlflow_experiment_name = Parameter('mlflow_experiment_name', 
                                     help='MLFlow experiment name',
                                     default='baseball-model-training')
    
    @step
    def start(self):
        """Start the flow and load data"""
        print("Starting the ML Training Flow")
        
        self.next(self.load_data)
    
    @step
    def load_data(self):
        """Load and prepare the data for training"""
        print(f"Loading data from {self.data_path} and {self.test_data_path}")
        
        # Load training data
        train_df = pd.read_csv(self.data_path)
        
        # Load test data
        test_df = pd.read_csv(self.test_data_path)
        
        # Split into features and target
        self.X_train = train_df.drop('On-base Plus Slugging', axis=1)
        self.y_train = train_df['On-base Plus Slugging']
        
        self.X_test = test_df.drop('On-base Plus Slugging', axis=1)
        self.y_test = test_df['On-base Plus Slugging']
        
        print(f"Loaded training data with {len(self.X_train)} samples")
        print(f"Loaded test data with {len(self.X_test)} samples")
        
        self.feature_names = self.X_train.columns.tolist()
        
        self.next(self.train_models)
    
    @step
    def train_models(self):
        """Train multiple models using the prepared data"""
        # Initialize models
        models = {
            'RandomForest': RandomForestRegressor(random_state=self.random_state),
            'LinearRegression': LinearRegression(),
            'GradientBoosting': GradientBoostingRegressor(random_state=self.random_state)
        }
        
        # Store results
        self.model_results = {}
        
        for name, model in models.items():
            print(f"Training {name} model")
            
            # Perform cross-validation
            cv_scores = cross_val_score(
                model, self.X_train, self.y_train, 
                cv=self.cv_folds, 
                scoring='neg_mean_squared_error',
                n_jobs=-1
            )
            
            # Calculate metrics
            cv_rmse = np.sqrt(-cv_scores.mean())
            
            # Train on full training data
            model.fit(self.X_train, self.y_train)
            
            # Calculate test score
            test_rmse = np.sqrt(np.mean((model.predict(self.X_test) - self.y_test) ** 2))
            test_r2 = model.score(self.X_test, self.y_test)
            
            # Store model and metrics
            self.model_results[name] = {
                'model': model,
                'cv_rmse': cv_rmse,
                'test_rmse': test_rmse,
                'test_r2': test_r2
            }
            
            print(f"{name} - CV RMSE: {cv_rmse:.4f}, Test RMSE: {test_rmse:.4f}, Test R²: {test_r2:.4f}")
        
        self.next(self.select_best_model)
    
    @step
    def select_best_model(self):
        """Select the best model based on test R² score"""
        # Choose the model with the highest R² on test data
        best_model_name = max(self.model_results, key=lambda k: self.model_results[k]['test_r2'])
        
        self.best_model_name = best_model_name
        self.best_model = self.model_results[best_model_name]['model']
        self.best_model_metrics = {
            'cv_rmse': self.model_results[best_model_name]['cv_rmse'],
            'test_rmse': self.model_results[best_model_name]['test_rmse'],
            'test_r2': self.model_results[best_model_name]['test_r2']
        }
        
        print(f"Selected best model: {best_model_name}")
        print(f"Best model metrics: {self.best_model_metrics}")
        
        self.next(self.register_model)
    
    @step
    def register_model(self):
        """Register the best model with MLflow"""
        # Set MLflow tracking URI
        mlflow.set_tracking_uri(self.mlflow_tracking_uri)
        
        # Set experiment
        mlflow.set_experiment(self.mlflow_experiment_name)
        
        # Start MLflow run
        with mlflow.start_run(run_name=f"metaflow-run-{self.best_model_name}"):
            # Log parameters
            mlflow.log_param("model_type", self.best_model_name)
            mlflow.log_param("cv_folds", self.cv_folds)
            mlflow.log_param("random_state", self.random_state)
            
            # Log metrics
            for metric_name, metric_value in self.best_model_metrics.items():
                mlflow.log_metric(metric_name, metric_value)
            
            # Log feature importance for tree-based models
            if hasattr(self.best_model, 'feature_importances_'):
                feature_importance = pd.DataFrame({
                    'feature': self.feature_names,
                    'importance': self.best_model.feature_importances_
                }).sort_values('importance', ascending=False)
                
                # Log feature importance as artifact
                feature_importance_path = "feature_importance.csv"
                feature_importance.to_csv(feature_importance_path, index=False)
                mlflow.log_artifact(feature_importance_path)
                os.remove(feature_importance_path)
            
            # Register the model
            mlflow.sklearn.log_model(
                self.best_model, 
                artifact_path="model",
                registered_model_name="baseball-performance-predictor"
            )
            
            # Save model info for the scoring flow
            self.model_info = {
                'name': self.best_model_name,
                'run_id': mlflow.active_run().info.run_id,
                'registered_model_name': "baseball-performance-predictor"
            }
            
            print(f"Registered model '{self.best_model_name}' with MLflow")
            print(f"Model info: {self.model_info}")
        
        self.next(self.end)
    
    @step
    def end(self):
        """End the flow"""
        print("ML Training Flow completed successfully")
        print(f"Best model: {self.best_model_name}")
        print(f"Best model metrics: {self.best_model_metrics}")
        print(f"Model info: {self.model_info}")
        
if __name__ == '__main__':
    MLTrainingFlow()