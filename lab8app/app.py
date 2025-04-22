from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
import joblib
import os

# Define the FastAPI app
app = FastAPI(
    title="Reddit Comment Classifier",
    description="Classify Reddit comments as either 1 = Remove or 0 = Do Not Remove.",
    version="0.1",
)

# Define the input data model
class CommentInput(BaseModel):
    reddit_comment: str

# Define the prediction output model
class PredictionOutput(BaseModel):
    prediction: float
    remove_probability: float
    keep_probability: float

# Load the model
def load_model():
    try:
        model_path = "reddit_model_pipeline.joblib"
        model = joblib.load(model_path)
        return model
    except Exception as e:
        print(f"Error loading model: {e}")
        raise HTTPException(status_code=500, detail="Model could not be loaded")

model = load_model()

# Root endpoint
@app.get('/')
def root():
    return {'message': 'Reddit Comment Classification API'}

# Prediction endpoint
@app.post('/predict', response_model=PredictionOutput)
def predict(comment_input: CommentInput):
    try:
        # Extract the comment from the request
        comment = [comment_input.reddit_comment]
        
        # Make prediction
        proba = model.predict_proba(comment)
        prediction = model.predict(comment)
        
        # Return the prediction and probabilities
        return {
            "prediction": float(prediction[0]),
            "remove_probability": float(proba[0][1]),
            "keep_probability": float(proba[0][0])
        }
    except Exception as e:
        print(f"Error making prediction: {e}")
        raise HTTPException(status_code=500, detail=str(e))

# Run the app with uvicorn
if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000)