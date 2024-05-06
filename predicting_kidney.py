from fastapi import FastAPI, File, UploadFile, Form
from fastapi.responses import HTMLResponse
from PIL import Image
import numpy as np
import uvicorn
from tensorflow.keras.models import load_model
from datetime import date

app = FastAPI()

# Load the trained model
model = load_model("C:/Users/suresh/Desktop2/Machine Learning/predicting_kidney_model.h5")

# Define the class names
classes = ['Cyst', 'Normal', 'Stone', 'Tumor']

@app.post("/predict/")
async def predict_image(name:str=Form(...),age:int=Form(...),sex:str=Form(...),dob:date=Form(...),file: UploadFile = File(...)):
    try:
        # Read the uploaded image
        image = Image.open(file.file)
        # Resize the image to match the input size of the model
        image = image.resize((224, 224))
        # Convert the image to numpy array
        image = np.array(image) / 255.0  # Normalize pixel values
        # Reshape the image to match the input shape of the model
        image = np.expand_dims(image, axis=0)

        # Predict the class probabilities
        
        predictions = model.predict(image)
        # Get the predicted class index
        predicted_class_index = np.argmax(predictions)
        # Get the predicted class name
        predicted_class_name = classes[predicted_class_index]
        # Get the maximum probability
        max_probability = np.max(predictions)
        # Convert maximum probability to percentage format
        prediction_accuracy = f"{max_probability * 100:.2f}"

        # Return HTML with predicted class and accuracy as JSON data
        return HTMLResponse(content=f"""<!DOCTYPE html>
                <html lang="en">
                <head>
                 <meta charset="UTF-8">
                <meta name="viewport" content="width=device-width, initial-scale=1.0">
    <title>Upload</title>
</head>
<body>
    </div>
    <center>
             <h4>Name</h4>
             <input type="text"  value="{name}"><br> 
             <h4>Age</h4>
             <input type="number"  value="{age}"><br>  
             <h4>Sex</h4>
             <input type="text"  value="{sex}"><br>  
             <h4>DOB</h4>
             <input type="date"  value="{dob}"><br>            
            <h4>Predicted Class:</h4>
            <input type="text" id="predictedClass" value="{predicted_class_name}"><br>
            <h4>Predicted accuracy:</h4>
            <input type="number" id="predictedAccuracy" value="{prediction_accuracy}">
        </body>
        </html>
        """)

    except Exception as e:
        return JSONResponse(status_code=500, content={"error": str(e)})

if __name__ == "__main__":
    uvicorn.run(app)
