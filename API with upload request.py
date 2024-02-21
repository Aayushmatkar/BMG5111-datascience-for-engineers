 #mainurl.py

from fastapi import FastAPI, HTTPException, File,UploadFile
from fastapi.responses import HTMLResponse
from pydantic import BaseModel
import tensorflow as tf
from keras.preprocessing import image
import numpy as np

app = FastAPI()

# Load the saved model
#model = tf.keras.models.load_model('csa_glaucoma1.h5')

# Define the request payload using Pydantic BaseModel
class ImagePayload(BaseModel):
    image_url: str


# Endpoint to make predictions
@app.get("/predict")
async def predict_image():
        
        # Load the saved model
        model = tf.keras.models.load_model('csa_glaucoma1.h5')
    
        pic = r"C:\Users\aayus\OneDrive\Desktop\WINTER 2024\project\val\glauc True 1.png"
        # Download and preprocess the image
        img = image.load_img(pic, target_size=(150, 150))
        img_array = image.img_to_array(img)
        img_array = np.expand_dims(img_array, axis=0)
        img_array /= 255.0  # Normalize pixel values to [0, 1]

        # Make predictions using te loaded model
        prediction = model.predict(img_array)
        result = 1 if prediction[0][0] > 0.5 else 0
        # print(prediction)
        return {"prediction": result,
                "value": float(prediction[0][0])}
        #testing api
        return {"test":"Hello World!",
                "Data": test_var
                }


@app.post("/files/")
async def create_files(files: list[bytes] = File()):
    return {"file_sizes": [len(file) for file in files]}


@app.post("/uploadfiles/")
async def create_upload_files(files: list[UploadFile]):
    return {"filenames": [file.filename for file in files]}


@app.get("/") 
async def main():
    content = """
<body>
<form action="/files/" enctype="multipart/form-data" method="post">
<input name="files" type="file" multiple>
<input type="submit">
</form>
</body>
    """
    return HTMLResponse(content=content)
 
