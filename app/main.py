from fastapi import FastAPI, File, UploadFile
from fastapi.middleware.cors import CORSMiddleware
from app.model import load_model, predict_digit
from PIL import Image
import io

app = FastAPI()

# CORS settings
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"], # allows all origins. Fine for tech demo, otherwise unsave and needs adjustment
    allow_credentials=True, 
    allow_methods=["*"],
    allow_headers=["*"],
)

# Load model on startup
model = load_model()

@app.post("/predict")
async def predict(file: UploadFile = File(...)):
    try:
        image_bytes = await file.read()
        image = Image.open(io.BytesIO(image_bytes)).convert("L").resize((28, 28))
        predicted_digit = predict_digit(model, image)
        print(f"[OK] Predicted digit: {predicted_digit}")
        return {"prediction": predicted_digit}
    except Exception as e:
        print(f"[ERROR] Failed to predict: {e}")
        return {"prediction": "Error"}


# check state or wake it up
@app.get("/ping")
async def ping():
    return {"status": "alive"}

# ping for UpTimeRobot
@app.api_route("/", methods=["GET", "HEAD"])
async def root():
    return {"message": "Digit Classifier API is alive."}

