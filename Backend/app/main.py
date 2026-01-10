import uvicorn
import os
from dotenv import load_dotenv
load_dotenv()
import io
import numpy as np
import tensorflow as tf
from PIL import Image 
from typing import Dict, Any, List, Optional
import time 
import bcrypt
import threading 
from fastapi import FastAPI, UploadFile, File, HTTPException, status, Depends, Form 
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel, Field
import psycopg2
from psycopg2 import pool, sql
from psycopg2.extras import DictCursor 
from tensorflow.python.keras.models import load_model








DATABASE_URL = os.getenv("DATABASE_URL")


MODEL_PATH = "./best_coffee_disease2_model.h5" 
IMG_HEIGHT = 224 
IMG_WIDTH = 224 
CONFIDENCE_THRESHOLD = 0.70 
NON_COFFEE_LEAF_CLASS_NAME = "Other_Non_Coffee_Leaf" 


NUM_CLASSES = 7 

CLASS_NAMES = [
    'Cerscospora', 'Other_Non_Coffee_Leaf', 'coffee___healthy', 'coffee___red_spider_mite', 'coffee___rust', 'miner', 'phoma'
]

RECOMMENDATIONS = {
    # FIX: Keys must match the exact class names (e.g., 'coffee___rust' not 'Coffee Leaf Rust')
    'coffee___rust': "Coffee Leaf Rust (La Roya) detected. Use resistant varieties, apply systemic fungicides, and ensure proper shade management. Action: **Fungicide and Shade Management**",
    'miner': "Coffee Leaf Miner attack. Prune infected leaves, use biological controls (predators/parasites), or targeted insecticides. Action: **Prune and Targeted Insecticides**",
    'phoma': "Phoma Leaf Spot. Reduce canopy density for better airflow, avoid overhead watering, and apply copper-based fungicides if necessary. Action: **Prune and Apply Copper**",
    'coffee___red_spider_mite': "Red Spider Mite infestation. Apply miticides specifically targeted at mites. Increase humidity and use insecticidal soaps/oils. Action: **Miticides and Humidity**",
    'Cerscospora': "Cercospora Leaf Spot detected. Often caused by poor nutrition or high humidity. Improve soil fertility (especially Potassium/Zinc) and apply fungicides if severe. Action: **Improve Nutrition and Fungicide**",
    'coffee___healthy': "Your coffee plant appears healthy! Continue good agricultural practices, including proper fertilization, pest monitoring, and pruning. Action: **Maintain Good Practices**",
    NON_COFFEE_LEAF_CLASS_NAME: "The uploaded image is not a coffee leaf. Please ensure you are submitting a clear photo of a coffee leaf for diagnosis. Action: **Retake Photo**"
}


app = FastAPI(
    title="CoffeeScan AI Backend",
    description="API for detecting coffee leaf diseases and managing user scan history with PostgreSQL.",
    version="1.0.0", 
)

model = None
db_pool = None
model_lock = threading.Lock() 

origins = ["https://coffeescan.netlify.app", "http://localhost:3000", "http://127.0.0.1:8000", "http://localhost:5173", "*"] 
app.add_middleware(
    CORSMiddleware,
    allow_origins=origins,
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# --- Pydantic Models (UPDATED for Username) ---
class UserRegistration(BaseModel):
    username: str # Added username field
    email: str
    password: str
class UserLogin(BaseModel):
    email: str
    password: str
class SavedScan(BaseModel):
    user_email: str 
    diagnosis_result: str
    confidence_score : float
    treatment_recommendation: str 
    scan_date: Optional[str] = None 
class PredictionResponse(BaseModel):
    status: str = Field(..., description="SUCCESS, REJECTED (Non-Coffee), or LOW_CONFIDENCE")
    prediction: str = Field(..., description="The predicted class name.")
    confidence: float
    message: str = Field(..., description="A summary of the result.")
    recommendation: str = Field(..., description="Specific advice or instruction for the user.")
class PredictionAndSaveResponse(PredictionResponse):
    """Extends PredictionResponse to include scan saving status and ID."""
    save_status: str = Field(..., description="Success/Failure status of the DB save operation.")
    scan_id: Optional[int] = Field(None, description="The ID of the saved scan record, if successful.")

# --- Database Connection and Utility Functions (omitted for brevity) ---

def initialize_pool():
    global db_pool
    if db_pool is None:
        if not DATABASE_URL:
            # Enforce DATABASE_URL existence since local fallback was removed
            raise EnvironmentError("DATABASE_URL environment variable is required for production connection.")
            
        try:
            print("Using DATABASE_URL for Neon connection.")
            db_pool = psycopg2.pool.SimpleConnectionPool(
                minconn=1, 
                maxconn=10, 
                dsn=DATABASE_URL,
            )
            print("PostgreSQL connection pool initialized successfully. 🥳")
        except Exception as e:
            print(f"Failed to initialize connection pool: {e}")
            raise

def get_db_connection():
    if db_pool is None:
        raise HTTPException(status_code=500, detail="Database pool not initialized.")
        
    start_time = time.time()
    conn = db_pool.getconn()
    elapsed = time.time() - start_time

    try:
        yield conn
    finally:
        db_pool.putconn(conn)

# --- DB Helpers (omitted for brevity) ---
def get_user_id_by_email(email: str, conn: psycopg2.connect) -> Optional[int]:
    """Helper function to fetch user_id from email."""
    try:
        with conn.cursor() as cur:
            cur.execute(sql.SQL("SELECT user_id FROM users WHERE email = %s"), (email,))
            result = cur.fetchone()
            return result[0] if result else None
    except Exception as e:
        print(f"Error fetching user ID for {email}: {e}")
        return None
        
def hash_password(password: str) -> str:
    """Hashes the plain text password using bcrypt."""
    salt = bcrypt.gensalt()
    hashed = bcrypt.hashpw(password.encode('utf-8'), salt)
    return hashed.decode('utf-8')

def verify_password(plain_password: str, hashed_password: str) -> bool:
    """Verifies a plain text password against a stored hashed password."""
    try:
        return bcrypt.checkpw(plain_password.encode('utf-8'), hashed_password.encode('utf-8'))
    except Exception as e:
        print(f"Error during password verification: {e}")
        return False

def save_scan_to_db(user_email: str, diagnosis: str, confidence: float, recommendation: str, conn: psycopg2.connect) -> int:
    """
    Saves a prediction scan to the user's persistent history in PostgreSQL.
    Raises HTTPException on failure.
    """
    
    user_id = get_user_id_by_email(user_email, conn)
    
    if user_id is None:
        raise HTTPException(status_code=status.HTTP_404_NOT_FOUND, detail="User not found.")
        
    try:
        with conn.cursor() as cur:
            cur.execute(
                sql.SQL("""
                    INSERT INTO scans (user_id, diagnosis_result, confidence_score, treatment_recommendation)
                    VALUES (%s, %s, %s, %s)
                    RETURNING scan_id;
                """),
                (
                    user_id, 
                    diagnosis, 
                    confidence, 
                    recommendation
                )
            )
            scan_id = cur.fetchone()[0]
            conn.commit()
            return scan_id
            
    except Exception as e:
        conn.rollback()
        print(f"Database insertion error for scan: {e}")
        raise HTTPException(status_code=status.HTTP_500_INTERNAL_SERVER_ERROR, detail="Failed to save scan history due to a database error.")


# --- MODEL ARCHITECTURE REBUILD (CRITICAL STEP) ---

def create_model_architecture(input_shape=(IMG_HEIGHT, IMG_WIDTH, 3), num_classes=NUM_CLASSES):
    """
    Rebuilds the transfer learning model architecture used for CoffeeScan AI,
    matching the exact head used in train_model.py.
    """
    try:
        # 1. Load the base MobileNetV2 model pre-trained on ImageNet
        base_model = tf.keras.applications.MobileNetV2(
            input_shape=input_shape,
            include_top=False, 
            weights='imagenet'
        )
        base_model.trainable = False # Freeze the base layers
        
        # 2. Build the custom classification head using the Functional API 
        #    to exactly match the layers in train_model.py:
        #    GAP -> Dropout(0.5) -> Dense(256) -> Dropout(0.5) -> Dense(N)
        
        x = base_model.output
        x = tf.keras.layers.GlobalAveragePooling2D()(x) # Global Average Pooling
        x = tf.keras.layers.Dropout(0.5)(x)              # First Dropout
        x = tf.keras.layers.Dense(256, activation='relu')(x) # Dense(256) layer
        x = tf.keras.layers.Dropout(0.5)(x)              # Second Dropout
        predictions = tf.keras.layers.Dense(num_classes, activation='softmax')(x) # Final Classification Head
        
        model = tf.keras.Model(inputs=base_model.input, outputs=predictions)
        
        # NOTE: Compiling is required before loading weights
        # Use a generic optimizer/loss just for compilation/weight loading structure
        model.compile(optimizer='adam', loss='categorical_crossentropy', metrics=['accuracy'])
        
        return model
    except Exception as e:
        raise RuntimeError(f"Failed to build model architecture: {e}")


def load_ml_model_lazy():
    """Loads weights into the model architecture built in code."""
    global model
    try:
        # 1. Build the model architecture from scratch using pure Python/TensorFlow code
        model = create_model_architecture() 
        
        # 2. Load only the weights from the H5 file
        model.load_weights(MODEL_PATH)
        
        print(f"ML weights loaded successfully (Architecture rebuilt and weights applied).")
    except Exception as e:
        # If model building or weight loading fails, raise 503
        print(f"ERROR: Could not lazy load ML model weights. Reason: {e}")
        raise HTTPException(status_code=503, detail="ML model failed to load during first request.")
# ---------------------------------------------


# --- Startup/Shutdown Events (Restore simple DB startup) ---

@app.on_event("startup")
def startup_events():
    """Initialize DB pool only."""
    initialize_pool()
    # Model is loaded lazily on first request

@app.on_event("shutdown")
def shutdown_db_event():
    global db_pool
    if db_pool:
        db_pool.closeall()
        print("PostgreSQL connection pool closed.")

# --- ML Prediction Core Function (Restored Lazy Load Check) ---

async def predict_disease_actual_model(image_bytes: bytes) -> PredictionResponse:
    """Performs image preprocessing and model inference."""
    global model
    
    # Restore Lazy Load Check
    if model is None:
        with model_lock:
            if model is None:
                print(f"ATTEMPTING LAZY LOAD OF ML MODEL from {MODEL_PATH}")
                try:
                    load_ml_model_lazy()
                except HTTPException as e:
                    # Propagate the 503 error up to the endpoint handler
                    raise e
    
    # If model is still None after lazy load attempt (due to error), crash gracefully
    if model is None:
        raise HTTPException(
            status_code=status.HTTP_503_SERVICE_UNAVAILABLE, 
            detail="ML model failed to load during request."
        )

    try:
        # Load and preprocess the image
        image = Image.open(io.BytesIO(image_bytes)).convert("RGB")
        image = image.resize((IMG_HEIGHT, IMG_WIDTH)) 
        image_array = np.asarray(image) 
        image_array = image_array / 255.0 
        image_batch = np.expand_dims(image_array, axis=0) 

        # Make prediction
        predictions = model.predict(image_batch, verbose=0)
        predicted_probabilities = predictions[0] 
        predicted_class_index = np.argmax(predicted_probabilities)
        confidence = float(predicted_probabilities[predicted_class_index])
        predicted_disease = CLASS_NAMES[predicted_class_index]
        
    except Image.UnidentifiedImageError:
        raise HTTPException(status_code=400, detail="Invalid image file. Could not identify image format.")
    except Exception as e:
        print(f"Prediction error: {e}")
        raise HTTPException(status_code=500, detail=f"Failed to process image or make prediction: {str(e)}")

    # A. Check for Non-Coffee Image (Updated variable name)
    if predicted_disease == NON_COFFEE_LEAF_CLASS_NAME:
        rejection_message = RECOMMENDATIONS[NON_COFFEE_LEAF_CLASS_NAME]
        return PredictionResponse(
            status="REJECTED",
            prediction=predicted_disease,
            confidence=round(confidence, 4),
            message="Image Rejected: The uploaded photo is not a coffee leaf (Non-Coffee Leaf detected).",
            recommendation=rejection_message
        )

    # B. Check for Low Confidence Warning
    if confidence < CONFIDENCE_THRESHOLD:
        generic_recommendation = "Diagnosis uncertainty is high. Retake the photo or consult a local expert for verification."
        return PredictionResponse(
            status="LOW_CONFIDENCE",
            prediction=predicted_disease,
            confidence=round(confidence, 4),
            message=f"The top prediction is **{predicted_disease}**, but the confidence score ({round(confidence * 100, 2)}%) is below the {CONFIDENCE_THRESHOLD * 100}% threshold. Retrying with a clearer image is advised.",
            recommendation=generic_recommendation
        )

    # C. Successful Prediction (High Confidence)
    recommendation = RECOMMENDATIONS.get(predicted_disease, "Consult a local agricultural expert for precise guidance.")
    
    return PredictionResponse(
        status="SUCCESS",
        prediction=predicted_disease,
        confidence=round(confidence, 4),
        message=f"High confidence diagnosis: **{predicted_disease}**",
        recommendation=recommendation
    )


# --- ENDPOINTS (Authentication, History, etc. remain the same) ---
@app.get("/")
async def read_root():
    """Root endpoint for the CoffeeScan AI API."""
    return {"message": "Welcome to CoffeeScan AI Unified Backend! Go to /docs for API documentation."}

@app.post("/predict", response_model=PredictionAndSaveResponse, tags=["ML Prediction"])
async def predict_disease_and_save_endpoint(
    file: UploadFile = File(..., description="The image file of the coffee leaf."), 
    user_email: str = Form(..., description="The email of the user to associate the scan with."),
    conn: psycopg2.connect = Depends(get_db_connection)
):
    """
    Receives an image, performs prediction, and automatically saves the result 
    to the user's scan history if the prediction is accepted (SUCCESS/LOW_CONFIDENCE).
    """
    # 1. Read Image and Get Prediction Result
    image_bytes = await file.read()
    
    # Note: Model loading is handled via lazy load and crash handling in predict_disease_actual_model
    prediction_result = await predict_disease_actual_model(image_bytes)

    # 2. Handle Saving (omitted for brevity)
    scan_id = None
    save_status = "NOT_SAVED_REJECTED"
    
    if prediction_result.status in ["SUCCESS", "LOW_CONFIDENCE"]:
        try:
            scan_id = save_scan_to_db(
                user_email=user_email,
                diagnosis=prediction_result.prediction,
                confidence=prediction_result.confidence,
                recommendation=prediction_result.recommendation,
                conn=conn
            )
            save_status = "SAVED_SUCCESS"
        except HTTPException as e:
            save_status = f"SAVED_FAILED_{e.detail.replace(' ', '_').upper()}"
            print(f"Failed to save scan for user {user_email}: {e.detail}")
        except Exception as e:
            save_status = "SAVED_FAILED_UNKNOWN_ERROR"
            print(f"Unexpected error during save: {e}")

    # 3. Construct Final Response with Save Status
    return PredictionAndSaveResponse(
        status=prediction_result.status,
        prediction=prediction_result.prediction,
        confidence=prediction_result.confidence,
        message=prediction_result.message,
        recommendation=prediction_result.recommendation,
        save_status=save_status,
        scan_id=scan_id
    )

@app.post("/register", tags=["Auth"])
def register_user(user_data: UserRegistration, conn: psycopg2.connect = Depends(get_db_connection)):
    """Handles user registration using PostgreSQL with password hashing, including username."""
    username = user_data.username
    email = user_data.email
    hashed_password = hash_password(user_data.password)
    
    try:
        with conn.cursor() as cur:
            # Check if email is already registered
            cur.execute(sql.SQL("SELECT user_id FROM users WHERE email = %s"), (email,))
            if cur.fetchone():
                raise HTTPException(
                    status_code=status.HTTP_409_CONFLICT,
                    detail="Email already registered."
                )
                
            # Check if username is already taken
            cur.execute(sql.SQL("SELECT user_id FROM users WHERE username = %s"), (username,))
            if cur.fetchone():
                raise HTTPException(
                    status_code=status.HTTP_409_CONFLICT,
                    detail="Username already taken."
                )
            
            cur.execute(
                sql.SQL("INSERT INTO users (username, email, hashed_password) VALUES (%s, %s, %s) RETURNING user_id;"),
                (username, email, hashed_password)
            )
            user_id = cur.fetchone()[0]
            conn.commit()
            
            return {"message": "User registered successfully.", "user_id": user_id, "email": email, "username": username}
            
    except HTTPException as e:
        # Re-raise explicit HTTP exceptions
        raise e
    except Exception as e:
        conn.rollback()
        print(f"Database insertion error: {e}")
        raise HTTPException(status_code=500, detail=f"Registration failed due to server error.")

@app.post("/login", tags=["Auth"])
def login_user(user_data: UserLogin, conn: psycopg2.connect = Depends(get_db_connection)):
    """Authenticates the user by checking email and password against PostgreSQL, returning username."""
    email = user_data.email
    password = user_data.password

    try:
        with conn.cursor() as cur:
            # Select username along with ID and password
            cur.execute(
                sql.SQL("SELECT user_id, username, hashed_password FROM users WHERE email = %s;"),
                (email,)
            )
            result = cur.fetchone()
            
            if not result:
                raise HTTPException(
                    status_code=status.HTTP_401_UNAUTHORIZED,
                    detail="Invalid credentials."
                )
            
            user_id, username, stored_hashed_password = result 
            if not verify_password(password, stored_hashed_password):
                raise HTTPException(
                    status_code=status.HTTP_401_UNAUTHORIZED,
                    detail="Invalid credentials."
                )
            
            mock_token = f"fake_auth_token_for_{email}" 
            # Include username in the successful login response
            return {"message": "Login successful.", "user_id": user_id, "email": email, "username": username, "token": mock_token}

    except HTTPException as e:
        raise e
    except Exception as e:
        conn.rollback()
        print(f"Login error: {e}")
        raise HTTPException(status_code=status.HTTP_500_INTERNAL_SERVER_ERROR, detail="Login failed due to server error.")

@app.post("/logout", tags=["Auth"])
async def logout_user():
    """
    Handles user logout. Since this application uses client-side managed tokens, 
    this endpoint primarily signals success to the frontend, which then clears 
    its local token/session. 
    """
    return {"message": "Logout successful. Client must destroy local session/token."}


@app.delete("/delete_scan/{scan_id}", tags=["History (PostgreSQL)"])
def delete_scan(scan_id: int, conn: psycopg2.connect = Depends(get_db_connection)):
    """
    Deletes a specific scan by its scan_id from the database.
    Returns success or failure message.
    """
    try:
        with conn.cursor() as cur:
            # Check if the scan exists
            cur.execute("SELECT scan_id FROM scans WHERE scan_id = %s;", (scan_id,))
            if not cur.fetchone():
                raise HTTPException(
                    status_code=status.HTTP_404_NOT_FOUND,
                    detail=f"Scan with ID {scan_id} not found."
                )

            # Delete the scan
            cur.execute("DELETE FROM scans WHERE scan_id = %s;", (scan_id,))
            conn.commit()
            
            return {"message": f"Scan with ID {scan_id} deleted successfully."}

    except HTTPException as e:
        raise e
    except Exception as e:
        conn.rollback()
        print(f"Database error while deleting scan {scan_id}: {e}")
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Failed to delete scan {scan_id} due to server error."
        )

@app.get("/diseases", tags=["DB Data"], response_model=List[Dict[str, Any]])
def get_disease_list(conn: psycopg2.connect = Depends(get_db_connection)):
    """Fetches a list of known diseases from the database."""
    try:
        with conn.cursor() as cur:
            cur.execute("SELECT disease_name, description FROM diseases;")
            columns = [desc[0] for desc in cur.description]
            diseases = [dict(zip(columns, row)) for row in cur.fetchall()]
            return diseases
    except Exception as e:
        conn.rollback()
        print(f"Database error fetching diseases: {e}")
        raise HTTPException(status_code=500, detail=f"Database error: Could not fetch disease list.")

@app.get("/get_scans/{user_email}", tags=["History (PostgreSQL)"])
async def get_scans_endpoint(user_email: str, conn: psycopg2.connect = Depends(get_db_connection)):
    """Retrieves all saved scans for a specific user from PostgreSQL."""
    
    user_id = get_user_id_by_email(user_email, conn)
    
    if user_id is None:
        return {"scans": [], "count": 0, "message": "User not found or no scans available."}
        
    try:
        with conn.cursor(cursor_factory=DictCursor) as cur: 
            cur.execute(
                sql.SQL("""
                    SELECT 
                        scan_id,
                        diagnosis_result as prediction, 
                        confidence_score as confidence, 
                        treatment_recommendation,
                        scan_date as date
                    FROM scans 
                    WHERE user_id = %s
                    ORDER BY scan_date DESC;
                """),
                (user_id,)
            )
            scans = [dict(row) for row in cur.fetchall()] 
            
            return {"scans": scans, "count": len(scans), "message": "Scans retrieved successfully from PostgreSQL."}
            
    except Exception as e:
        conn.rollback()
        print(f"Database query error for scans: {e}")
        raise HTTPException(status_code=status.HTTP_500_INTERNAL_SERVER_ERROR, detail="Failed to retrieve scan history.")


# Main block for running the application
if __name__ == "__main__":
    # Use the PORT environment variable if it exists (for Render), otherwise default to 8000 (for local development)
    PORT = int(os.environ.get("PORT", 8000))
    uvicorn.run("app.main:app", host="0.0.0.0", port=PORT)
    
