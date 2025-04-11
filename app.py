from dotenv import load_dotenv
import os
import uuid
import pandas as pd
from fastapi import FastAPI, UploadFile, File, Form, HTTPException
from fastapi.responses import JSONResponse, FileResponse
from fastapi.middleware.cors import CORSMiddleware
from fastapi.staticfiles import StaticFiles
from typing import Optional
import uvicorn
import json
from langchain_anthropic import ChatAnthropic
from pandasai import SmartDataframe
import shutil
import matplotlib
matplotlib.use('Agg')  # Set the backend to 'Agg' to prevent charts from opening

# Load environment variables
load_dotenv()

# Create FastAPI app
app = FastAPI(title="Data Visualization API")

# Add CORS middleware
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Create directories if they don't exist
os.makedirs("uploads", exist_ok=True)
os.makedirs("exports/charts", exist_ok=True)
os.makedirs("static", exist_ok=True)

# Mount static files directory
app.mount("/static", StaticFiles(directory="static"), name="static")

# Global variables to store data and model
data_store = {}

@app.get("/")
async def get_index():
    return FileResponse("static/index.html")

@app.post("/upload")
async def upload_file(file: UploadFile = File(...)):
    try:
        # Generate a unique ID for this session
        session_id = str(uuid.uuid4())
        
        # Save the uploaded file
        file_path = f"uploads/{session_id}_{file.filename}"
        with open(file_path, "wb") as buffer:
            shutil.copyfileobj(file.file, buffer)
        
        # Read the file based on its extension
        file_extension = file.filename.split(".")[-1].lower()
        
        if file_extension == "csv":
            data = pd.read_csv(file_path)
        elif file_extension in ["xlsx", "xls"]:
            data = pd.read_excel(file_path)
        else:
            raise HTTPException(status_code=400, detail="Unsupported file format. Please upload a CSV or Excel file.")
        
        # Initialize SmartDataframe with the uploaded data
        model = ChatAnthropic(model="claude-3-haiku-20240307")
        
        # Custom prompt template for visualization focus
        visualization_prompt = """
        You are a data visualization expert. Your task is to create insightful visualizations based on the data.
        Focus on creating clear, informative charts that reveal patterns, trends, or insights in the data.
        
        When asked to create a visualization:
        1. Choose the most appropriate chart type for the data and question
        2. Create a clean, well-labeled visualization
        3. Provide a brief interpretation of what the visualization shows
        
        Current data summary:
        {df_summary}
        
        User question: {prompt}
        """
        
        smart_df = SmartDataframe(
            data, 
            config={
                "llm": model,
                "verbose": False,
                "display_charts": False,
                "save_charts": True,  # Ensure charts are saved but not displayed
                "save_charts_path": "exports/charts/",
                "custom_prompts": {"generate_visualization": visualization_prompt},
                "open_charts": False  # Explicitly set to not open charts
            }
        )
        
        # Store data and model for this session
        data_store[session_id] = {
            "data": data,
            "smart_df": smart_df,
            "file_name": file.filename,
            "file_path": file_path
        }
        
        # Analyze data for initial response
        columns_info = {col: str(data[col].dtype) for col in data.columns}
        numeric_cols = data.select_dtypes(include=['number']).columns.tolist()
        categorical_cols = data.select_dtypes(include=['object', 'category']).columns.tolist()
        date_cols = data.select_dtypes(include=['datetime']).columns.tolist()
        
        # Create data preview (first 5 rows) - ensure JSON serializable
        preview_data = data.head(5).copy()
        # Convert any non-serializable types
        for col in preview_data.select_dtypes(include=['datetime64']).columns:
            preview_data[col] = preview_data[col].astype(str)
        
        # Convert DataFrame to a list of dictionaries
        records = preview_data.to_dict(orient="records")
        
        # Custom JSON serialization to handle non-JSON compliant values
        def clean_for_json(obj):
            if isinstance(obj, dict):
                return {k: clean_for_json(v) for k, v in obj.items()}
            elif isinstance(obj, list):
                return [clean_for_json(i) for i in obj]
            elif isinstance(obj, float):
                if pd.isna(obj) or pd.isnull(obj):
                    return 0
                elif obj == float('inf'):
                    return "Infinity"  # Use string representation
                elif obj == float('-inf'):
                    return "-Infinity"  # Use string representation
                else:
                    return obj
            else:
                return obj
        
        # Clean the records for JSON serialization
        preview_data = clean_for_json(records)
        
        # Generate visualization suggestions
        suggestions = []
        
        if len(numeric_cols) >= 1:
            suggestions.append(f"Distribution analysis of {numeric_cols[0]} (histogram or box plot)")
        
        if len(numeric_cols) >= 2:
            suggestions.append(f"Correlation between {numeric_cols[0]} and {numeric_cols[1]} (scatter plot)")
        
        if categorical_cols and numeric_cols:
            suggestions.append(f"{numeric_cols[0]} by {categorical_cols[0]} (bar chart)")
        
        if len(categorical_cols) >= 2:
            suggestions.append(f"Relationship between {categorical_cols[0]} and {categorical_cols[1]} (heatmap)")
        
        if date_cols and numeric_cols:
            suggestions.append(f"{numeric_cols[0]} over time using {date_cols[0]} (line chart)")
        
        if not suggestions:
            suggestions.append("Basic summary statistics and counts")
        
        return JSONResponse({
            "session_id": session_id,
            "message": f"File '{file.filename}' uploaded successfully!",
            "data_info": {
                "rows": data.shape[0],
                "columns": data.shape[1],
                "numeric_columns": numeric_cols,
                "categorical_columns": categorical_cols,
                "date_columns": date_cols,
                "preview": preview_data,
                "column_names": data.columns.tolist()
            },
            "suggestions": suggestions
        })
    
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

@app.post("/chat/{session_id}")
async def chat(session_id: str, prompt: str = Form(...)):
    try:
        if session_id not in data_store:
            raise HTTPException(status_code=404, detail="Session not found. Please upload a file first.")
        
        # Get the SmartDataframe for this session
        smart_df = data_store[session_id]["smart_df"]
        
        # Enhance the prompt to focus on visualization
        viz_prompt = f"Create a visualization: {prompt}"
        
        # Get response from PandasAI
        response = smart_df.chat(viz_prompt)
        
        # Check if the response is just a file path
        if response and response.strip().endswith('.png') and os.path.exists(response.strip()):
            # If the response is just a file path, add a descriptive message
            chart_path = response.strip()
            response = f"Here's the visualization you requested. The chart shows the data according to your specifications."
        else:
            # If not, use the existing response
            chart_path = None
        
        # Generate a unique filename for the chart
        chart_id = str(uuid.uuid4())
        new_chart_path = f"exports/charts/{chart_id}.png"
        
        # Check for charts in the exports/charts directory
        chart_exists = False
        chart_url = None
        
        # First check if we have a direct chart path from the response
        if chart_path and os.path.exists(chart_path):
            chart_exists = True
            # Copy the chart to the new location
            shutil.copy(chart_path, new_chart_path)
        else:
            # Check for temp_chart.png
            temp_chart_path = "exports/charts/temp_chart.png"
            if os.path.exists(temp_chart_path):
                chart_exists = True
                # Rename the temp chart to a unique name to preserve it
                os.rename(temp_chart_path, new_chart_path)
            else:
                # Look for any recently created PNG files in the exports/charts directory
                chart_files = [f for f in os.listdir("exports/charts") if f.endswith('.png')]
                if chart_files:
                    # Sort by modification time, newest first
                    chart_files.sort(key=lambda x: os.path.getmtime(os.path.join("exports/charts", x)), reverse=True)
                    # Use the most recent chart
                    most_recent_chart = os.path.join("exports/charts", chart_files[0])
                    chart_exists = True
                    # Copy the chart to the new location
                    shutil.copy(most_recent_chart, new_chart_path)
        
        if chart_exists:
            # Copy the chart to the static directory for serving
            os.makedirs("static/charts", exist_ok=True)
            shutil.copy(new_chart_path, f"static/charts/{chart_id}.png")
            chart_url = f"/static/charts/{chart_id}.png"
        
        return JSONResponse({
            "response": response,
            "chart_url": chart_url
        })
    
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

@app.get("/sessions/{session_id}")
async def get_session(session_id: str):
    if session_id not in data_store:
        raise HTTPException(status_code=404, detail="Session not found")
    
    session_data = data_store[session_id]
    
    return JSONResponse({
        "file_name": session_data["file_name"],
        "columns": session_data["data"].columns.tolist(),
        "rows": session_data["data"].shape[0]
    })

@app.delete("/sessions/{session_id}")
async def delete_session(session_id: str):
    if session_id not in data_store:
        raise HTTPException(status_code=404, detail="Session not found")
    
    # Remove the file
    if os.path.exists(data_store[session_id]["file_path"]):
        os.remove(data_store[session_id]["file_path"])
    
    # Remove from data store
    del data_store[session_id]
    
if __name__ == "__main__":
    uvicorn.run("app:app", host="0.0.0.0", port=8000, reload=True)




#conda activate pandasai-env
