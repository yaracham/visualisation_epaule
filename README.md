# Shoulder Movement Analysis – Motion Capture Visualization

This project focuses on the analysis of shoulder movement using motion capture data.  
It provides tools to visualize and quantify body motion through 3D trajectories and derived indicators.

The goal is to better understand how different parts of the body contribute to movement, especially in the context of biomechanical analysis.

---

## Features

- Import motion capture files (`.TRC`)
- 3D visualization of marker trajectories (point cloud)
- Selection of specific body markers
- Computation of kinematic indicators:
  - Total displacement
  - Trajectory length
  - Spatial dispersion
- Ranking of marker involvement
- Optional simulation of movement limitation
- Export of results (CSV)

---

## Project Structure

visualisation_epaule/
│
├── app.py                  # Main Streamlit application (UI + logic)
├── src/                    # Core processing modules
├── data/                   # Sample TRC files for testing
│   ├── example1.trc
│   ├── example2.trc
│   └── ... (more files will be added)
├── outputs/                # Optional exported results
├── requirements.txt        # Python dependencies
└── README.md               # Project documentation

The `data/` folder contains several `.TRC` test files that can be used directly in the application.  
More datasets will be added progressively.

---

## Installation

Clone the repository:

    git clone https://github.com/yaracham/visualisation_epaule.git
    cd visualisation_epaule

Install dependencies:

    pip install -r requirements.txt

Or manually:

    pip install pandas numpy plotly streamlit

---

## Running the Application

    streamlit run app.py

Then open your browser at:

    http://localhost:8501

---

## How to Use the App

### 1. Upload a file

- Click on "Upload TRC file"
- Select a `.trc` file (you can use files from the `/data` folder)

### 2. Configure the analysis

In the sidebar:

- Select markers
- Adjust sampling
- Modify point size
- Choose number of markers in ranking

### 3. Simulate limitation (optional)

- Enable "Simulate limited motion"
- Adjust limitation factor

### 4. Explore results

The app displays:

- 3D point cloud (interactive)
- Bar chart (marker involvement)
- Table of computed indicators

### 5. Export results

- Click "Download CSV"

---

## Kinematic Indicators

- Total displacement  
- Trajectory length  
- Spatial dispersion  

These are combined into a global involvement score.

---

##  Project Scope

This project focuses on motion analysis using dynamic data and identifying how movement is distributed across the body.

The interface is a support tool for visualization, not the core objective.

---

##  Technologies

- Python
- Pandas / NumPy
- Plotly
- Streamlit

---

## Notes

- Requires `.TRC` files
- Data in `/data` is for testing

---

## Acknowledgment

Academic project focused on biomechanical analysis of shoulder movement.
