import dash
from dash import dcc, html
from dash.dependencies import Input, Output
import plotly.graph_objects as go
import pandas as pd
import umap
import os
import PIL.Image
from PIL import Image
from sklearn.preprocessing import StandardScaler
import matplotlib.pyplot as plt
import plotly.express as px
import os
import gunicorn
import zipfile


# Path to the zip file
zip_file_path = 'assets.zip'  # Replace with the correct path to your zip file
assets_dir = 'assets'  # This is where you want to extract the files

# Check if the assets directory already exists (to avoid re-extraction every time)
if not os.path.exists(assets_dir):
    os.makedirs(assets_dir)

# Extract the ZIP file if not already extracted
if zip_file_path.endswith('.zip'):
    with zipfile.ZipFile(zip_file_path, 'r') as zip_ref:
        zip_ref.extractall(assets_dir)  # Extract files into the 'assets' folder
    print(f"Extracted {zip_file_path} to {assets_dir}")
else:
    print(f"{zip_file_path} is not a zip file.")



# Load feature data
features_no_scale = pd.read_csv("feature_table_no_scale_1203.csv")
features_scale = pd.read_csv("feature_table_scaled_1203.csv")



# PLOT 1: UMAP Zernike Moments (not scaled)
X_zernike_ns = features_no_scale.drop(columns=['z_moment_6', 'z_moment_12', 'hu_moment_1', 'hu_moment_2', 'hu_moment_3',
                                               'hu_moment_4', 'hu_moment_5', 'hu_moment_6', 'hu_moment_7', 'Image_Name', 
                                               'year', 'genus', 'length', 'family', 'order', 'quality', 'method'])

scaler = StandardScaler()
X_zernike_ns_normalized = scaler.fit_transform(X_zernike_ns)

# PLOT 2: tsne Hu Moments (not scaled)
X_hu_ns = features_no_scale.drop(columns=['z_moment_1', 'z_moment_2', 'z_moment_3', 'z_moment_4', 'z_moment_5', 
                                           'z_moment_6', 'z_moment_7', 'z_moment_8', 'z_moment_9', 'z_moment_10', 
                                           'z_moment_11', 'z_moment_12', 'z_moment_13', 'z_moment_14', 'z_moment_15', 
                                           'z_moment_16', 'z_moment_17', 'z_moment_18', 'z_moment_19', 'z_moment_20', 
                                           'z_moment_21', 'z_moment_22', 'z_moment_23', 'z_moment_24', 'z_moment_25', 
                                           'hu_moment_7', 'Image_Name', 'year', 'genus', 'length', 'family', 'order', 
                                           'quality', 'method'])

X_hu_ns_normalized = scaler.fit_transform(X_hu_ns)

# PLOT 3: Hu + Zernike (not scaled)
X_ns = features_no_scale.drop(columns=['z_moment_6', 'z_moment_12', 'hu_moment_7', 'Image_Name', 'year', 'genus', 
                                       'length', 'family', 'order', 'quality', 'method'])

X_ns_normalized = scaler.fit_transform(X_ns)


#PLOT 4: UMAP Zernike Moments (scaled)

X_zernike_s = features_scale.drop(columns=['original_shape2D_Elongation', 'original_shape2D_MajorAxisLength',
       'original_shape2D_MaximumDiameter', 'original_shape2D_MeshSurface',
       'original_shape2D_MinorAxisLength', 'original_shape2D_Perimeter',
       'original_shape2D_PerimeterSurfaceRatio',
       'original_shape2D_PixelSurface', 'original_shape2D_Sphericity','z_moment_6', 'z_moment_12', 'hu_moment_1',
       'hu_moment_2', 'hu_moment_3', 'hu_moment_4', 'hu_moment_5',
       'hu_moment_6', 'hu_moment_7', 'Image_Name', 'year', 'genus', 'length',
       'family', 'order', 'quality', 'method'])  # Drop non-numeric columns

X_zernike_s_normalized = scaler.fit_transform(X_zernike_s)


#PLOT 5: UMAP Hu Moments (scaled)

X_hu_s = features_scale.drop(columns=['original_shape2D_Elongation', 'original_shape2D_MajorAxisLength',
       'original_shape2D_MaximumDiameter', 'original_shape2D_MeshSurface',
       'original_shape2D_MinorAxisLength', 'original_shape2D_Perimeter',
       'original_shape2D_PerimeterSurfaceRatio',
       'original_shape2D_PixelSurface', 'original_shape2D_Sphericity','z_moment_1', 'z_moment_2', 'z_moment_3', 'z_moment_4', 'z_moment_5',
       'z_moment_6', 'z_moment_7', 'z_moment_8', 'z_moment_9', 'z_moment_10',
       'z_moment_11', 'z_moment_12', 'z_moment_13', 'z_moment_14',
       'z_moment_15', 'z_moment_16', 'z_moment_17', 'z_moment_18',
       'z_moment_19', 'z_moment_20', 'z_moment_21', 'z_moment_22',
       'z_moment_23', 'z_moment_24', 'z_moment_25','hu_moment_7', 'Image_Name', 'year', 'genus', 'length',
       'family', 'order', 'quality', 'method'])  # Drop non-numeric columns

X_hu_s_normalized = scaler.fit_transform(X_hu_s)


#PLOT 6: UMAP pyradiomics (scaled)

X_pyradiomics = features_scale.drop(columns=['z_moment_1', 'z_moment_2', 'z_moment_3', 'z_moment_4', 'z_moment_5',
       'z_moment_6', 'z_moment_7', 'z_moment_8', 'z_moment_9', 'z_moment_10',
       'z_moment_11', 'z_moment_12', 'z_moment_13', 'z_moment_14',
       'z_moment_15', 'z_moment_16', 'z_moment_17', 'z_moment_18',
       'z_moment_19', 'z_moment_20', 'z_moment_21', 'z_moment_22',
       'z_moment_23', 'z_moment_24', 'z_moment_25', 'hu_moment_1',
       'hu_moment_2', 'hu_moment_3', 'hu_moment_4', 'hu_moment_5',
       'hu_moment_6', 'hu_moment_7', 'Image_Name', 'year', 'genus', 'length',
       'family', 'order', 'quality', 'method'])  # Drop non-numeric columns

X_pyradiomics_normalized = scaler.fit_transform(X_pyradiomics)

#PLOT 7: UMAP all features (scaled)

X_all_s = features_scale.drop(columns=['z_moment_6', 'z_moment_12', 'hu_moment_7', 'Image_Name', 'year', 'genus', 'length',
       'family', 'order', 'quality', 'method'])  # Drop non-numeric columns

X_all_s_normalized = scaler.fit_transform(X_all_s)



# Function to create UMAP embeddings
def generate_umap(X_normalized):
    umap_reducer = umap.UMAP(n_components=2)
    embedding = umap_reducer.fit_transform(X_normalized)
    
    df_umap = pd.DataFrame(embedding, columns=['UMAP_1', 'UMAP_2'])
    df_umap['Image_Name'] = features_no_scale['Image_Name']
    df_umap = df_umap.merge(features_no_scale[['Image_Name', 'year', 'length', 'family', 'order', 'genus', 'quality', 'method']], 
                             on='Image_Name', how='left')
    return df_umap


# Generate UMAP projections only once
df_umap_1 = generate_umap(X_zernike_ns)
df_umap_2 = generate_umap(X_hu_ns)
df_umap_3 = generate_umap(X_ns)

df_umap_4 = generate_umap(X_zernike_s)
df_umap_5 = generate_umap(X_hu_s)
df_umap_6 = generate_umap(X_pyradiomics)
df_umap_7 = generate_umap(X_all_s)

# Store UMAP projections in a dictionary
umap_dict = {
    "UMAP - Zernike (not scaled)": df_umap_1,
    "UMAP - Hu (not scaled)": df_umap_2,
    "UMAP - Zernike + Hu (not scaled)": df_umap_3,
    "UMAP - Zernike (scaled)": df_umap_4,
    "UMAP - Hu (scaled)": df_umap_5,
    "UMAP - Pyradiomics (scaled)": df_umap_6,
    "UMAP - Zernike + Hu + pyradiomics (scaled)": df_umap_7
}

# Initialize Dash app
app = dash.Dash(__name__)
server = app.server 


# Get unique genera for the checklist and sort them alphabetically
unique_genera = sorted(df_umap_1['genus'].unique())  # Sort genera alphabetically

# Layout
app.layout = html.Div([
    html.H1("UMAP Projection with Images"),
    
    # Dropdown for selecting dataset
    dcc.Dropdown(
        id='dataset-selector',
        options=[{"label": k, "value": k} for k in umap_dict.keys()],
        value="UMAP - Zernike (not scaled)",
        clearable=False
    ),
    
    # Checklist for selecting genera (default all selected)
    dcc.Checklist(
        id='genus-selector',
        options=[{'label': genus, 'value': genus} for genus in unique_genera],
        value=unique_genera,  # Default: All genera are selected
        inline=True
    ),
    
    # UMAP Graph
    dcc.Graph(id='umap-graph')
])

# Callback to update UMAP plot based on selected dataset and genera
@app.callback(
    Output('umap-graph', 'figure'),
    Output('genus-selector', 'value'),
    Input('dataset-selector', 'value'),
    Input('genus-selector', 'value')
)
def update_umap(selected_dataset, selected_genera):
    df_umap = umap_dict[selected_dataset]

    # If a genus is selected, deselect all others and select only the clicked genus
    if len(selected_genera) == 1:
        selected_genera = [selected_genera[0]]  # Only keep the selected genus
    
    # Filter the dataframe based on the selected genera
    df_umap_filtered = df_umap[df_umap['genus'].isin(selected_genera)]

    fig = go.Figure()

    # Scatter plot
    fig.add_trace(go.Scatter(
        x=df_umap_filtered['UMAP_1'],
        y=df_umap_filtered['UMAP_2'],
        mode='markers',
        marker=dict(
            size=10,
            color=df_umap_filtered['genus'].astype('category').cat.codes,
            colorscale='Viridis'
        ),
        text=df_umap_filtered.apply(lambda row: f"Genus: {row['genus']}<br>Order: {row['order']}<br>Family: {row['family']}<br>Method: {row['method']}<br>Quality: {row['quality']}<br>Length: {row['length']}<br>image_name: {row['Image_Name']}", axis=1),
        hoverinfo="text"
    ))

    # Add images to the UMAP plot (only for selected genera)
    for _, row in df_umap_filtered.iterrows():
        img_path = f'assets/{row["Image_Name"]}'  # Adjust path based on your image directory
        if os.path.exists(img_path):
            fig.add_layout_image(
                dict(
                    source=Image.open(img_path),
                    x=row['UMAP_1'],
                    y=row['UMAP_2'],
                    xref="x",
                    yref="y",
                    sizex=1,
                    sizey=1,
                    xanchor="center",
                    yanchor="middle",
                    layer="above"
                )
            )

    fig.update_layout(
        title=f"UMAP Projection: {selected_dataset}",
        xaxis_title="UMAP Dimension 1",
        yaxis_title="UMAP Dimension 2",
        template="plotly_white",
        showlegend=False,
        height=800  # Taller height
    )

    return fig, selected_genera

if __name__ == '__main__':
    # Get the PORT from environment variables (Render automatically sets it)
    port = int(os.environ.get('PORT', 6056))  
    
    # Run the Dash app on host 0.0.0.0 
    app.run_server(debug=False, host='0.0.0.0', port=port)
 

  
