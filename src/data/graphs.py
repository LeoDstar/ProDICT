###Dependencies###
import os
import plotly.express as px
import umap

from sklearn.preprocessing import StandardScaler
from entity_model_settings import run_folder_name, target_class_name
import warnings
warnings.filterwarnings('ignore')

### Paths ###
project_root = os.path.abspath(os.getcwd())
output_dir = os.path.join(project_root, 'data', 'data_output', run_folder_name, 'model_fit')
os.makedirs(output_dir, exist_ok=True)

###Functions###
def create_umap_plot(df, feature_columns, n_components=3, color_column='code_oncotree', 
                       metadata_cols=['Sample name', 'code_oncotree', 'TCC'],
                       n_neighbors=15, min_dist=0.1,
                       title=target_class_name):
    """
    Create a 3D UMAP visualization with Plotly
    
    Parameters:
        df : pandas.DataFrame
            Input dataframe
        feature_columns : list
            List of column names to use as features for UMAP
        color_column : str
            Column name to use for coloring points (default: 'code_oncotree')
        hover_columns : list
            Column names to show on hover (default: ['Sample name', 'code_oncotree', 'TCC'])
        n_neighbors : int
            UMAP parameter for number of neighbors (default: 15)
        min_dist : float
            UMAP parameter for minimum distance (default: 0.1)
        title : str
            Plot title (default: "3D UMAP Visualization")
        
    Returns:

    plotly.graph_objects.Figure
        The 3D UMAP plot figure
    """
    
     
    # Filter and prepare data
    print(f"Original dataframe shape: {df.shape}")
    
    # Select feature columns
    feature_data = df[feature_columns].copy()
    print(f"Feature data shape: {feature_data.shape}")
    
    
    # Standardize features
    print("Standardizing features...")
    scaler = StandardScaler()
    feature_data_scaled = scaler.fit_transform(feature_data)
    
    # Apply UMAP
    print("Applying UMAP...")
    umap_model = umap.UMAP(
        n_components=n_components,
        n_neighbors=n_neighbors,
        min_dist=min_dist,
        random_state=93,
        verbose=True
    )
    
    embedding_3d = umap_model.fit_transform(feature_data_scaled)
    
    # Prepare data for plotting
    df_plot = df.copy()
    df_plot['UMAP_1'] = embedding_3d[:, 0]
    df_plot['UMAP_2'] = embedding_3d[:, 1]
    df_plot['UMAP_3'] = embedding_3d[:, 2]
    df_plot[metadata_cols] = df_plot[metadata_cols]

    
    
    # Get unique colors for each category
    unique_categories = df_plot[color_column].unique()
    n_categories = len(unique_categories)
    print(f"Number of unique categories in {color_column}: {n_categories}")
    
    fig = px.scatter_3d(
        df_plot,
        x='UMAP_1',
        y='UMAP_2',
        z='UMAP_3',
        color=color_column,
        hover_data={col: True for col in metadata_cols},
        title=target_class_name+'_UMAP Visualization',
        opacity=0.7
    )
    
    # Update hover template for cleaner display
    fig.update_traces(
        hovertemplate='<br>'.join([f'{col}: %{{customdata[{i}]}}' 
                                    for i, col in enumerate(metadata_cols)]) + '<extra></extra>'
    )

    # Update layout
    fig.update_layout(
        title={
            'text': title,
            'x': 0.5,
            'xanchor': 'center',
            'font': {'size': 14}
        },
        scene=dict(
            xaxis_title="UMAP 1",
            yaxis_title="UMAP 2",
            zaxis_title="UMAP 3",
            camera=dict(
                eye=dict(x=1.5, y=1.5, z=1.5)
            )
        ),
        width=900,
        height=700,
        margin=dict(l=0, r=0, b=0, t=50)
    )


    file_name = os.path.join(output_dir,f'UMAP_of_class.html')   
    fig.write_html(file_name, include_plotlyjs=True)
    
    print(f"Plot saved as: {file_name}")
    return fig


