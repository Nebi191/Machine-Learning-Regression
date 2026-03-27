import pandas as pd
import numpy as np
import os
from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import OneHotEncoder, OrdinalEncoder

def get_preprocessor():
    
    # 1. Hierarchy Definitions (Ordinal Encodings)
    quality_cats = ['None', 'Po', 'Fa', 'TA', 'Gd', 'Ex']
    shape_cats = ['IR3', 'IR2', 'IR1', 'Reg']
    garage_cats = ['None', 'Unf', 'RFn', 'Fin']

    # 2. Targeted Feature Groups
    qual_cols = ['Exter Qual', 'Exter Cond', 'Bsmt Qual', 'Bsmt Cond', 'Heating QC', 
                 'Kitchen Qual', 'Fireplace Qu', 'Garage Qual', 'Garage Cond']
    shape_cols = ['Lot Shape']
    garage_cols = ['Garage Finish']
    nominal_col = ['Neighborhood', 'MS Zoning', 'Street', 'Alley', 'Land Contour', 
                    'Lot Config', 'Bldg Type', 'House Style']

    # 3. Architectural Assembly
    preprocessor = ColumnTransformer(
        transformers=[
            ('qual_ord', OrdinalEncoder(categories=[quality_cats] * len(qual_cols)), qual_cols),
            ('shape_ord', OrdinalEncoder(categories=[shape_cats]), shape_cols),
            ('garage_ord', OrdinalEncoder(categories=[garage_cats]), garage_cols),
            ('nom', OneHotEncoder(handle_unknown='ignore', sparse_output=False), nominal_col)
        ], 
        remainder='passthrough'
    )
    
    return preprocessor

def run_full_preprocessing(input_csv, output_csv):
    """
    Executes the entire cleaning and transformation pipeline.
    This demonstrates the transformation from raw data to model-ready features.
    """
    if not os.path.exists(input_csv):
        print(f"ERROR: Input file not found at {input_csv}")
        return

    # --- Loading & Imputation ---
    df = pd.read_csv(input_csv) 
    df = df[df['SalePrice'] <= 400000]

    # Filling Numeric voids with Median
    numeric_cols = df.select_dtypes(include=[np.number]).columns
    for col in numeric_cols:
        df[col] = df[col].fillna(df[col].median())

    # Filling Categorical voids with 'None'
    categorical_cols = df.select_dtypes(include=['object']).columns
    for col in categorical_cols:
        df[col] = df[col].fillna('None')

    # Neighborhood-based Lot Frontage Imputation
    df["Lot Frontage"] = df.groupby("Neighborhood")["Lot Frontage"].transform(
        lambda x: x.fillna(x.median())
    )

    df = df.fillna(0) # Final safety net

    # --- Transformation ---
    preprocessor = get_preprocessor()
    X_processed = preprocessor.fit_transform(df)
    
    # --- Feature Name Sanitization ---
    raw_names = preprocessor.get_feature_names_out()
    clean_names = [col.split('__')[-1] for col in raw_names]

    # --- Final Filtering ---
    df_final = pd.DataFrame(X_processed, columns=clean_names)
    
    # Force numeric conversion and drop artifacts
    for col in df_final.columns:
        df_final[col] = pd.to_numeric(df_final[col], errors='coerce')
    
    df_final = df_final.dropna(axis=1, how='all').fillna(0)

    # --- Persistence ---
    df_final.to_csv(output_csv, index=False)
    print(f"SUCCESS: Processed data saved to {output_csv}")
    print(f"Final shape: {df_final.shape}")

if __name__ == "__main__":
    # Logic for finding paths relative to script location
    script_dir = os.path.dirname(os.path.abspath(__file__))
    project_root = os.path.dirname(script_dir)
    
    input_path = os.path.join(project_root, 'data', 'AmesHousing.csv')
    output_path = os.path.join(project_root, 'Ames_Housing_Processed.csv')
    
    # Execute the build process
    run_full_preprocessing(input_path, output_path)