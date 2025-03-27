import pandas as pd
import numpy as np
import copy

def recursive_prediction(initial_df, l2c_transform_func, model, max_iterations=5):
    """
    Perform recursive predictions on a dataset using L2C transformation and a pre-trained classifier.
    
    Parameters:
    -----------
    initial_df : pandas.DataFrame
        The initial dataset before L2C transformation
    l2c_transform_func : function
        The L2C transformation function (L2C_transform)
    model : TabPFNClassifier
        Pre-trained TabPFNClassifier model
    max_iterations : int, optional
        Maximum number of recursive prediction iterations (default: 5)
    
    Returns:
    --------
    pandas.DataFrame
        DataFrame with recursive predictions
    """
    # Deep copy to avoid modifying the original dataset
    working_df = initial_df.copy()
    
    # Store prediction history
    prediction_history = []
    
    for iteration in range(max_iterations):
        # Apply L2C transformation
        l2c_df = l2c_transform_func(working_df)
        
        # Prepare prediction features (exclude only specific target columns)
        prediction_features = l2c_df.drop(columns=['target_DX', 'target_Adas', 'target_Ventricles'])
        
        # Make predictions
        predictions = model.predict(prediction_features)
        predictions_proba = model.predict_proba(prediction_features)
        
        # Create a copy of the L2C dataframe to update
        updated_df = l2c_df.copy()
        
        # Update predictions
        updated_df['predicted_DX'] = predictions
        updated_df['prediction_proba'] = list(predictions_proba.max(axis=1))
        
        # Store prediction results
        prediction_history.append(updated_df)
        
        # Prepare for next iteration: duplicate rows and increment month
        next_iteration_df = []
        
        for ptid, group in working_df.groupby('PTID'):
            # Duplicate the last row for each patient
            new_row = group.iloc[-1].copy()
            
            # Increment month
            new_row['Month_bl'] += 1
            
            # Predict the next DX based on the most recent prediction
            patient_predictions = updated_df[updated_df['PTID'] == ptid]
            new_row['DX'] = patient_predictions['predicted_DX'].iloc[-1]
            
            # Append the new row to the group
            group_with_new_row = pd.concat([group, pd.DataFrame([new_row])], ignore_index=True)
            next_iteration_df.append(group_with_new_row)
        
        # Combine patient groups
        working_df = pd.concat(next_iteration_df, ignore_index=True)
    
    return prediction_history

# Example usage would look like:
# prediction_results = recursive_prediction(
#     initial_df, 
#     L2C_transform,  # Your L2C transformation function
#     trained_model   # Your pre-trained TabPFNClassifier
# )