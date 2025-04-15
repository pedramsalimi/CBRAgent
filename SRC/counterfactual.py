import pandas as pd
import numpy as np
import os
from sklearn.preprocessing import MinMaxScaler, LabelEncoder
import joblib
import xgboost as xgb

CATEGORICAL_FEATURES = ['person_home_ownership', 'loan_intent', 'loan_grade', 'cb_person_default_on_file']

FEATURES = [
    'person_age',
    'person_income',
    'person_emp_length',
    'loan_amnt',
    'loan_int_rate',
    'loan_percent_income',
    'cb_person_cred_hist_length',
    'person_home_ownership',
    'loan_intent',
    'loan_grade',
    'cb_person_default_on_file'
]

# Helper functions
def transform_ite_to_weights(ite_values, beta, fat_weights):
    """
    Transforms ITE values by applying beta scaling for negative values
    and multiplying by feature actionability weights.
    """
    adjusted_ite = np.where(ite_values < 0, beta * np.abs(ite_values), ite_values)
    adjusted_ite_with_fat = adjusted_ite * fat_weights
    return adjusted_ite_with_fat

def calculate_similarity_and_sparsity(query, nun, weight_q, weight_n):
    """
    Calculates similarity and sparsity between query and NUN instances.
    
    Parameters:
    - query (pd.Series): The query instance.
    - nun (pd.Series): The nearest unlike neighbor instance.
    - weight_q (np.ndarray): Weights for the query instance.
    - weight_n (np.ndarray): Weights for the NUN instance.
    
    Returns:
    - similarity (float): Calculated similarity score.
    - sparsity (int): Number of features that differ.
    """
    difference = np.abs(query - nun)
    weighted_difference = difference * weight_q * weight_n
    total_changes = np.sum(weighted_difference)
    similarity = 1 / (1 + total_changes)
    sparsity = np.count_nonzero(difference)
    return similarity, sparsity

def preprocess_instance(instance, features):
    """
    Preprocesses a single instance by encoding categorical variables and scaling numerical features.

    Parameters:
    - instance (dict): The data instance to preprocess.
    - features (list): Ordered list of feature names.

    Returns:
    - pd.DataFrame: Preprocessed data frame ready for prediction.
    """

    scaler_path = r'C:\Users\PS11810\Documents\Projects\loanApp\scaler.save'             # Scaler path
    encoders_save_path = r'C:\Users\PS11810\Documents\Projects\loanApp\encoders.save'   # Label encoders path
    label_encoders = joblib.load(encoders_save_path)
    scaler = joblib.load(scaler_path)
    # Reorder the instance to match feature order
    instance_ordered = {feature: instance.get(feature, 0) for feature in features}
    instance_df = pd.DataFrame([instance_ordered])

    # Identify categorical and numerical columns
    categorical_cols = [col for col in features if col in label_encoders]
    numerical_cols = [col for col in features if col not in categorical_cols]

    # Encode categorical variables
    for col in categorical_cols:
        le = label_encoders[col]
        if instance_df[col].iloc[0] in le.classes_:
            instance_df[col] = le.transform(instance_df[col])
        else:
            instance_df[col] = -1  # Handle unseen categories

    # Scale numerical features
    if numerical_cols:
        instance_df[numerical_cols] = scaler.transform(instance_df[numerical_cols])

    return instance_df

def inverse_preprocess_instance(instance_df, features):
    """
    Inverse preprocesses a single instance by decoding categorical variables and inverse scaling numerical features.

    Parameters:
    - instance_df (pd.DataFrame): The data instance to inverse preprocess.
    - features (list): Ordered list of feature names.

    Returns:
    - dict: Inverse preprocessed data as a dictionary.
    """

    scaler_path = r'C:\Users\PS11810\Documents\Projects\loanApp\scaler.save'             # Scaler path
    encoders_save_path = r'C:\Users\PS11810\Documents\Projects\loanApp\encoders.save'   # Label encoders path
    label_encoders = joblib.load(encoders_save_path)
    scaler = joblib.load(scaler_path)

    # Identify categorical and numerical columns
    categorical_cols = [col for col in features if col in label_encoders]
    numerical_cols = [col for col in features if col not in categorical_cols]

    # Inverse scale numerical features
    if numerical_cols:
        instance_df[numerical_cols] = scaler.inverse_transform(instance_df[numerical_cols])

    # Decode categorical variables
    for col in categorical_cols:
        le = label_encoders[col]
        instance_df[col] = le.inverse_transform(instance_df[col].astype(int))

    # Convert DataFrame to dictionary
    instance_dict = instance_df.iloc[0].to_dict()

    return instance_dict

def preprocess_data(data, features):
    """
    Preprocesses the data by encoding categorical variables and scaling numerical features.

    Parameters:
    - data (pd.DataFrame): The data to preprocess.
    - features (list): Ordered list of feature names.

    Returns:
    - pd.DataFrame: Preprocessed data frame.
    """
    scaler_path = r'C:\Users\PS11810\Documents\Projects\loanApp\scaler.save'             # Scaler path
    encoders_save_path = r'C:\Users\PS11810\Documents\Projects\loanApp\encoders.save'   # Label encoders path
    label_encoders = joblib.load(encoders_save_path)
    scaler = joblib.load(scaler_path)

    # Identify categorical and numerical columns
    categorical_cols = [col for col in features if col in label_encoders]
    numerical_cols = [col for col in features if col not in categorical_cols]

    # Encode categorical variables
    for col in categorical_cols:
        le = label_encoders[col]
        data[col] = data[col].apply(lambda x: le.transform([x])[0] if x in le.classes_ else -1)

    # Scale numerical features
    if numerical_cols:
        data[numerical_cols] = scaler.transform(data[numerical_cols])

    return data

def calculate_validity(modified_query, model, features, preprocessed=False):
    """
    Validates if the modified query instance meets the desired outcome.

    Parameters:
    - modified_query (dict or pd.Series): The modified query instance.
    - model: Trained XGBoost model.
    - features (list): Ordered list of feature names.
    - preprocessed (bool): Indicates whether modified_query is already preprocessed.

    Returns:
    - bool: True if the prediction meets the desired outcome, False otherwise.
    """
    if not preprocessed:
        # Preprocess the modified query
        modified_query_df = preprocess_instance(modified_query, features)
    else:
        # Assume modified_query is already a DataFrame or Series
        modified_query_df = pd.DataFrame([modified_query])
        modified_query_df = modified_query_df[features]

    # Predict probability for the positive class
    probabilities = model.predict_proba(modified_query_df)[:, 1]
    # Round the probability to get the predicted label
    predicted_label = np.round(probabilities)
    return predicted_label[0] == 1

def calculate_sparsity_and_proximity(original, modified):
    """
    Calculates sparsity and proximity between original and modified instances.
    
    Parameters:
    - original (dict): Original instance.
    - modified (dict): Modified instance.
    
    Returns:
    - sparsity (int): Number of features that differ.
    - proximity (float): Sum of absolute differences across features.
    """
    changes = np.abs(np.array(list(original.values())) - np.array(list(modified.values())))
    sparsity = np.count_nonzero(changes)
    proximity = np.sum(changes)
    return sparsity, proximity

# Function to find the nearest unlike neighbor (NUN)
def causal_nun(test_data, train_data, ite_test_data, ite_train_data, query_index, cf_label, feature_weights, beta=3, features=None):
    """
    Identifies the nearest unlike neighbor (NUN) for a given query instance.
    
    Parameters:
    - test_data (pd.DataFrame): Test dataset.
    - train_data (pd.DataFrame): Training dataset.
    - ite_test_data (pd.DataFrame): ITE values for test dataset.
    - ite_train_data (pd.DataFrame): ITE values for training dataset.
    - query_index (int): Index of the query instance.
    - cf_label (int): Desired label for the counterfactual.
    - feature_weights (dict): Weights for each feature based on actionability.
    - beta (float): Scaling factor for negative ITE values.
    - features (list): Ordered list of feature names.
    
    Returns:
    - pd.DataFrame: DataFrame containing the selected NUN instance and similarity score.
    """
    query_instance = test_data.iloc[query_index].drop('loan_status', errors='ignore')
    query_ite_values = ite_test_data.iloc[query_index].to_numpy()

    query_features = query_instance.index
    fat_weights = np.array([feature_weights.get(feature, 1.0) for feature in query_features])  # Default weight if feature not found

    weights_query = transform_ite_to_weights(query_ite_values, beta, fat_weights)

    potential_nuns = train_data[train_data['loan_status'] == cf_label]

    best_nun = None
    best_similarity = -1

    for index, row in potential_nuns.iterrows():
        nun_instance = row.drop('loan_status', errors='ignore')  
        # print(nun_instance)
        nun_ite_values = ite_train_data.iloc[index].to_numpy()

        weights_nun = transform_ite_to_weights(nun_ite_values, beta, fat_weights)

        similarity, _ = calculate_similarity_and_sparsity(query_instance, nun_instance, weights_query, weights_nun)

        if similarity > best_similarity:
            best_similarity = similarity
            best_nun = nun_instance  


    

    if best_nun is not None:
        nuns_df = pd.DataFrame([best_nun]).reset_index(drop=True)
        if features:
            nuns_df = nuns_df[features]
        nuns_df['Similarity'] = [best_similarity]
        return nuns_df
    else:
        print("\n[causal_nun] No NUN found.")
        return pd.DataFrame()

def copy_values_to_children(query_instance, feature, nun_instance, causal_relationships, updated_features, cat_features, ite_values, updated_feature_list):
    """
    Recursively copies values from NUN to the query instance based on causal relationships.
    
    Parameters:
    - query_instance (dict): Current state of the query instance.
    - feature (str): Feature to modify.
    - nun_instance (dict): NUN instance.
    - causal_relationships (dict): Causal relationships among features.
    - updated_features (set): Set of already updated features.
    - cat_features (list): List of categorical features.
    - ite_values (dict): ITE values.
    - updated_feature_list (list): List of features that have been updated.
    
    Returns:
    - tuple: Updated query_instance, proximity, and sparsity.
    """
    proximity = 0
    sparsity = 0

    def update_children(feature, proximity, sparsity):
        if feature in causal_relationships:
            for child in causal_relationships[feature]:
                if ite_values.get(child, 0) == 0:
                    continue

                if child not in updated_features and query_instance[child] != nun_instance[child]:
                    if child in cat_features:
                        proximity += 1
                    else:
                        proximity += np.abs(query_instance[child] - nun_instance[child])
                    query_instance[child] = nun_instance[child]
                    sparsity += 1
                    updated_features.add(child)
                    updated_feature_list.append(child)
                    proximity, sparsity = update_children(child, proximity, sparsity)

        return proximity, sparsity

    if ite_values.get(feature, 0) == 0:
        return query_instance, proximity, sparsity

    if query_instance[feature] != nun_instance[feature] and feature not in updated_features:
        if feature in cat_features:
            proximity += 1
        else:
            proximity += np.abs(query_instance[feature] - nun_instance[feature])
        sparsity += 1
        query_instance[feature] = nun_instance[feature]
        updated_features.add(feature)
        updated_feature_list.append(feature)
        proximity, sparsity = update_children(feature, proximity, sparsity)

    return query_instance, proximity, sparsity

def copy_values(query_instance, feature, nun_instance, updated_features, cat_features, updated_feature_list):
    """
    Copies value from NUN to query instance for a specific feature.
    
    Parameters:
    - query_instance (dict): Current state of the query instance.
    - feature (str): Feature to modify.
    - nun_instance (dict): NUN instance.
    - updated_features (set): Set of already updated features.
    - cat_features (list): List of categorical features.
    - updated_feature_list (list): List of features that have been updated.
    
    Returns:
    - tuple: Updated query_instance, proximity, and sparsity.
    """
    proximity = 0
    sparsity = 0
    if query_instance[feature] != nun_instance[feature] and feature not in updated_features:
        if feature in cat_features:
            proximity += 1
        else:
            proximity += np.abs(query_instance[feature] - nun_instance[feature])
        sparsity += 1
        query_instance[feature] = nun_instance[feature]
        updated_features.add(feature)
        updated_feature_list.append(feature)
    return query_instance, proximity, sparsity

def sort_features_by_actionability(fat):
    """
    Sorts features based on their Feature Actionability Type (FAT).
    
    Parameters:
    - fat (dict): Feature Actionability Types.
    
    Returns:
    - list: Sorted list of features.
    """

    actionability_order = {"DM": 1, "IM": 2, "NSI": 3, "SI": 4}
    sorted_features = sorted(fat.items(), key=lambda x: actionability_order.get(x[1], 5))
    return [feature for feature, _ in sorted_features]

def generate_counterfactual(query_instance, ite_values, constraints):
    """
    Generates a counterfactual instance based on the query and NUN instances.

    Parameters:
    - query_instance (dict): Query instance.
    - ite_values (dict): ITE values for the query instance.
    - constraints (list): List of constraints.

    Returns:
    - dict: Counterfactual instance in original units.
    """
    cat_features = ['person_home_ownership', 'loan_intent', 'loan_grade', 'cb_person_default_on_file']

    features = [
        'person_age',
        'person_income',
        'person_emp_length',
        'loan_amnt',
        'loan_int_rate',
        'loan_percent_income',
        'cb_person_cred_hist_length',
        'person_home_ownership',
        'loan_intent',
        'loan_grade',
        'cb_person_default_on_file'
    ]
    FAT = {
        "person_age": "NSI",
        "person_income": "DM",
        "person_home_ownership": "IM",
        "person_emp_length": "IM",
        "loan_intent": "IM",
        "loan_grade": "DM",
        "loan_amnt": "DM",
        "loan_int_rate": "DM",
        "loan_percent_income": "DM",
        "cb_person_cred_hist_length": "IM",
        "cb_person_default_on_file": "IM"  
    }
    causal_relationships = {
        'person_age': ['person_income', 'loan_amnt', 'loan_status'],
        'person_home_ownership': ['loan_int_rate', 'loan_status'],
        'person_emp_length': ['loan_status'],
        'loan_intent': ['loan_status'],
        'loan_grade': ['person_home_ownership', 'loan_int_rate'],
        'loan_amnt': ['loan_status'],
        'loan_int_rate': ['person_emp_length'],
        'loan_percent_income': ['loan_grade', 'cb_person_cred_hist_length'],
        'cb_person_default_on_file': [ 'person_income', 'loan_amnt', 'loan_int_rate'],
        'cb_person_cred_hist_length': ['loan_grade']
    }

    for f in constraints:
        FAT[f] = 'NSI'
    print(FAT)
    for c in constraints:
        for r in causal_relationships.keys():
            if c in causal_relationships[r]:
                causal_relationships[r].remove(c)

    train_data_path = r'C:\Users\PS11810\Documents\Projects\loanApp\credit_train_fold_1.csv'
    test_data_path = r'C:\Users\PS11810\Documents\Projects\loanApp\credit_test_fold_1.csv'
    ite_train_data_path = r'C:\Users\PS11810\Documents\Projects\loanApp\ite_credit_fold_1.csv'
    ite_test_data_path = r'C:\Users\PS11810\Documents\Projects\loanApp\ite_credit_test_fold_1.csv'
    model_path = r'C:\Users\PS11810\Documents\Projects\loanApp\xgb_predictor_model.pkl'  # XGBoost model path
    scaler_path = r'C:\Users\PS11810\Documents\Projects\loanApp\scaler.save'             # Scaler path
    encoders_save_path = r'C:\Users\PS11810\Documents\Projects\loanApp\encoders.save'   # Label encoders path

    for path in [train_data_path, test_data_path, ite_train_data_path, ite_test_data_path, model_path, scaler_path, encoders_save_path]:
        if not os.path.exists(path):
            raise FileNotFoundError(f"File not found: {path}")

    # Load label encoders and scaler before defining helper functions
    label_encoders = joblib.load(encoders_save_path)
    scaler = joblib.load(scaler_path)
    model = joblib.load(model_path) 

    # Load data
    train_data = pd.read_csv(train_data_path)
    test_data = pd.read_csv(test_data_path)
    ite_train_data = pd.read_csv(ite_train_data_path)
    ite_test_data = pd.read_csv(ite_test_data_path)

    weights = {
        "DM": 0.01,
        "IM": 0.1,
        "NSI": 1.0,
        "SI": 1.0
    }
    target_variable = "loan_status"

    feature_weights = {feature: weights[FAT[feature]] for feature in FAT.keys()}

    data_features = features + [target_variable]
    train_data = train_data[data_features]
    test_data = test_data[data_features]

    train_features = preprocess_data(train_data[features].copy(), features)
    train_data = pd.concat([train_features, train_data[[target_variable]].reset_index(drop=True)], axis=1)

    test_features = preprocess_data(test_data[features].copy(), features)
    test_data = pd.concat([test_features, test_data[[target_variable]].reset_index(drop=True)], axis=1)

    ite_train_data = ite_train_data[features]
    ite_test_data = ite_test_data[features]

    query_instance_df = pd.DataFrame([query_instance])
    query_instance_df = query_instance_df[features]
    query_instance_preprocessed = preprocess_instance(query_instance, features)
    query_instance_scaled = query_instance_preprocessed.iloc[0]

    ite_values_series = pd.Series(ite_values)
    ite_values_series = ite_values_series[features]

    nuns_with_scores_df = causal_nun(
        test_data=pd.DataFrame([query_instance_scaled]),
        train_data=train_data,
        ite_test_data=pd.DataFrame([ite_values_series]),
        ite_train_data=ite_train_data,
        query_index=0,  # Assuming single instance
        cf_label=1,
        feature_weights=feature_weights,
        beta=3,
        features=features  # Pass the features list to ensure correct ordering
    )
    if nuns_with_scores_df.empty:
        return "No counterfactual found."

    nun_instance = nuns_with_scores_df.iloc[0][features].to_dict()
    # Get the index of the selected NUN
    nun_index = nuns_with_scores_df.index[0]
    nun_ite_values = ite_train_data.iloc[nun_index].to_dict()

    sorted_features = sorted(ite_values.items(), key=lambda x: -np.abs(x[1]))  # Sort by absolute ITE values
    modified_instance = query_instance_scaled.copy()
    updated_features = set()
    updated_feature_list = []
    sparsity = 0
    proximity = 0

    sorted_features_by_actionability = sort_features_by_actionability(FAT)
    
    # for f in constraints:
    #     FAT[f] = 'NSI'
    # print(FAT)
    # for c in constraints:
    #     for r in causal_relationships.keys():
    #         if c in causal_relationships[r]:
    #             causal_relationships[r].remove(c)

    print(causal_relationships)
    # First phase: Try modifying features based on ITE values and causal relationships
    for feature, _ in sorted_features:
        if feature not in modified_instance or FAT.get(feature) in ['SI', 'NSI']:
            continue
        modified_instance, new_proximity, new_sparsity = copy_values_to_children(
            modified_instance, feature, nun_instance, causal_relationships, updated_features, cat_features, ite_values, updated_feature_list
        )
        sparsity += new_sparsity
        proximity += new_proximity

        # Validate counterfactual
        if calculate_validity(modified_instance, model, features, preprocessed=True):
            if sparsity == 0:
                sparsity += 0.0001
            proximity = proximity / sparsity
            # Before returning, inverse preprocess the modified instance
            modified_instance_df = pd.DataFrame([modified_instance])
            modified_instance_df = modified_instance_df[features]
            modified_instance_original = inverse_preprocess_instance(modified_instance_df, features)
            return modified_instance_original

    # Second phase: Follow FAT feature order, avoiding immutable features
    for feature in sorted_features_by_actionability:
        if feature in updated_features or feature not in modified_instance or FAT.get(feature) in ['SI', 'NSI']:
            continue

        modified_instance, new_proximity, new_sparsity = copy_values(
            modified_instance, feature, nun_instance, updated_features, cat_features, updated_feature_list
        )
        sparsity += new_sparsity
        proximity += new_proximity

        # Validate counterfactual
        if calculate_validity(modified_instance, model, features, preprocessed=True):
            if sparsity == 0:
                sparsity += 0.0001
            proximity = proximity / sparsity
            modified_instance_df = pd.DataFrame([modified_instance])
            modified_instance_df = modified_instance_df[features]
            modified_instance_original = inverse_preprocess_instance(modified_instance_df, features)
            return modified_instance_original

    return None  # No counterfactual found


# def predict_loan_approval(user_data, model, scaler, label_encoders, features):
def predict_loan_approval(user_data):

    """
    Predicts loan approval based on user data.
    
    Parameters:
    - user_data (dict): Dictionary containing user features.
    - model: Trained XGBoost model.
    - scaler: Scaler used for numerical features.
    - label_encoders (dict): Dictionary of label encoders for categorical features.
    - features (list): Ordered list of feature names.
    
    Returns:
    - str: "Approved" or "Not Approved"
    """

    model_path = r'C:\Users\PS11810\Documents\Projects\loanApp\xgb_predictor_model.pkl'  
    scaler_path = r'C:\Users\PS11810\Documents\Projects\loanApp\scaler.save'             
    encoders_save_path = r'C:\Users\PS11810\Documents\Projects\loanApp\encoders.save'   
    label_encoders = joblib.load(encoders_save_path)
    scaler = joblib.load(scaler_path)
    model = joblib.load(model_path) 

    features = [
    'person_age',
    'person_income',
    'person_emp_length',
    'loan_amnt',
    'loan_int_rate',
    'loan_percent_income',
    'cb_person_cred_hist_length',
    'person_home_ownership',
    'loan_intent',
    'loan_grade',
    'cb_person_default_on_file'
   ]
    target_variable = 'loan_status'

    missing_features = set(features) - set(user_data.keys())
    if missing_features:
        raise ValueError(f"Missing features in user data: {missing_features}")

    user_processed = preprocess_instance(user_data, features)

    # print("\n[predict_loan_approval] User data after preprocessing:")
    # print(user_processed)
    # print("Feature order in processed user data:", list(user_processed.columns))

    try:
        prediction_proba = model.predict_proba(user_processed)[0][1]
        prediction = model.predict(user_processed)[0]
    except ValueError as e:
        print("[predict_loan_approval] Error during prediction:", e)
        raise

    print(f"[predict_loan_approval] Prediction probability: {prediction_proba}")
    print(f"[predict_loan_approval] Predicted label: {prediction}")

    return "Approved" if prediction == 1 else "Not Approved"

if __name__ == '__main__':

    query_data = {
        'person_age': 27.0,
        'person_income': 62900.0,
        'person_home_ownership': 'RENT',  
        'person_emp_length': 9.0,
        'loan_intent': 'EDUCATION',       
        'loan_grade': 'C',                
        'loan_amnt': 7450.0,
        'loan_int_rate': 15.99,
        'loan_status': 1.0,
        'loan_percent_income': 0.1,
        'cb_person_default_on_file': 'N', 
        'cb_person_cred_hist_length': 10.0
    }
    selected_ite_values = {
        'person_age': -0.0025391578674316,
        'person_income': 0.0,
        'person_home_ownership': 2.240622043609619,
        'person_emp_length': 0.0952982902526855,
        'loan_intent': 0.680086612701416,
        'loan_grade': 0.304718017578125,
        'loan_amnt': -0.5441474914550781,
        'loan_int_rate': 2.384185791015625e-06,
        'loan_percent_income': 2.384185791015625e-06,
        'cb_person_default_on_file': 0.0001311302185058,
        'cb_person_cred_hist_length': 0.0054240226745605
    }

    constraints = ["person_age", "loan_amnt"]  

    original_counterfactual = generate_counterfactual(
        query_instance=query_data,
        ite_values=selected_ite_values,
        constraints=constraints
    )
    print("Query:")
    print(query_data)
    print("Generated Counterfactual:")
    print(original_counterfactual)
