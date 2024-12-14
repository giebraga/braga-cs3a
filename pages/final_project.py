import streamlit as st
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_squared_error, r2_score
import scipy.stats as stats

class ModelingSimulationApp:
    def __init__(self):
        # Configure Streamlit page
        st.set_page_config(
            page_title="Modeling & Simulation Workflow", 
            page_icon="🔬",
            layout="wide"
        )
        # Initialize session state for data
        if 'generated_data' not in st.session_state:
            st.session_state.generated_data = None
        
    def introduction_page(self):
        """
        Project introduction and overview page
        """
        st.title("🧪 Modeling and Simulation Project")
        
        # Introduction section
        st.header("Project Introduction")
        st.markdown("""
        This interactive application demonstrates a comprehensive workflow for 
        **Modeling and Simulation using Python**. The goal is to provide 
        hands-on experience with powerful Python libraries and data science techniques.
        """)

         # Project steps overview
        st.header("Project Workflow Steps")
        steps = [
            "**Data Generation**: Create synthetic data mimicking real-world scenarios",
            "**Exploratory Data Analysis (EDA)**: Investigate data characteristics",
            "**Modeling**: Apply appropriate machine learning techniques",
            "**Simulation**: Generate predictive outcomes",
            "**Evaluation**: Assess model performance and reliability"
        ]
        
        for step in steps:
            st.markdown(f"- {step}")
        
        st.info("""
        🔍 This application will guide you through each step of the modeling 
        and simulation process, demonstrating key data science concepts and techniques.
        """)

    def data_generation_page(self):
        """
        Interactive data generation section
        """
        st.header("🔢 Data Generation")
        st.markdown("""
        Generate synthetic data with controllable properties to simulate 
        real-world scenarios. Customize data generation parameters below.
        """)
        
        # Sidebar configuration
        st.sidebar.header("Data Generation Parameters")
        
        # Feature configuration
        features = st.sidebar.multiselect(
            "Select Features", 
            ["Temperature", "Pressure", "Humidity", "Wind Speed", "Altitude"],
            default=["Temperature", "Pressure", "Humidity"]
        )
        
        # Sample size selection
        n_samples = st.sidebar.slider(
            "Number of Samples", 
            min_value=100, 
            max_value=10000, 
            value=1000
        )
        
        # Noise level
        noise_level = st.sidebar.slider(
            "Noise Level", 
            min_value=0.0, 
            max_value=1.0, 
            value=0.1, 
            step=0.01
        )
        
        # Generate data button
        if st.sidebar.button("Generate Data"):
            # Synthetic data generation
            data = self.generate_synthetic_data(
                features, 
                n_samples, 
                noise_level
            )
            
            # Store in session state
            st.session_state.generated_data = data
            
            # Display generated data
            st.subheader("Generated Synthetic Data")
            st.dataframe(data)
        
        # Display existing data if available
        if st.session_state.generated_data is not None:
            st.subheader("Current Dataset")
            st.dataframe(st.session_state.generated_data)
    
    def generate_synthetic_data(self, features, n_samples, noise_level):
        """
        Generate synthetic data with specified features
        """
        np.random.seed(42)
        
        # Create base data
        data = pd.DataFrame()
        
        # Generate correlated features
        for feature in features:
            if feature == "Temperature":
                data[feature] = np.random.normal(25, 5, n_samples)
            elif feature == "Pressure":
                data[feature] = data["Temperature"] * 0.5 + np.random.normal(1000, 50, n_samples)
            elif feature == "Humidity":
                data[feature] = np.clip(data["Temperature"] * 0.5 + np.random.normal(50, 10, n_samples), 0, 100)
            elif feature == "Wind Speed":
                data[feature] = np.abs(np.random.normal(5, 2, n_samples))
            elif feature == "Altitude":
                data[feature] = np.random.normal(500, 100, n_samples)
        
        # Add target variable with synthetic relationship
        data['Target'] = (
            data['Temperature'] * 0.5 + 
            data['Pressure'] * 0.3 + 
            data['Humidity'] * 0.2 + 
            np.random.normal(0, noise_level, n_samples)
        )
        
        return data
    
    def exploratory_analysis_page(self):
        """
        Perform Exploratory Data Analysis
        """
        st.header("📊 Exploratory Data Analysis")
        
        # Check if data is generated
        if st.session_state.generated_data is None:
            st.warning("Please generate data first in the Data Generation page.")
            return
        
        data = st.session_state.generated_data
        
        # Descriptive statistics
        st.subheader("Descriptive Statistics")
        st.dataframe(data.describe())
        
        # Correlation analysis
        st.subheader("Correlation Matrix")
        corr_matrix = data.corr()
        plt.figure(figsize=(10, 8))
        sns.heatmap(corr_matrix, annot=True, cmap='coolwarm', center=0)
        st.pyplot(plt)
        
        # Distribution plots
        st.subheader("Feature Distributions")
        numeric_cols = data.select_dtypes(include=[np.number]).columns
        fig, axes = plt.subplots(len(numeric_cols), 1, figsize=(10, 3*len(numeric_cols)))
        
        for i, col in enumerate(numeric_cols):
            sns.histplot(data[col], kde=True, ax=axes[i])
            axes[i].set_title(f'{col} Distribution')
        
        plt.tight_layout()
        st.pyplot(fig)
    
    def modeling_page(self):
        """
        Modeling and simulation section
        """
        st.header("🤖 Modeling and Simulation")
        
        # Check if data is generated
        if st.session_state.generated_data is None:
            st.warning("Please generate data first in the Data Generation page.")
            return
        
        data = st.session_state.generated_data
        
        # Prepare data for modeling
        X = data.drop('Target', axis=1)
        y = data['Target']
        
        # Train-test split
        X_train, X_test, y_train, y_test = train_test_split(
            X, y, test_size=0.2, random_state=42
        )
        
        # Scale the features
        scaler = StandardScaler()
        X_train_scaled = scaler.fit_transform(X_train)
        X_test_scaled = scaler.transform(X_test)
        
        # Train Random Forest Regressor
        rf_model = RandomForestRegressor(n_estimators=100, random_state=42)
        rf_model.fit(X_train_scaled, y_train)
        
        # Predictions
        y_pred = rf_model.predict(X_test_scaled)
        
        # Model evaluation
        st.subheader("Model Performance")
        mse = mean_squared_error(y_test, y_pred)
        r2 = r2_score(y_test, y_pred)
        
        col1, col2 = st.columns(2)
        with col1:
            st.metric("Mean Squared Error", f"{mse:.4f}")
        with col2:
            st.metric("R² Score", f"{r2:.4f}")
        
        # Feature importance
        st.subheader("Feature Importance")
        feature_importance = pd.DataFrame({
            'feature': X.columns,
            'importance': rf_model.feature_importances_
        }).sort_values('importance', ascending=False)
        
        plt.figure(figsize=(10, 6))
        sns.barplot(x='importance', y='feature', data=feature_importance)
        plt.title('Feature Importance in Predictive Model')
        st.pyplot(plt)
    
    def simulation_page(self):
        """
        Simulation and scenario analysis
        """
        st.header("🔮 Scenario Simulation")
        
        # Check if data is generated
        if st.session_state.generated_data is None:
            st.warning("Please generate data first in the Data Generation page.")
            return
        
        data = st.session_state.generated_data
        
        # Prepare data
        X = data.drop('Target', axis=1)
        y = data['Target']
        
        # Train model
        X_train, X_test, y_train, y_test = train_test_split(
            X, y, test_size=0.2, random_state=42
        )
        scaler = StandardScaler()
        X_train_scaled = scaler.fit_transform(X_train)
        X_test_scaled = scaler.transform(X_test)
        
        rf_model = RandomForestRegressor(n_estimators=100, random_state=42)
        rf_model.fit(X_train_scaled, y_train)
        
        # Scenario simulation
        st.subheader("What-If Scenario Analysis")
        
        # Interactive feature adjustment
        scenario_features = st.multiselect(
            "Select features to modify", 
            X.columns.tolist(),
            default=['Temperature']
        )
        
        # Scenario sliders
        scenario_data = X_test.copy()
        for feature in scenario_features:
            min_val, max_val = X[feature].min(), X[feature].max()
            scenario_value = st.slider(
                f"Modify {feature}", 
                min_value=float(min_val), 
                max_value=float(max_val), 
                value=float(X[feature].mean())
            )
            scenario_data[feature] = scenario_value
        
        # Scale scenario data
        scenario_scaled = scaler.transform(scenario_data)
        
        # Predict
        scenario_prediction = rf_model.predict(scenario_scaled)
        
        st.subheader("Simulation Results")
        st.metric("Predicted Target Value", f"{scenario_prediction[0]:.4f}")
    
    def conclusion_page(self):
        """
        Project conclusion and key takeaways
        """
        st.header("🏁 Project Conclusion")
        
        st.markdown("""
        ### Key Learnings from Modeling and Simulation Project
        
        1. **Data Generation**: Creating synthetic data with controllable properties
        2. **Exploratory Analysis**: Understanding data characteristics through visualization
        3. **Modeling**: Applying machine learning techniques for prediction
        4. **Simulation**: Exploring scenarios and understanding model behavior
        
        ### Next Steps and Recommendations
        - Experiment with different data generation techniques
        - Try various machine learning algorithms
        - Explore more complex modeling scenarios
        - Apply these techniques to real-world datasets
        """)
        
        st.info("""
        💡 **Continuous Learning**: 
        Modeling and simulation are powerful techniques that require practice and 
        continuous exploration. Keep experimenting and learning!
        """)
    
    def main(self):
        """
        Main application workflow
        """
        # Sidebar navigation
        pages = {
            "🏠 Introduction": self.introduction_page,
            "🔢 Data Generation": self.data_generation_page,
            "📊 Exploratory Analysis": self.exploratory_analysis_page,
            "🤖 Modeling": self.modeling_page,
            "🔮 Simulation": self.simulation_page,
            "🏁 Conclusion": self.conclusion_page
        }
        
        # Page selection
        st.sidebar.title("Navigation")
        selection = st.sidebar.radio("Go to", list(pages.keys()))
        
        # Run selected page
        pages[selection]()

# Run the application
if __name__ == "__main__":
    app = ModelingSimulationApp()
    app.main()