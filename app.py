import streamlit as st
import pandas as pd
import numpy as np
import plotly.express as px
from src.model import (
    FEATURE_COLUMNS,
    encode_features,
    predict_customer,
    customer_risk_level,
    customer_recommendations,
    explain_customer_lime,
    get_feature_importance,
    get_shap_values,
    evaluate_model,
    load_pipeline_from_pkl,
)


@st.cache_resource(show_spinner=False)
def load_pipeline():
    import os
    import dill
    pkl_path = 'model.pkl'
    if not os.path.exists(pkl_path):
        # Train and save if pkl doesn't exist (first run on any environment)
        from src.model import load_raw_data, build_pipeline
        raw = load_raw_data('data/Churn_Modelling.csv')
        pipeline = build_pipeline(raw)
        with open(pkl_path, 'wb') as f:
            dill.dump(pipeline, f)
        return pipeline
    with open(pkl_path, 'rb') as f:
        return dill.load(f)


def render_overview(raw: pd.DataFrame, pipeline: dict):
    st.header('Overview Dashboard')
    st.write('High-level insights into customer churn patterns and trends.')

    # Compute metrics
    churn_counts = raw['Exited'].value_counts()
    churn_counts.index = churn_counts.index.map({0: 'Stayed', 1: 'Churned'})
    churn_rate = churn_counts.loc['Churned'] / churn_counts.sum()
    avg_tenure = raw['Tenure'].mean()
    avg_salary = raw['EstimatedSalary'].mean()
    high_risk = raw[
        (raw['Tenure'] <= 12)
        & (raw['Balance'] > raw['Balance'].median())
        & (raw['EstimatedSalary'] > raw['EstimatedSalary'].median())
    ].shape[0]

    # Metrics row
    col1, col2, col3, col4 = st.columns(4)
    col1.metric('Total customers', raw.shape[0])
    col2.metric('Churn Rate', f'{churn_rate:.1%}')
    col3.metric('Avg Tenure (months)', f'{avg_tenure:.1f}')
    col4.metric('High-risk customers', high_risk)

    st.markdown('---')

    # Churn distribution
    st.subheader('Churn Distribution')
    churn_df = churn_counts.reset_index()
    churn_df.columns = ['Status', 'Count']
    fig_dist = px.bar(
        churn_df,
        x='Status',
        y='Count',
        color='Status',
        labels={'Status': 'Churn Status', 'Count': 'Number of Customers'},
        color_discrete_map={'Stayed': '#2ecc71', 'Churned': '#e74c3c'},
    )
    st.plotly_chart(fig_dist, width='stretch')

    # Churn by geography
    st.subheader('Churn by Geography')
    raw_copy = raw.copy()
    raw_copy['Exited_Label'] = raw_copy['Exited'].map({0: 'Stayed', 1: 'Churned'})
    fig_geo = px.histogram(
        raw_copy,
        x='Geography',
        color='Exited_Label',
        barmode='group',
        labels={'Exited_Label': 'Churn Status'},
        color_discrete_map={'Stayed': '#2ecc71', 'Churned': '#e74c3c'},
    )
    st.plotly_chart(fig_geo, width='stretch')

    # Two-column layout for tenure and salary charts
    col1, col2 = st.columns(2)

    with col1:
        st.subheader('Tenure and Churn')
        raw_copy = raw.copy()
        raw_copy['Exited_Label'] = raw_copy['Exited'].map({0: 'Stayed', 1: 'Churned'})
        fig_tenure = px.histogram(
            raw_copy,
            x='Tenure',
            color='Exited_Label',
            nbins=20,
            barmode='overlay',
            opacity=0.75,
            labels={'Exited_Label': 'Churn Status'},
            color_discrete_map={'Stayed': '#2ecc71', 'Churned': '#e74c3c'},
        )
        st.plotly_chart(fig_tenure, width='stretch')

    with col2:
        st.subheader('Estimated Salary vs Churn')
        raw_copy = raw.copy()
        raw_copy['Exited_Label'] = raw_copy['Exited'].map({0: 'Stayed', 1: 'Churned'})
        fig_salary = px.box(
            raw_copy,
            x='Exited_Label',
            y='EstimatedSalary',
            color='Exited_Label',
            labels={'Exited_Label': 'Churn Status', 'EstimatedSalary': 'Estimated Salary ($)'},
            color_discrete_map={'Stayed': '#2ecc71', 'Churned': '#e74c3c'},
        )
        st.plotly_chart(fig_salary, width='stretch')

    # Feature correlation heatmap
    st.subheader('Feature Correlation Heatmap')
    numeric_features = raw[FEATURE_COLUMNS].select_dtypes(include=[np.number]).columns.tolist() + ['Exited']
    corr = raw[numeric_features].corr()
    fig_corr = px.imshow(
        corr,
        text_auto='.2f',
        aspect='auto',
        color_continuous_scale='RdBu_r',
        labels={'color': 'Correlation'},
    )
    st.plotly_chart(fig_corr, width='stretch')


def render_prediction(raw: pd.DataFrame, pipeline: dict):
    st.header('Customer Prediction')
    st.write('Enter customer attributes to score churn risk and receive model-backed recommendations.')

    with st.form('customer_form'):
        col1, col2 = st.columns(2)
        
        # Left column
        credit_score = col1.number_input('Credit score', min_value=300, max_value=850, value=650)
        geography = col1.selectbox('Geography', ['France', 'Germany', 'Spain'], index=0)
        gender = col1.selectbox('Gender', ['Female', 'Male'], index=1)
        age = col1.number_input('Age', min_value=18, max_value=100, value=42)
        tenure = col1.number_input('Tenure (months)', min_value=0, max_value=72, value=5)
        
        # Right column
        balance = col2.number_input('Balance ($)', min_value=0.0, value=83840.0, step=1000.0, format='%0.2f')
        products = col2.selectbox('Number of products', [1, 2, 3, 4], index=1)
        has_card = col2.selectbox('Has credit card', ['Yes', 'No'], index=0)
        active_member = col2.selectbox('Is active member', ['Yes', 'No'], index=0)
        salary = col2.number_input('Estimated salary ($)', min_value=0.0, value=112542.0, step=1000.0, format='%0.2f')
        
        submitted = st.form_submit_button('🔮 Predict Churn', width='stretch')

    if submitted:
        # Create user input row
        user_row = {
            'CreditScore': credit_score,
            'Geography': geography,
            'Gender': gender,
            'Age': age,
            'Tenure': tenure,
            'Balance': balance,
            'NumOfProducts': products,
            'HasCrCard': has_card,
            'IsActiveMember': active_member,
            'EstimatedSalary': salary,
        }
        
        # Encode and predict
        encoded = encode_features(user_row, pipeline['categorical_encoders'])
        label, proba = predict_customer(pipeline['model'], encoded)
        risk = customer_risk_level(proba)
        advice = customer_recommendations(proba)

        # Display prediction results
        st.markdown('---')
        st.subheader('Prediction Result')
        
        col1, col2, col3 = st.columns(3)
        col1.metric('Churn Probability', f'{proba:.1%}')
        col2.metric('Risk Level', risk)
        col3.metric('Predicted Outcome', 'Churn ⚠️' if label == 1 else 'Stay ✅')

        # Risk-based alerts
        if label == 1:
            if risk == 'High':
                st.error(f'🚨 **HIGH RISK** - This customer has a {proba:.1%} churn probability and requires immediate attention.')
            else:
                st.warning(f'⚠️ **MEDIUM RISK** - This customer has a {proba:.1%} churn probability and should be monitored.')
        else:
            st.success(f'✅ **LOW RISK** - This customer is likely to stay (churn probability: {proba:.1%}).')

        # Recommendations
        st.subheader('Recommended Actions')
        for i, item in enumerate(advice, 1):
            st.write(f'{i}. {item}')

        # Download report
        st.markdown('---')
        report = pd.DataFrame([user_row])
        report['Churn_Probability'] = proba
        report['Risk_Level'] = risk
        report['Predicted_Outcome'] = 'Churn' if label == 1 else 'Stay'
        csv = report.to_csv(index=False).encode('utf-8')
        st.download_button(
            label='📥 Download Prediction Report (CSV)',
            data=csv,
            file_name='churn_prediction_report.csv',
            mime='text/csv',
        )

        # Explainability section
        with st.expander('🔬 **View SHAP & LIME Explanations** (Click to expand)'):
            st.write('Understand **why** the model made this prediction using SHAP and LIME local explanations.')
            
            # SHAP explanation
            st.subheader('SHAP Local Contributions')
            st.write('Shows how each feature pushes the prediction toward churn (red) or away from churn (blue).')
            
            shap_row = pipeline['shap_explainer'].shap_values(encoded)[0]
            shap_df = pd.DataFrame({
                'Feature': pipeline['feature_names'],
                'SHAP Value': shap_row,
                'Abs SHAP': np.abs(shap_row),
            }).sort_values('Abs SHAP', ascending=False)

            # Display table
            st.dataframe(shap_df[['Feature', 'SHAP Value']].head(10).reset_index(drop=True))

            # SHAP bar chart
            fig_shap = px.bar(
                shap_df.head(10).sort_values('SHAP Value'),
                x='SHAP Value',
                y='Feature',
                orientation='h',
                color='SHAP Value',
                color_continuous_scale='RdBu',
                labels={'SHAP Value': 'SHAP Impact', 'Feature': 'Feature'},
                title='Top 10 Features by SHAP Impact',
            )
            fig_shap.update_layout(height=400)
            st.plotly_chart(fig_shap, width='stretch')

            # LIME explanation
            st.subheader('LIME Local Explanation')
            st.write('Local interpretable model-agnostic explanations for this specific prediction.')
            
            lime_df = explain_customer_lime(
                pipeline['lime_explainer'],
                pipeline['model'].predict_proba,
                encoded.values[0],
                num_features=5,
            )
            st.dataframe(lime_df)

            # Natural language summary
            st.subheader('📝 Key Drivers Summary')
            st.write('The top factors influencing this prediction:')
            for _, row in shap_df.head(5).iterrows():
                direction = '📈 **increases**' if row['SHAP Value'] > 0 else '📉 **decreases**'
                st.write(f"- **{row['Feature']}**: {direction} churn risk by **{abs(row['SHAP Value']):.4f}**")


def render_explainability(pipeline: dict):
    st.header('Explainability & Model Performance')
    st.write('Global insights into the model: feature importance, SHAP analysis, and evaluation metrics.')

    st.markdown('---')

    # XGBoost Feature Importance
    st.subheader('1. XGBoost Feature Importance')
    st.write('Built-in feature importance from the trained XGBoost model. Higher values indicate stronger predictive power.')
    
    importance = get_feature_importance(pipeline['model'], pipeline['feature_names'])
    fig_importance = px.bar(
        importance.sort_values('importance', ascending=True),
        x='importance',
        y='feature',
        orientation='h',
        labels={'importance': 'Feature Importance', 'feature': 'Feature'},
        title='XGBoost Feature Importance (Top 10)',
        color='importance',
        color_continuous_scale='Viridis',
    )
    fig_importance.update_layout(height=400)
    st.plotly_chart(fig_importance, width='stretch')

    st.markdown('---')

    # SHAP Feature Importance
    st.subheader('2. SHAP Feature Importance')
    st.write('Mean absolute SHAP values across the training dataset. Shows average impact of each feature on model predictions.')
    
    with st.spinner('Computing SHAP values (this may take a moment)...'):
        shap_values = get_shap_values(pipeline['shap_explainer'], pipeline['train'])
        shap_summary = pd.DataFrame({
            'Feature': pipeline['feature_names'],
            'Mean |SHAP|': np.abs(shap_values).mean(axis=0),
        }).sort_values('Mean |SHAP|', ascending=False)

    fig_shap = px.bar(
        shap_summary.sort_values('Mean |SHAP|', ascending=True),
        x='Mean |SHAP|',
        y='Feature',
        orientation='h',
        labels={'Mean |SHAP|': 'Mean |SHAP Value|', 'Feature': 'Feature'},
        title='SHAP Feature Importance (Top 10)',
        color='Mean |SHAP|',
        color_continuous_scale='RdYlGn_r',
    )
    fig_shap.update_layout(height=400)
    st.plotly_chart(fig_shap, width='stretch')

    # Expandable table of top SHAP features
    with st.expander('📊 View Top 10 SHAP Features'):
        st.dataframe(shap_summary.head(10).reset_index(drop=True), width='stretch')

    st.markdown('---')

    # Model Evaluation
    st.subheader('3. Model Evaluation Metrics')
    st.write('Performance metrics on the test dataset.')
    
    with st.spinner('Computing evaluation metrics...'):
        metrics = evaluate_model(pipeline['model'], pipeline['test'], pipeline['labels_test'])

    # Metrics row
    col1, col2, col3 = st.columns(3)
    col1.metric('Accuracy', f"{metrics['accuracy']:.2%}")
    col1.caption('Overall correctness of predictions')
    
    col2.metric('Churn Precision', f"{metrics['classification_report']['1']['precision']:.2%}")
    col2.caption('Of predicted churners, how many actually churned')
    
    col3.metric('Churn Recall', f"{metrics['classification_report']['1']['recall']:.2%}")
    col3.caption('Of actual churners, how many were caught')

    # Classification report
    st.subheader('Classification Report')
    st.write('Detailed per-class metrics (0 = Stay, 1 = Churn).')
    
    report_df = pd.DataFrame(metrics['classification_report']).transpose()
    report_df = report_df.round(3)
    st.dataframe(report_df, width='stretch')

    # Confusion matrix
    st.subheader('Confusion Matrix')
    st.write('Shows True Positives, True Negatives, False Positives, False Negatives.')
    
    conf_matrix = metrics['confusion_matrix']
    fig_conf = px.imshow(
        conf_matrix,
        x=['Predicted: Stay', 'Predicted: Churn'],
        y=['Actual: Stay', 'Actual: Churn'],
        color_continuous_scale='Blues',
        text_auto=True,
        labels={'x': 'Predicted', 'y': 'Actual', 'color': 'Count'},
        title='Confusion Matrix',
    )
    fig_conf.update_layout(height=400)
    st.plotly_chart(fig_conf, width='stretch')

    st.markdown('---')

    # Summary insights
    st.subheader('📈 Key Model Insights')
    
    top_feature = shap_summary.iloc[0]
    accuracy = metrics['accuracy']
    precision = metrics['classification_report']['1']['precision']
    recall = metrics['classification_report']['1']['recall']
    
    col1, col2 = st.columns(2)
    
    with col1:
        st.markdown(f"""
        **Model Performance:**
        - Overall accuracy: **{accuracy:.1%}**
        - Ability to catch churners: **{recall:.1%}**
        - False alarm rate: **{100 - precision * 100:.1%}**
        """)
    
    with col2:
        st.markdown(f"""
        **Most Important Features:**
        1. **{shap_summary.iloc[0]['Feature']}** - Impact: {shap_summary.iloc[0]['Mean |SHAP|']:.4f}
        2. **{shap_summary.iloc[1]['Feature']}** - Impact: {shap_summary.iloc[1]['Mean |SHAP|']:.4f}
        3. **{shap_summary.iloc[2]['Feature']}** - Impact: {shap_summary.iloc[2]['Mean |SHAP|']:.4f}
        """)


def main():
    st.set_page_config(
        page_title='Customer Churn AI',
        page_icon='📊',
        layout='wide',
    )

    pipeline = load_pipeline()
    raw = pipeline['raw']

    st.sidebar.title('Navigation')
    page = st.sidebar.radio('Choose a page', ['Overview', 'Prediction', 'Explainability'])

    st.sidebar.markdown('---')
    st.sidebar.markdown('### Churn AI')
    st.sidebar.markdown('Predict customer churn with XGBoost and explainability.')

    if page == 'Overview':
        render_overview(raw, pipeline)
    elif page == 'Prediction':
        render_prediction(raw, pipeline)
    else:
        render_explainability(pipeline)


if __name__ == '__main__':
    main()