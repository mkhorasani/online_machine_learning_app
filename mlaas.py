import streamlit as st
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn import metrics
import plotly.express as px
import plotly.graph_objects as go
import plotly.figure_factory as ff
import base64
import os

def confusion_matrix_plot(data):
    z = data.tolist()[::-1]
    x = ['Negative','Positive']
    y = ['Positive','Negative']
    z_text = z

    fig = ff.create_annotated_heatmap(z, x, y, annotation_text=z_text, text=z,hoverinfo='text',colorscale='Blackbody')
    fig.update_layout(font_family="IBM Plex Sans")

    st.write(fig)

def roc_plot(data):
    fig = px.line(data, x="False Positive", y="True Positive")
    fig.update_layout(font_family="IBM Plex Sans")

    st.write(fig)

def download(df,filename):
    csv = df.to_csv(index=False)
    b64 = base64.b64encode(csv.encode()).decode()
    href = (f'<a href="data:file/csv;base64,{b64}" download="%s.csv">Download csv file</a>' % (filename))
    return href

def file_upload(name):
    uploaded_file = st.sidebar.file_uploader('%s' % (name),key='%s' % (name),accept_multiple_files=False)
    content = False
    if uploaded_file is not None:
        try:
            uploaded_df = pd.read_csv(uploaded_file)
            content = True
            return content, uploaded_df
        except:
            try:
                uploaded_df = pd.read_excel(uploaded_file)
                content = True
                return content, uploaded_df
            except:
                st.error('Please ensure file is .csv or .xlsx format and/or reupload file')
                return content, None
    else:
        return content, None

def lr_main(engine):

    st.sidebar.subheader('Training Dataset')
    status, df = file_upload('Please upload a training dataset')

    if status == True:
        col_names = list(df)

        st.title('Training')
        st.subheader('Parameters')
        col1, col2, col3 = st.columns((3,3,2))

        with col1:
            feature_cols = st.multiselect('Please select features',col_names)
        with col2:
            label_col = st.selectbox('Please select label',col_names)
        with col3:
            test_size = st.number_input('Please enter test size',0.01,0.99,0.25,0.05)

        with st.expander('Advanced Parameters'):
            col2_1, col2_2 = st.columns(2)
            with col2_1:
                penalty = st.selectbox('Penalty',['l2','l1','elasticnet','none'])
                tol = st.number_input('Tolerance (1e-4)',value=1)/10000
                fit_intercept = st.radio('Intercept',[True,False])
                class_weight = st.radio('Class weight',[None,'balanced'])
                solver = st.selectbox('Solver',['lbfgs','newton-cg','liblinear','sag','saga'])
                multi_class = st.selectbox('Multi class',['auto','ovr','multinomial'])
                warm_start = st.radio('Warm start',[False,True])
            with col2_2:
                dual = st.radio('Dual or primal formulation',[False,True])
                C = st.number_input('Inverse regularization strength',0.0,99.0,1.0,0.1)
                intercept_scaling = st.number_input('Intercept scaling',0.0,99.0,1.0,0.1)
                random_state = st.radio('Random state',[None,'Custom'])
                if random_state == 'Custom':
                    random_state = st.number_input('Custom random state',0,99,1,1)
                max_iter = st.number_input('Maximum iterations',0,100,100,1)
                verbose = st.number_input('Verbose',0,99,0,1)
                l1_ratio = st.radio('L1 ratio',[None,'Custom'])
                if l1_ratio == 'Custom':
                    l1_ratio = st.number_input('Custom l1 ratio',0.0,1.0,1.0,0.01)

            st.markdown('For further information please refer to ths [link](https://scikit-learn.org/stable/modules/generated/sklearn.linear_model.LogisticRegression.html)')

        try:
            X = df[feature_cols]
            y = df[label_col]
            X_train,X_test,y_train,y_test=train_test_split(X,y,test_size=test_size,random_state=0)
            logreg = LogisticRegression(penalty=penalty, dual=dual, tol=tol, C=C,
                                        fit_intercept=fit_intercept, intercept_scaling=intercept_scaling,
                                        class_weight=class_weight, random_state=random_state, solver=solver,
                                        max_iter=max_iter, multi_class=multi_class, verbose=verbose,
                                        warm_start=warm_start, l1_ratio=l1_ratio)
            logreg.fit(X_train,y_train)
            y_pred = logreg.predict(X_test)
            cnf_matrix = metrics.confusion_matrix(y_test, y_pred)

            st.subheader('Confusion Matrix')
            confusion_matrix_plot(cnf_matrix)

            accuracy = metrics.accuracy_score(y_test, y_pred)
            precision = metrics.precision_score(y_test, y_pred)
            recall = metrics.recall_score(y_test, y_pred)
            f1 = metrics.f1_score(y_test, y_pred)

            st.subheader('Metrics')
            col2_1, col2_2, col2_3, col2_4 = st.columns(4)

            with col2_1:
                st.info('Accuracy: **%s**' % (round(accuracy,3)))
            with col2_2:
                st.info('Precision: **%s**' % (round(precision,3)))
            with col2_3:
                st.info('Recall: **%s**' % (round(recall,3)))
            with col2_4:
                st.info('F1 Score: **%s**' % (round(f1,3)))

            y_pred_proba = logreg.predict_proba(X_test)[::,1]
            fpr, tpr, _ = metrics.roc_curve(y_test,  y_pred_proba)
            roc_data = pd.DataFrame([])
            roc_data['True Positive'] =  tpr
            roc_data['False Positive'] = fpr
            st.subheader('ROC Curve')
            roc_plot(roc_data)
            auc = metrics.roc_auc_score(y_test, y_pred_proba)
            st.info('Area Under Curve: **%s**' % (round(auc,3)))

            st.sidebar.subheader('Test Dataset')
            status_test, df_test = file_upload('Please upload a test dataset')

            if status_test == True:
                try:
                    st.title('Testing')
                    X_test_test = df_test[feature_cols]
                    y_pred_test = logreg.predict(X_test_test)

                    X_pred = df_test.copy()
                    X_pred[label_col] = y_pred_test
                    X_pred = X_pred.sort_index()

                    st.subheader('Predicted Labels')
                    st.write(X_pred)
                    st.markdown(download(X_pred,'Logistic Regression Classifier - Predicted Labels'), unsafe_allow_html=True)
                except:
                    st.warning('Please upload a test dataset with the same feature set as the training dataset')

            elif status_test == False:
                st.sidebar.warning('Please upload a test dataset')

        except:
            st.warning('Please select at least one feature, a suitable binary label and appropriate advanced paramters')

    elif status == False:
        st.sidebar.warning('Please upload a training dataset')
