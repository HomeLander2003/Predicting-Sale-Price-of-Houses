import numpy as np
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import ElasticNet,LinearRegression
from sklearn.svm import SVR
from sklearn.model_selection import train_test_split,GridSearchCV,cross_validate
from sklearn.metrics import mean_absolute_error,mean_squared_error,r2_score
from sklearn.pipeline import Pipeline
import streamlit as st
import os
import time
import logging

class EDA:
    
    
    def load(self):
        self.df=None
        self.file_path=r"AMES_Final_DF.csv"
        
        if os.path.isfile(self.file_path):
            
            try:
                self.df=pd.read_csv(self.file_path) 
                with st.status("Loading data...", expanded=True) as status:
                    st.write("Checking file path...")
                    time.sleep(1)
                    st.write("Reading CSV file...")
                    time.sleep(1)
                    status.update(label="Data loaded successfully!", state="complete")
            
            except Exception as e:
                
                st.error(e)
        
        else:
            st.error("Unable to Load Data..")
        
    def preprocessing(self):
        
        if self.file_path is not None:
            
        
            try: 
                self.df=pd.read_csv(self.file_path)                
                st.title("Data Set")
                st.write(self.df.head(5))
                
                percent_nan=self.df.isnull().sum().sum()/len(self.df)*100
                
                if percent_nan>0:
                    
                    self.df.dropna(inplace=True)
                    st.info("null values detected and dropped")
                
                else:
                    st.info("No Null values Found in dataset")
                    
                percent_dup=self.df.duplicated().sum()/len(self.df)*100
                
                if percent_dup>0:
                    
                    self.df.drop_duplicates(keep="first",inplace=True)
                    st.info("duplicate values detected and dropped")
                
                else:
                    st.info("No duplicate values Found in dataset")
                
                st.title("Columns")
                col=self.df.columns
                st.write(col)
                st.info(f"Total Length of columns : {len(col)}")
                st.markdown("<h4 style='color:green;'>To **Delete** any column click given button below </h4>",unsafe_allow_html=True)
                
                selected_cols = []
                for c in self.df.columns:
                    if st.checkbox(c, key=c):  
                        selected_cols.append(c)

                st.write("You selected:", selected_cols)
                
                st.markdown("""
                                <style>

                                div.stButton > button:first-child:hover {
                                    background-color: green;
                                    color: white;
                                    transform: scale(1.2);
                                    cursor: pointer;
                                }
                                </style>
                            """, unsafe_allow_html=True)
                                    
                    
                if st.button("Delete"):
                    
                    self.df.drop(columns=selected_cols ,axis=1, inplace=True)
                    st.success("Column Dropped Successfully.....")
                    st.markdown("<p2 style = 'color:green;'>Updated Columns</p2>",unsafe_allow_html=True)
                    st.info(f"Total Length of columns after dropping : {len(self.df.columns)}")
                    st.write(self.df.columns)
                    
                    
                st.title("Outliers")

                numeric_cols = self.df.select_dtypes(include=['number']).columns
                col_selected = st.selectbox("Choose a column to view boxplot", numeric_cols)
                st.warning("**Dropping Outliers may cause difficulty for the model because some outliers represent real houses (mansions, luxury homes)**..... ")

                fig, ax = plt.subplots(figsize=(10,6))
                sns.boxplot(x=self.df[col_selected], ax=ax)
                ax.set_title(f"Boxplot of {col_selected}")
                st.pyplot(fig)
                
                if st.button("Keep as-is"):
                  st.info("Data is in Original Form....")
               
     
                
            except Exception as e:
                 st.error(e)   
                        
        else:
            st.error("Unable to Clean data\n **(hint:Check Load Function)**")    
            
    def analysis(self):
        
        if self.df is not None:
            
            try:
                st.title("Descriptive Analysis")
                # st.write(self.df.describe())
                st.dataframe(self.df.describe().T.style.background_gradient(cmap="Blues"))
                
                st.subheader("Correlation Of Columns with Target Variable")
                
                self.gr1=self.df.corr()[["SalePrice"]].reset_index()
                st.write(self.gr1)

                st.subheader("Average Sale Price($) According to Yr-old")
                
                self.gr2=self.df.groupby("Yr Sold")["SalePrice"].mean()
                st.write(self.gr2)
                
                st.subheader("Average Sale Price($) According to Buit-Year")
                
                self.gr3=self.df.groupby("Year Built")["SalePrice"].mean()
                st.write(self.gr3)
                
                
            except Exception as e:
                st.error(e)
        
        else:
            st.error("Unable to Analyze data\n **(hint:Check Load/Preprocessing Function)**")   
            
    def insights(self):
        
        if self.df is not None:
            
            try:
                st.title("Correlations")
                
                fig1,ax1=plt.subplots(figsize=(20,15),dpi=200)
                col1=self.df.sample(50,axis=1)
                sns.heatmap(col1.corr(numeric_only=True),annot=False,ax=ax1,cmap="rocket")
                st.pyplot(fig1)
                
                st.subheader("Average Sale Price($) According to Yr-Sold")
                
                fig2,ax2=plt.subplots(figsize=(10,6),dpi=100)
                sns.barplot(x=self.df["Yr Sold"],y=self.df["SalePrice"],palette="coolwarm",ax=ax2)
                st.pyplot(fig2)
                
                st.subheader("Average Sale Price($) According to Built-year")
                
                fig2,ax2=plt.subplots(figsize=(10,6),dpi=50)
                gr=self.df.groupby("Year Built")["SalePrice"].mean()
                plt.barh(gr.index,gr.values)
                plt.xlabel("Sale Price")
                plt.ylabel("Built Year")
                plt.tight_layout()
                st.pyplot(fig2)
                
                
            except Exception as e:
                st.error(e)
        
        else:
            st.error("Unable to Visualize data\n **(hint:Check Load/Preprocessing Function)**")   

class ML(EDA):
    
    def ml(self):
        
        if self.df is not None:
            
            try:
                st.markdown("<h2 style = 'color:green;'> Predicting Sale Price of Houses</h2>",unsafe_allow_html=True)
               
                X=self.df.drop("SalePrice",axis=1)
                y=self.df["SalePrice"]
                
                st.subheader("Choose test Size/Random State")
                test_input1=st.selectbox("**choose test size**",[0.4,0.3,0.2,0.1])
                test_input2=st.selectbox("**choose random State**",[101,42,23])
                self.X_train, self.X_test, self.y_train, self.y_test = train_test_split(X, y, test_size=test_input1, random_state=test_input2)
                col1,col2=st.columns(2)
                
                with col1:
                    st.markdown("X-train")
                    st.write(self.X_train)
                with col2:
                    st.markdown("Y-train")
                    st.write(self.y_train)
                    
                st.info(f"Length of X_train and Correspondant Y_train={len(self.X_train)}")
                                
                models={
                    "Linear Regression":LinearRegression(),
                    "ElasticNet": ElasticNet(),
                    "SVR":SVR()
                }
                
                
                self.key_select=st.selectbox("Choose Model",list(models.keys()))
                self.value_select=models[self.key_select]
                
                operation1=Pipeline([("scaler",StandardScaler()),
                                     ("model", self.value_select)])
                
                cross_val=cross_validate(operation1,self.X_train,self.y_train,scoring=["neg_mean_absolute_error","neg_root_mean_squared_error","r2"]
                    ,cv=5)
                
                st.markdown("**Validation result (5-Fold CV)**")
                score = pd.DataFrame(cross_val)
                st.dataframe(score.T.style.background_gradient(cmap="Greens"))

                mean_scores = pd.DataFrame({
                "MAE": abs(score["test_neg_mean_absolute_error"]).mean(),
                "RMSE": abs(score["test_neg_root_mean_squared_error"]).mean(),
                "R²": score["test_r2"].mean()}, index=[self.key_select])

                st.markdown("**Validation Mean Scores (5-Fold CV)**")
                st.dataframe(mean_scores.T.style.background_gradient(cmap="Greens"))
                
                mean_r2 = score["test_r2"].mean()
                if mean_r2<0.6:
                    st.warning(f"{self.key_select} is not performing well on validation  **Tune it with GridSearch for Better Result**")
            
                st.markdown("<h3 style='color:green;'>Check Evaluation on Test Data </h3>", unsafe_allow_html=True)  

                if st.button("See Result"):
                    
                    if self.key_select=="Linear Regression":
                
                        operation1.fit(self.X_train, self.y_train)
                        pred =operation1.predict(self.X_test)

                        st.subheader("Linear Regression Test Performance")
                        st.text(f"MAE: {mean_absolute_error(self.y_test, pred)}")
                        st.text(f"RMSE: {np.sqrt(mean_squared_error(self.y_test, pred))}")
                        st.text(f"R²: {r2_score(self.y_test, pred)}")

                    elif self.key_select == "ElasticNet":
                        
                        grid_model1=self
                        operation1.fit(self.X_train, self.y_train)
                        pred =operation1.predict(self.X_test)

                        st.subheader("ElasticNet Test Performance")
                        st.text(f"MAE: {mean_absolute_error(self.y_test, pred)}")
                        st.text(f"RMSE: {np.sqrt(mean_squared_error(self.y_test, pred))}")
                        st.text(f"R²: {r2_score(self.y_test, pred)}")

                    elif self.key_select == "SVR":
                        operation1.fit(self.X_train, self.y_train)
                        pred = operation1.predict(self.X_test)

                        st.subheader("SVR Test Performance")
                        st.text(f"MAE: {mean_absolute_error(self.y_test, pred)}")
                        st.text(f"RMSE: {np.sqrt(mean_squared_error(self.y_test, pred))}")
                        st.text(f"R²: {r2_score(self.y_test, pred)}")
                    
            
            except Exception as e:
                
                st.error(e)
                
        else:
            st.error("Unable to perform Model Evaluation **(hint:Check Load/Preprocessing Function)**")
        
   
class stream(ML):
    
    def load_data(self):
        
        self.load()
    
    def run_clean(self):
        self.load()
        self.preprocessing() 
        
    def run_analysis(self):
        self.load()
        self.analysis()
        
    def run_insights(self):
        self.load()
        self.insights()
    
    def run_ML(self):
        self.load()
        self.ml()
    
        
    def app(self):
        
        st.sidebar.title("Data Set Link")
        st.sidebar.link_button("click Here","https://github.com/HomeLander2003/Predicting-Sale-Price-of-Houses/blob/main/DATA/AMES_Final_DF.csv")
        
        
        st.sidebar.title("Choose Options")
        
        options={
            "Preprocessing":self.run_clean,
            "Analysis":self.run_analysis,
            "Insights":self.run_insights,
            "Model training and evaluation":self.run_ML,
            "Deployment":self.run_clean
        }
        opt_name=st.sidebar.selectbox("Choose Option",(list(options.keys())))
        
        select_opt=options[opt_name]
        select_opt()
        
        
str=stream()
str.app()


                
            
        

        
