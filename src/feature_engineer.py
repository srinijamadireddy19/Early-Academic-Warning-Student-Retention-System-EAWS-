import pandas as pd
import numpy as np

class FeatureEngineer:
    def __init__(self,data):
        self.df = data

    def create_features(self):
        self.create_at_risk()
        self.create_academic_level()
        self.create_attendance_risk()
        self.create_sleep_risk()
        self.create_workload_risk()
        self.create_engagement_risk()

        print("Feature engineering completed successfully.")

    def create_at_risk(self):
        self.df['at_risk'] = self.df['final_grade'].isin(['D','F']).astype(int)

    def create_academic_level(self):
        def academic_momentum(grade):
            if grade>=80:
                return 'excellent'
            elif grade>=60:
                return 'average'
            else:
                return 'poor'

        self.df['academic_level'] = self.df['final_exam_score'].apply(academic_momentum)

    def create_attendance_risk(self):
        def attendance_risk(attendance):
            if attendance<60:
                return 'Critical'
            elif attendance<75:
                return 'Moderate'
            else:
                return 'Healthy'

        self.df['attendance_risk_level'] = self.df['attendance_percent'].apply(attendance_risk)


    def create_sleep_risk(self):
        self.df['sleep_deviation'] = abs(self.df['sleep_hours'] - 7.5)
        self.df['deviation_bin'] = self.df['sleep_deviation'].round()

    def create_workload_risk(self):
        threshold = 7.0
        self.df['high_workload_low_sleep'] = (
            (self.df['part_time_job'].astype(str).str.strip().str.lower() == 'yes') & 
            (self.df['sleep_hours'] < threshold)
        ).astype(int)

    def create_engagement_risk(self):
        extracurricular_scaled = self.df['extracurricular_activities'].map({'Yes': 1.0, 'No': 0.0})

        attendance_min_max = (self.df['attendance_percent'] - self.df['attendance_percent'].min()) / (self.df['attendance_percent'].max() - self.df['attendance_percent'].min())
        study_min_max = (self.df['study_time_hours'] - self.df['study_time_hours'].min()) / (self.df['study_time_hours'].max() - self.df['study_time_hours'].min())

        self.df['engagement_score'] = (attendance_min_max + study_min_max + extracurricular_scaled) / 3



    