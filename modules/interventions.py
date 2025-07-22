def recommend_intervention(risk_score):
    if risk_score > 0.75:
        return "High Risk - Immediate Counselling Required"
    elif risk_score > 0.5:
        return "Medium Risk - Monitor & Assign Mentor"
    else:
        return "Low Risk - Encourage Participation"
