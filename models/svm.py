from sklearn.svm import SVC
from sklearn.metrics import classification_report, accuracy_score, confusion_matrix
from sklearn.model_selection import cross_val_score

def trainSVM(X_train, X_test, y_train, y_test):
    model = SVC(random_state=42)  
    model.fit(X_train, y_train)
    scores = cross_val_score(model,X_train, y_train, cv=5)
    scoresFolds = [(i + 1, score) for i, score in enumerate(scores)]
    
    cv_report = {
        "Cross-Validation Scores": scoresFolds,
        "Mean CV Score": scores.mean(),
        "Standard Deviation of CV Score": scores.std()
    }

    y_pred = model.predict(X_test)

    accuracy = accuracy_score(y_test, y_pred)
    report = classification_report(y_test, y_pred, output_dict=True)
    cm = confusion_matrix(y_test, y_pred)
    return model, accuracy, report, cv_report, cm