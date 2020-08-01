import classifier

import testing.intent_predictor as pred

import eval.metrics as metrics

saved_model_dir = '/mount/arbeitsdaten/studenten1/team-lab-nlp/mandavsi_rileyic/saved_models/experiment_4'
y_pred, y_true = pred.load_model_and_predict_test_data(saved_model_dir)

confusion_matrix = metrics.get_confusion_matrix(y_true, y_pred)

print("Confusion Matrix :: ")
print(confusion_matrix)

plot_file_path = saved_model_dir+'/confusion_matrix_plot.png'
metrics.plot_confusion_matrix(confusion_matrix, "BiLSTM Classifier + Attention with ELMo", plot_file_path)
